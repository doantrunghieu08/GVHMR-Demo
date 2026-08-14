import shutil
import ffmpeg
import torch
from pathlib import Path
from omegaconf import OmegaConf

from app.config import OUTPUT_DIR
from app.services.job_service import update_job
from app.services.model_service import get_api_config, get_model
from hmr4d.utils.net_utils import detach_to_cpu
from hmr4d.utils.video_io_utils import merge_videos_horizontal
from tools.demo.demo import (
    run_preprocess,
    load_data_dict,
    render_incam,
    render_global
)

def extract_fps_and_copy(input_path: Path, output_path: Path):
    """
    Kiểm tra và trả về FPS gốc của video, sau đó copy video sang output_path.
    """
    fps = 30.0
    try:
        probe = ffmpeg.probe(str(input_path))
        video_stream = None
        for stream in probe.get("streams", []):
            if stream["codec_type"] == "video":
                video_stream = stream
                break
        if video_stream and 'avg_frame_rate' in video_stream:
            num, den = video_stream['avg_frame_rate'].split('/')
            fps = float(num) / float(den)
    except Exception as e:
        print(f"Loi doc FPS: {e}")
        
    shutil.copy(input_path, output_path)
    return fps

def process_video_task(job_id: str, video_path: Path, static_cam: bool, use_dpvo: bool):
    """
    Xử lý tác vụ video ngầm
    """
    model = get_model()
    update_job(job_id, {
        "status": "PROCESSING", 
        "progress": "Tiền xử lý video (tracking/keypoints)",
        "result": None
    })
    
    try:
        # 1. Tạo cấu hình Hydra động cho video
        cfg = get_api_config(video_path, static_cam, use_dpvo)

        update_job(job_id, {"progress": "Extracting FPS"})
        actual_fps = extract_fps_and_copy(video_path, cfg.video_path)
        
        # Tiêm thông số fps thực tế vào config để hệ thống tự động nhận
        OmegaConf.set_struct(cfg, False)
        cfg.video_fps = actual_fps
        OmegaConf.set_struct(cfg, True)

        update_job(job_id, {"progress": "Running preprocessing"})
        run_preprocess(cfg)

        # 3. Gom dữ liệu vào mô hình dự đoán
        data = load_data_dict(cfg)

        # Lọc sạch NaN ở dữ liệu đầu vào nếu có
        for k in ["kp2d", "bbx_xys", "cam_angvel", "f_imgseq"]:
            if k in data and isinstance(data[k], torch.Tensor):
                data[k] = torch.nan_to_num(data[k], nan=0.0)

        with torch.no_grad():
            pred = model.predict(data, static_cam=cfg.static_cam)
        pred = detach_to_cpu(pred)

        # Lọc sạch NaN trong dict dự đoán trước khi lưu và render
        def clean_nans(d):
            if isinstance(d, dict):
                return {k: clean_nans(v) for k, v in d.items()}
            elif isinstance(d, torch.Tensor):
                return torch.nan_to_num(d, nan=0.0)
            return d

        pred = clean_nans(pred)
        torch.save(pred, cfg.paths.hmr4d_results)

        # 4. Render kết quả trả về 
        update_job(job_id, {"progress": "Danh render video mesh 3d"})
        
        render_incam(cfg)
        render_global(cfg)

        merge_videos_horizontal([cfg.paths.incam_video, cfg.paths.global_video], cfg.paths.incam_global_horiz_video)

        result_filename = Path(cfg.paths.incam_global_horiz_video).name
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        final_video_path = OUTPUT_DIR / f"{job_id}_{result_filename}"
        shutil.copy(cfg.paths.incam_global_horiz_video, final_video_path)

        update_job(job_id, {
            "status": "COMPLETED",
            "progress": "Hoàn thành",
            "result": {
                "output_video_url": f"/api/v1/download/{final_video_path.name}",
                "result_file_path": str(cfg.paths.hmr4d_results)
            }
        })
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Task {job_id} failed with error:\n")
        traceback.print_exc()
        update_job(job_id, {
            "status": "FAILED",
            "progress": str(e),
            "result": None
        })
