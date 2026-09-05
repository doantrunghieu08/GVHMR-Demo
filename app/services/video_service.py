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
from hmr4d.utils.preproc import Tracker
from tools.demo.demo import (
    get_paths_for_person,
    run_preprocess,
    load_data_dict,
    render_incam,
    render_global,
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
    Xử lý tác vụ video ngầm cho tối đa 2 người chính trong video.
    Mỗi người sẽ có output video riêng biệt.
    """
    model = get_model()
    update_job(job_id, {
        "status": "PROCESSING",
        "progress": "Tạo cấu hình và chuẩn bị video",
        "result": None
    })

    try:
        # 1. Tạo cấu hình Hydra động cho video
        cfg = get_api_config(video_path, static_cam, use_dpvo)

        update_job(job_id, {"progress": "Extracting FPS"})
        actual_fps = extract_fps_and_copy(video_path, cfg.video_path)

        # Tiêm thông số fps thực tế vào config
        OmegaConf.set_struct(cfg, False)
        cfg.video_fps = actual_fps
        OmegaConf.set_struct(cfg, True)

        # 2. Track tất cả người (tối đa 2) — chạy 1 lần duy nhất
        update_job(job_id, {"progress": "Đang tracking người trong video (YOLO)..."})
        tracker = Tracker()
        all_bbx_xyxy = tracker.get_n_tracks(cfg.video_path, n=2)  # List[Tensor(F, 4)]
        del tracker
        num_people = len(all_bbx_xyxy)
        print(f"[Job {job_id}] Phát hiện {num_people} người trong video.")

        if num_people == 0:
            raise RuntimeError("Không phát hiện được người nào trong video.")

        # 3. Xử lý từng người
        output_video_urls = []
        result_file_paths = []

        for person_id, bbx_xyxy in enumerate(all_bbx_xyxy):
            print(f"\n[Job {job_id}] ===== Xử lý Người {person_id} =====")
            paths_p = get_paths_for_person(cfg, person_id)

            # --- Tiền xử lý ---
            update_job(job_id, {"progress": f"[Người {person_id}] Tiền xử lý (tracking/keypoints/features)"})
            run_preprocess(cfg, bbx_xyxy, person_id=person_id)

            # --- Load data ---
            data = load_data_dict(cfg, person_id=person_id)

            # Lọc sạch NaN ở dữ liệu đầu vào nếu có
            for k in ["kp2d", "bbx_xys", "cam_angvel", "f_imgseq"]:
                if k in data and isinstance(data[k], torch.Tensor):
                    data[k] = torch.nan_to_num(data[k], nan=0.0)

            # --- Predict ---
            update_job(job_id, {"progress": f"[Người {person_id}] Đang chạy mô hình GVHMR..."})
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
            torch.save(pred, paths_p["hmr4d_results"])

            # --- Render ---
            update_job(job_id, {"progress": f"[Người {person_id}] Đang render video mesh 3D..."})
            render_incam(cfg, person_id=person_id)
            render_global(cfg, person_id=person_id)

            horiz_path = paths_p["incam_global_horiz_video"]
            merge_videos_horizontal(
                [paths_p["incam_video"], paths_p["global_video"]],
                horiz_path
            )

            # --- Lưu kết quả ---
            result_filename = Path(horiz_path).name
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            final_video_path = OUTPUT_DIR / f"{job_id}_{result_filename}"
            shutil.copy(horiz_path, final_video_path)

            output_video_urls.append(f"/api/v1/download/{final_video_path.name}")
            result_file_paths.append(str(paths_p["hmr4d_results"]))

        # 4. Hoàn thành
        update_job(job_id, {
            "status": "COMPLETED",
            "progress": f"Hoàn thành — {num_people} người đã xử lý",
            "result": {
                "num_people": num_people,
                "output_video_urls": output_video_urls,
                "result_file_paths": result_file_paths,
                # Giữ tương thích ngược với client cũ (lấy người đầu tiên)
                "output_video_url": output_video_urls[0] if output_video_urls else None,
                "result_file_path": result_file_paths[0] if result_file_paths else None,
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
