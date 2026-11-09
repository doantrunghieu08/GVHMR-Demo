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
    precompute_vitpose,
    run_preprocess,
    load_data_dict,
    render_incam,
    render_global,
)


def load_gt_bboxes(video_path: Path):
    """
    Tìm và nạp GT bboxes từ processed_data/bbox/<cam_name>.
    Trả về dict {'aria01': Tensor(F, 4), 'aria02': Tensor(F, 4)} hoặc None nếu không tìm thấy.
    """
    import numpy as np
    from collections import defaultdict

    cam_name = video_path.parent.name
    search_dirs = [
        video_path.parent.parent.parent / "processed_data" / "bbox" / cam_name,
        Path("input/MMA/processed_data/bbox") / cam_name,
        Path("dataset") / cam_name,
    ]
    gt_dir = None
    for d in search_dirs:
        if d.exists() and d.is_dir():
            gt_dir = d
            break

    if gt_dir is None:
        matches = list(Path("input").glob(f"**/processed_data/bbox/{cam_name}"))
        if matches:
            gt_dir = matches[0]

    if gt_dir is None:
        return None

    files = sorted([f for f in gt_dir.glob("*.npy") if "(" not in f.name])
    if not files:
        return None

    gt_dict = defaultdict(list)
    for f in files:
        data = np.load(f, allow_pickle=True).item()
        for k, v in data.items():
            gt_dict[k].append(v)

    return {k: torch.from_numpy(np.array(v)).float() for k, v in gt_dict.items()}


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
    Bỏ qua bước render video để tăng tốc lấy kết quả, gán nhãn aria01/aria02 dựa theo GT nếu có.
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

        # 2. Lấy bboxes: ưu tiên nạp GT (aria01, aria02), nếu không có thì chạy YOLO tracker
        gt_dict = load_gt_bboxes(video_path)
        if gt_dict is not None:
            people_tracks = list(gt_dict.items())  # [('aria01', bbox1), ('aria02', bbox2)]
            print(f"[Job {job_id}] Đã nhận diện nhãn GT: {[p[0] for p in people_tracks]}")
        else:
            update_job(job_id, {"progress": "Đang tracking người trong video (YOLO)..."})
            tracker = Tracker()
            all_bbx_xyxy = tracker.get_n_tracks(cfg.video_path, n=2)
            del tracker
            people_tracks = [(f"p{i}", bbx) for i, bbx in enumerate(all_bbx_xyxy)]

        num_people = len(people_tracks)
        print(f"[Job {job_id}] Xử lý {num_people} đối tượng: {[p[0] for p in people_tracks]}")

        if num_people == 0:
            raise RuntimeError("Không phát hiện được người nào trong video.")

        update_job(job_id, {"progress": "Đang chạy ViTPose..."})
        precompute_vitpose(cfg, people_tracks)

        # 3. Xử lý từng người
        output_video_urls = []
        result_file_paths = []

        for person_label, bbx_xyxy in people_tracks:
            print(f"\n[Job {job_id}] ===== Xử lý {person_label} =====")
            paths_p = get_paths_for_person(cfg, person_label)

            # --- Tiền xử lý ---
            update_job(job_id, {"progress": f"[{person_label}] Tiền xử lý (tracking/keypoints/features)"})
            run_preprocess(cfg, bbx_xyxy, person_id=person_label)

            # --- Load data ---
            data = load_data_dict(cfg, person_id=person_label)

            # Lọc sạch NaN ở dữ liệu đầu vào nếu có
            for k in ["kp2d", "bbx_xys", "cam_angvel", "f_imgseq"]:
                if k in data and isinstance(data[k], torch.Tensor):
                    data[k] = torch.nan_to_num(data[k], nan=0.0)

            # --- Predict ---
            update_job(job_id, {"progress": f"[{person_label}] Đang chạy mô hình GVHMR..."})
            with torch.no_grad():
                pred = model.predict(data, static_cam=cfg.static_cam)
            pred = detach_to_cpu(pred)

            # Lọc sạch NaN trong dict dự đoán trước khi lưu
            def clean_nans(d):
                if isinstance(d, dict):
                    return {k: clean_nans(v) for k, v in d.items()}
                elif isinstance(d, torch.Tensor):
                    return torch.nan_to_num(d, nan=0.0)
                return d

            pred = clean_nans(pred)
            pred["subject"] = person_label
            torch.save(pred, paths_p["hmr4d_results"])

            # Lưu file kết quả .pt ra OUTPUT_DIR (bỏ qua render video)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            saved_pt = OUTPUT_DIR / f"{job_id}_{person_label}_results.pt"
            shutil.copy(paths_p["hmr4d_results"], saved_pt)

            result_file_paths.append(str(paths_p["hmr4d_results"]))
            del data, pred
            torch.cuda.empty_cache()

        # 4. Hoàn thành
        update_job(job_id, {
            "status": "COMPLETED",
            "progress": f"Hoàn thành — {num_people} đối tượng đã xử lý ({', '.join(p[0] for p in people_tracks)})",
            "result": {
                "num_people": num_people,
                "labels": [p[0] for p in people_tracks],
                "output_dir": str(cfg.output_dir),
                "output_video_urls": output_video_urls,
                "result_file_paths": result_file_paths,
                # Giữ tương thích ngược với client cũ
                "output_video_url": None,
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
