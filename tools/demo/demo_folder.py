import argparse
import os
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import hydra
from hydra import initialize_config_module, compose
from hydra.core.global_hydra import GlobalHydra

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.pylogger import Log
from hmr4d.utils.video_io_utils import get_video_lwh
from hmr4d.utils.preproc import Tracker
from hmr4d.utils.net_utils import detach_to_cpu
from tools.demo.demo import (
    get_paths_for_person,
    precompute_vitpose,
    run_preprocess,
    load_data_dict,
    render_incam,
    render_global,
    merge_videos_horizontal,
)

_HYDRA_INITIALIZED = False


def load_gt_bboxes(video_path: Path):
    """
    Tìm và nạp GT bboxes từ processed_data/bbox/<cam_name>.
    Trả về dict {'aria01': Tensor(F, 4), 'aria02': Tensor(F, 4)} hoặc None nếu không tìm thấy.
    """
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


def get_cfg_for_cam(video_path: Path, output_root: Path, static_cam: bool = True, use_dpvo: bool = False):
    global _HYDRA_INITIALIZED
    if not _HYDRA_INITIALIZED:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        initialize_config_module(version_base="1.3", config_module="hmr4d.configs")
        _HYDRA_INITIALIZED = True

    cam_name = video_path.parent.name
    out_dir = output_root / cam_name

    overrides = [
        f"video_name={cam_name}",
        f"static_cam={str(static_cam).lower()}",
        f"verbose=False",
        f"use_dpvo={str(use_dpvo).lower()}",
        f"output_root={str(output_root).replace(chr(92), '/')}",
    ]
    register_store_gvhmr()
    cfg = compose(config_name="demo", overrides=overrides)

    OmegaConf.set_struct(cfg, False)
    cfg.output_dir = str(out_dir)
    cfg.preprocess_dir = str(out_dir / "preprocess")
    cfg.video_path = str(out_dir / "0_input_video.mp4")
    OmegaConf.set_struct(cfg, True)

    out_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def clean_nans(d):
    if isinstance(d, dict):
        return {k: clean_nans(v) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        return torch.nan_to_num(d, nan=0.0)
    return d


def process_single_video(video_path: Path, output_root: Path, model, tracker, static_cam: bool, use_dpvo: bool, skip_render: bool):
    cam_name = video_path.parent.name
    Log.info(f"\n{'='*60}\n[Video]: {video_path.name} | Camera: {cam_name}\n{'='*60}")

    cfg = get_cfg_for_cam(video_path, output_root, static_cam=static_cam, use_dpvo=use_dpvo)

    # 1. Copy video sang thư mục output nếu chưa có
    if not Path(cfg.video_path).exists() or Path(cfg.video_path).stat().st_size != video_path.stat().st_size:
        shutil.copy(video_path, cfg.video_path)

    # 2. Lấy bboxes: ưu tiên nạp GT bbox (aria01, aria02)
    gt_dict = load_gt_bboxes(video_path)
    if gt_dict is not None:
        people_tracks = list(gt_dict.items())  # [('aria01', bbox1), ('aria02', bbox2)]
        Log.info(f"[{cam_name}] Tìm thấy GT bbox: gán nhãn {[p[0] for p in people_tracks]}")
    else:
        Log.info(f"[{cam_name}] Không có GT bbox. Chạy YOLO tracker...")
        all_bbx = tracker.get_n_tracks(cfg.video_path, n=2)
        people_tracks = [(f"p{i}", bbx) for i, bbx in enumerate(all_bbx)]

    if len(people_tracks) == 0:
        Log.warn(f"[{cam_name}] Không tìm thấy người nào trong video!")
        return

    # Precompute ViTPose
    precompute_vitpose(cfg, people_tracks)

    # 3. Chạy dự đoán GVHMR cho từng người
    saved_files = []
    for idx, (person_label, bbx_xyxy) in enumerate(people_tracks):
        paths_p = get_paths_for_person(cfg, person_label)
        result_file = Path(paths_p["hmr4d_results"])

        if not result_file.exists():
            Log.info(f"[{cam_name}] Preprocessing {person_label}...")
            run_preprocess(cfg, bbx_xyxy, person_id=person_label)
            data = load_data_dict(cfg, person_id=person_label)

            for k in ["kp2d", "bbx_xys", "cam_angvel", "f_imgseq"]:
                if k in data and isinstance(data[k], torch.Tensor):
                    data[k] = torch.nan_to_num(data[k], nan=0.0)

            Log.info(f"[{cam_name}] Predicting GVHMR cho {person_label}...")
            with torch.no_grad():
                pred = model.predict(data, static_cam=cfg.static_cam)
            pred = detach_to_cpu(pred)
            pred = clean_nans(pred)
            pred["subject"] = person_label
            pred["cam"] = cam_name

            torch.save(pred, result_file)
            del data, pred
            torch.cuda.empty_cache()
        else:
            Log.info(f"[{cam_name}] Kết quả đã có sẵn tại: {result_file}")

        saved_files.append(str(result_file))

        # Lưu thêm alias _p{idx}.pt để tương thích tối đa với downstream scripts
        alias_file = Path(cfg.output_dir) / f"hmr4d_results_p{idx}.pt"
        if not alias_file.exists():
            shutil.copy(result_file, alias_file)

        # 4. Render video nếu được bật (mặc định tắt theo yêu cầu)
        if not skip_render:
            render_incam(cfg, person_id=person_label)
            render_global(cfg, person_id=person_label)
            horiz_path = paths_p["incam_global_horiz_video"]
            if not Path(horiz_path).exists():
                merge_videos_horizontal(
                    [paths_p["incam_video"], paths_p["global_video"]],
                    horiz_path
                )

    # Đồng bộ sang thư mục theo cấu trúc tương đối nếu khác output/<cam_name>
    try:
        rel_parent = video_path.relative_to(Path("input")).parent
        nested_out = output_root / rel_parent
        if nested_out != Path(cfg.output_dir) and not nested_out.exists():
            nested_out.parent.mkdir(parents=True, exist_ok=True)
            # Tạo junction hoặc copy
            shutil.copytree(cfg.output_dir, nested_out, dirs_exist_ok=True)
    except Exception:
        pass

    Log.info(f"[{cam_name}] Hoàn thành -> Kết quả: {saved_files}")


def main():
    parser = argparse.ArgumentParser(description="Chạy GVHMR batch trên toàn bộ thư mục input")
    parser.add_argument("-f", "--folder", type=str, default="input", help="Thư mục input gốc (mặc định: input)")
    parser.add_argument("-d", "--output_root", type=str, default="output", help="Thư mục output gốc (mặc định: output)")
    parser.add_argument("-s", "--static_cam", action="store_true", default=True, help="Camera tĩnh (mặc định: True cho exo)")
    parser.add_argument("--use_dpvo", action="store_true", default=False, help="Dùng DPVO nếu camera di chuyển")
    parser.add_argument("--render", action="store_true", default=False, help="Render video 3D mesh (mặc định: False để tăng tốc)")
    args = parser.parse_args()

    input_folder = Path(args.folder)
    output_root = Path(args.output_root)
    skip_render = not args.render

    assert input_folder.exists(), f"Thư mục input không tồn tại: {input_folder}"

    # Quét toàn bộ video mp4 (dùng set để tránh trùng lặp trên Windows)
    all_found = list(input_folder.rglob("*.mp4")) + list(input_folder.rglob("*.MP4"))
    seen = set()
    mp4_paths = []
    for p in all_found:
        resolved = str(p.resolve()).lower()
        if resolved not in seen:
            seen.add(resolved)
            mp4_paths.append(p)
    mp4_paths = sorted(mp4_paths)
    # Loại trừ video tạm trong preprocess hoặc temp
    mp4_paths = [p for p in mp4_paths if "temp" not in p.parts and "preprocess" not in p.parts and not p.name.startswith("0_input_video")]

    Log.info(f"Tìm thấy {len(mp4_paths)} video cần xử lý trong: {input_folder}")
    for p in mp4_paths:
        Log.info(f"  - [{p.parent.name}] {p}")

    if not mp4_paths:
        return

    # Tải model vào GPU 1 lần duy nhất
    Log.info("Đang nạp GVHMR model vào GPU...")
    from app.services.model_service import load_model_into_gpu, get_model
    load_model_into_gpu()
    model = get_model()
    tracker = Tracker()

    Log.info(f"\nBắt đầu chạy batch (skip_render={skip_render}, static_cam={args.static_cam})...\n")
    for mp4_path in tqdm(mp4_paths, desc="Xử lý danh sách video"):
        try:
            process_single_video(
                mp4_path,
                output_root=output_root,
                model=model,
                tracker=tracker,
                static_cam=args.static_cam,
                use_dpvo=args.use_dpvo,
                skip_render=skip_render
            )
        except Exception as e:
            import traceback
            Log.error(f"Lỗi khi xử lý {mp4_path}: {e}")
            traceback.print_exc()

    Log.info("\n=== TOÀN BỘ QUÁ TRÌNH BATCH ĐÃ HOÀN THÀNH ===")


if __name__ == "__main__":
    main()
