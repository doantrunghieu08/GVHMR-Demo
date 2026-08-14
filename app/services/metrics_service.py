import torch
import numpy as np
from pathlib import Path
from fastapi import HTTPException
from hmr4d.utils.eval.eval_utils import (
    batch_align_by_pelvis,
    batch_compute_similarity_transform_torch,
    compute_jpe,
    convert_joints22_to_24
)
from app.services.job_service import get_job

def evaluate_metrics_logic(request):
    pred_data = request.pred_j3d

    # Nếu truyền job_id, tự động tìm pred_j3d trong kết quả công việc đã thực hiện
    if request.job_id:
        job_info = get_job(request.job_id)
        if not job_info:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy tác vụ với job_id: {request.job_id}")
        
        if job_info.get("status") != "COMPLETED":
            raise HTTPException(status_code=400, detail=f"Tác vụ {request.job_id} chưa hoàn thành (trạng thái: {job_info.get('status')})")
        
        result_info = job_info.get("result", {})
        result_file_path = result_info.get("result_file_path") if result_info else None
        if not result_file_path or not Path(result_file_path).exists():
            raise HTTPException(status_code=404, detail="Không tìm thấy file kết quả của tác vụ.")
        
        try:
            pt_data = torch.load(result_file_path, map_location="cpu")
            if "j3d_cam" in pt_data:
                pred_data = pt_data["j3d_cam"]
            elif "j3d_glob" in pt_data:
                pred_data = pt_data["j3d_glob"]
            elif "pred_j3d" in pt_data:
                pred_data = pt_data["pred_j3d"]
            elif "smpl_params_incam" in pt_data or "smpl_params_global" in pt_data:
                smpl_params = pt_data.get("smpl_params_incam") or pt_data.get("smpl_params_global")
                from hmr4d.model.gvhmr.utils.endecoder import EnDecoder
                endecoder = EnDecoder()
                with torch.no_grad():
                    body_pose = torch.as_tensor(smpl_params["body_pose"], dtype=torch.float32)
                    betas = torch.as_tensor(smpl_params["betas"], dtype=torch.float32)
                    global_orient = torch.as_tensor(smpl_params["global_orient"], dtype=torch.float32)
                    transl = torch.as_tensor(smpl_params["transl"], dtype=torch.float32)
                    if body_pose.dim() == 1 or body_pose.dim() == 2:
                        body_pose = body_pose.reshape(1, -1, 63)
                    if betas.dim() == 1 or betas.dim() == 2:
                        betas = betas.reshape(1, -1, 10)
                    if global_orient.dim() == 1 or global_orient.dim() == 2:
                        global_orient = global_orient.reshape(1, -1, 3)
                    if transl.dim() == 1 or transl.dim() == 2:
                        transl = transl.reshape(1, -1, 3)
                    pred_data = endecoder.fk_v2(body_pose, betas, global_orient, transl)[0]  # (F, 22, 3)
            else:
                raise HTTPException(status_code=400, detail="File kết quả không chứa dữ liệu j3d hoặc smpl_params hợp lệ.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lỗi khi đọc file kết quả job: {str(e)}")

    target_data = request.target_j3d

    # Nếu truyền target_file_path, tự động nạp ma trận GT từ file .npy hoặc .pt
    if request.target_file_path:
        gt_path = Path(request.target_file_path)
        if not gt_path.exists():
            raise HTTPException(status_code=404, detail=f"Không tìm thấy file GT: {request.target_file_path}")
        
        try:
            if gt_path.suffix == ".npy":
                loaded_npy = np.load(gt_path, allow_pickle=True)
                if isinstance(loaded_npy, np.ndarray):
                    if loaded_npy.dtype == object and loaded_npy.ndim == 0:
                        dict_data = loaded_npy.item()
                        target_data = dict_data.get("j3d") or dict_data.get("gt_j3d") or dict_data.get("joints3d") or dict_data.get("target_j3d") or dict_data.get("joints")
                    else:
                        target_data = loaded_npy
                elif isinstance(loaded_npy, dict):
                    target_data = loaded_npy.get("j3d") or loaded_npy.get("gt_j3d") or loaded_npy.get("joints3d") or loaded_npy.get("target_j3d")
            elif gt_path.suffix in [".pt", ".pth"]:
                loaded_pt = torch.load(gt_path, map_location="cpu")
                if isinstance(loaded_pt, dict):
                    target_data = loaded_pt.get("j3d") or loaded_pt.get("j3d_cam") or loaded_pt.get("j3d_glob") or loaded_pt.get("target_j3d") or loaded_pt.get("gt_j3d")
                else:
                    target_data = loaded_pt
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lỗi khi nạp file GT {request.target_file_path}: {str(e)}")

    if target_data is None:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp ma trận 'target_j3d' hoặc đường dẫn 'target_file_path' hợp lệ.")

    try:
        pred_tensor = torch.tensor(pred_data, dtype=torch.float32)
        target_tensor = torch.tensor(target_data, dtype=torch.float32)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể chuyển đổi dữ liệu đầu vào thành Tensor: {str(e)}")

    if pred_tensor.dim() == 2:
        pred_tensor = pred_tensor.unsqueeze(0)
    if target_tensor.dim() == 2:
        target_tensor = target_tensor.unsqueeze(0)

    if pred_tensor.dim() != 3 or target_tensor.dim() != 3:
        raise HTTPException(status_code=400, detail="Dữ liệu ma trận khớp 3D phải có dạng (F, J, 3) hoặc (J, 3).")

    if pred_tensor.shape[-1] != 3 or target_tensor.shape[-1] != 3:
        raise HTTPException(status_code=400, detail="Chiều cuối cùng của tọa độ khớp 3D phải bằng 3 (x, y, z).")

    # 1. Tự động cắt theo số lượng frame nhỏ hơn nếu F khác nhau
    if pred_tensor.shape[0] != target_tensor.shape[0]:
        min_f = min(pred_tensor.shape[0], target_tensor.shape[0])
        pred_tensor = pred_tensor[:min_f]
        target_tensor = target_tensor[:min_f]

    # 2. Tự động chuyển đổi hoặc căn chỉnh khớp 22 vs 24 joints
    j_pred = pred_tensor.shape[1]
    j_target = target_tensor.shape[1]

    if j_pred != j_target:
        if j_pred == 22 and j_target == 24:
            pred_tensor = convert_joints22_to_24(pred_tensor)
        elif j_pred == 24 and j_target == 22:
            target_tensor = convert_joints22_to_24(target_tensor)
        elif j_pred in (22, 24) and j_target == 17:
            # Map SMPL 22/24 joints to H36M 17 joints
            smpl_to_h36m17 = [0, 2, 5, 8, 1, 4, 7, 3, 12, 15, 15, 13, 16, 18, 14, 17, 19]
            pred_tensor = pred_tensor[:, smpl_to_h36m17, :]
        elif j_pred == 17 and j_target in (22, 24):
            smpl_to_h36m17 = [0, 2, 5, 8, 1, 4, 7, 3, 12, 15, 15, 13, 16, 18, 14, 17, 19]
            target_tensor = target_tensor[:, smpl_to_h36m17, :]
        else:
            min_j = min(j_pred, j_target)
            pred_tensor = pred_tensor[:, :min_j, :]
            target_tensor = target_tensor[:, :min_j, :]

    num_frames, num_joints, _ = pred_tensor.shape

    # 3. Tự động quy đổi đơn vị: nếu ma trận GT tính bằng mm (> 50.0) còn pred tính bằng m (<= 50.0) -> chuyển GT về m
    if target_tensor.abs().max() > 50.0 and pred_tensor.abs().max() <= 50.0:
        target_tensor = target_tensor / 1000.0
    elif pred_tensor.abs().max() > 50.0 and target_tensor.abs().max() <= 50.0:
        pred_tensor = pred_tensor / 1000.0

    # 4. Kiểm tra pelvis_idxs có hợp lệ với num_joints hay không
    pelvis_idxs = request.pelvis_idxs
    if max(pelvis_idxs) >= num_joints:
        pelvis_idxs = [0]

    dummy_verts_pred = torch.zeros((num_frames, 1, 3), dtype=pred_tensor.dtype, device=pred_tensor.device)
    dummy_verts_target = torch.zeros((num_frames, 1, 3), dtype=target_tensor.dtype, device=target_tensor.device)

    try:
        # Align by pelvis (Root Alignment)
        pred_aligned, target_aligned, _, _ = batch_align_by_pelvis(
            [pred_tensor, target_tensor, dummy_verts_pred, dummy_verts_target],
            pelvis_idxs=pelvis_idxs
        )

        scale_factor = 1000.0 if request.unit.lower() == "mm" else 1.0

        # MPJPE (Pelvis root aligned error)
        mpjpe_per_frame = compute_jpe(pred_aligned, target_aligned) * scale_factor

        # PA-MPJPE (Procrustes rigid alignment error)
        if num_frames in [2, 3]:
            S1 = pred_aligned.permute(0, 2, 1)
            S2 = target_aligned.permute(0, 2, 1)
            mu1 = S1.mean(dim=-1, keepdim=True)
            mu2 = S2.mean(dim=-1, keepdim=True)
            X1 = S1 - mu1
            X2 = S2 - mu2
            var1 = torch.sum(X1**2, dim=1).sum(dim=1)
            K = X1.bmm(X2.permute(0, 2, 1))
            U, s, V = torch.svd(K)
            Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0).repeat(U.shape[0], 1, 1)
            Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))
            R = V.bmm(Z.bmm(U.permute(0, 2, 1)))
            scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1
            t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))
            S1_hat = (scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t).permute(0, 2, 1)
        else:
            S1_hat = batch_compute_similarity_transform_torch(pred_aligned, target_aligned)

        pa_mpjpe_per_frame = compute_jpe(S1_hat, target_aligned) * scale_factor

        return {
            "status": "success",
            "unit": request.unit,
            "num_frames": int(num_frames),
            "num_joints": int(num_joints),
            "mpjpe": {
                "mean": float(np.mean(mpjpe_per_frame)),
                "min": float(np.min(mpjpe_per_frame)),
                "max": float(np.max(mpjpe_per_frame)),
                "std": float(np.std(mpjpe_per_frame)),
                "per_frame": [float(val) for val in mpjpe_per_frame]
            },
            "pa_mpjpe": {
                "mean": float(np.mean(pa_mpjpe_per_frame)),
                "min": float(np.min(pa_mpjpe_per_frame)),
                "max": float(np.max(pa_mpjpe_per_frame)),
                "std": float(np.std(pa_mpjpe_per_frame)),
                "per_frame": [float(val) for val in pa_mpjpe_per_frame]
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi khi tính toán chỉ số MPJPE/PA-MPJPE: {str(e)}")
