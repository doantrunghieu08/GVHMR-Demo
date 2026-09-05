from uuid import uuid4
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
import zipfile
import os
from pathlib import Path

from app.security import verify_token
from app.models.schemas import JobCreateRequest
from app.services.job_service import create_job_entry, get_job
from app.services.video_service import process_video_task
from app.services.model_service import get_model
from app.config import UPLOAD_DIR

router = APIRouter()

@router.post("/api/v1/jobs", dependencies=[Depends(verify_token)])
async def create_job(
    request: JobCreateRequest,
    background_tasks: BackgroundTasks):
    
    """
    Tạo tác vụ xử lý video mới
    """
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model chưa được tải xong")
    
    matching_file = list(UPLOAD_DIR.glob(f"{request.video_id}_*"))

    if not matching_file:
        raise HTTPException(status_code=404, detail="Không tìm thấy video")
    video_path = matching_file[0]
    
    job_id = str(uuid4())
    create_job_entry(job_id, request.video_id)

    background_tasks.add_task(
        process_video_task,
        job_id,
        video_path,
        request.static_cam,
        request.use_dpvo
    )
    return {
        "status": "success",
        "message": "Đã bắt đầu tác vụ xử lý ngầm",
        "job_id": job_id
    }

@router.get("/api/v1/jobs/{job_id}", dependencies=[Depends(verify_token)])
def get_job_status(job_id: str):
    """
    Kiểm tra trạng thái API
    """
    job_info = get_job(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ (job_id) được yêu cầu.")

    return job_info

@router.get("/api/v1/jobs/{job_id}/download", dependencies=[Depends(verify_token)])
def download_job_results(job_id: str):
    """
    Nén (ZIP) tất cả file video và file tọa độ 3D (.pt) của tác vụ rồi tải về.
    Hỗ trợ cả job 1 người và 2 người.
    """
    from app.config import OUTPUT_DIR
    job_info = get_job(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ (job_id) được yêu cầu.")

    if job_info.get("status") != "COMPLETED":
        raise HTTPException(status_code=400, detail="Tác vụ chưa hoàn thành.")

    result = job_info.get("result")
    if not result:
        raise HTTPException(status_code=500, detail="Không tìm thấy kết quả của tác vụ.")

    # Hỗ trợ cả format mới (danh sách) lẫn format cũ (1 file)
    video_urls = result.get("output_video_urls") or []
    if not video_urls and result.get("output_video_url"):
        video_urls = [result["output_video_url"]]

    result_file_paths_raw = result.get("result_file_paths") or []
    if not result_file_paths_raw and result.get("result_file_path"):
        result_file_paths_raw = [result["result_file_path"]]

    if not video_urls:
        raise HTTPException(status_code=500, detail="Thiếu thông tin đường dẫn file kết quả.")

    # Thu thập tất cả các file cần đóng gói
    files_to_zip = []  # list of (path, arcname)

    for url in video_urls:
        filename = url.split("/")[-1]
        path = OUTPUT_DIR / filename
        if path.exists():
            files_to_zip.append((path, filename))

    for pt_path_str in result_file_paths_raw:
        pt_path = Path(pt_path_str)
        if pt_path.exists():
            files_to_zip.append((pt_path, pt_path.name))

    if not files_to_zip:
        raise HTTPException(status_code=404, detail="File kết quả không tồn tại trên hệ thống.")

    zip_filename = f"{job_id}_results.zip"
    zip_path = OUTPUT_DIR / zip_filename

    # Tạo file zip nếu chưa có
    if not zip_path.exists():
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path, arcname in files_to_zip:
                zipf.write(file_path, arcname=arcname)

    return FileResponse(path=str(zip_path), filename=zip_filename, media_type="application/zip")

