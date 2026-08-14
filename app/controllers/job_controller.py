from uuid import uuid4
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException

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
    create_job_entry(job_id)

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
