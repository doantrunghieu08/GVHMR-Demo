import shutil
from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from app.security import verify_token
from app.config import UPLOAD_DIR, OUTPUT_DIR

router = APIRouter()

@router.post("/api/v1/video/upload", dependencies=[Depends(verify_token)])
async def upload_video(video: UploadFile = File(...)):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    video_id = str(uuid4())
    filename = Path(video.filename or "video.mp4").name
    file_path = UPLOAD_DIR / f"{video_id}_{filename}"

    print(f"\n[UPLOAD] Dang nhan file: {filename} (ID: {video_id})")
    try:
        with open(file_path, "wb") as buffer:
            size = 0
            while chunk := await video.read(1024 * 1024):
                buffer.write(chunk)
                size += len(chunk)
            
        print(f"[UPLOAD SUCCESS] Da luu file {filename} ({size / (1024*1024):.2f} MB)")
        return {
            "status": "success",
            "video_id": video_id,
            "filename": filename,
            "saved_path": str(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể lưu video: {str(e)}")

from pydantic import BaseModel, Field

class RegisterLocalVideoRequest(BaseModel):
    file_path: str = Field(..., example="input/demo.mp4", description="Đường dẫn file video trên server RunPod (VD: input/demo.mp4 hoặc /root/.../video.mp4)")

@router.post("/api/v1/video/register-local", dependencies=[Depends(verify_token)])
def register_local_video(request: RegisterLocalVideoRequest):
    """
    Sử dụng video có sẵn trên server RunPod để tạo video_id (Tránh lỗi giới hạn dung lượng upload qua Proxy Web)
    """
    src_path = Path(request.file_path)
    if not src_path.exists():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file tại đường dẫn: {request.file_path}")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    video_id = str(uuid4())
    filename = src_path.name
    dest_path = UPLOAD_DIR / f"{video_id}_{filename}"

    shutil.copy(src_path, dest_path)
    print(f"[REGISTER SUCCESS] Đã đăng ký file có sẵn: {filename} -> {video_id}")
    return {
        "status": "success",
        "video_id": video_id,
        "filename": filename,
        "saved_path": str(dest_path)
    }

@router.get("/api/v1/download/{filename}", dependencies=[Depends(verify_token)])
def download_result(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy file kết quả.")
    return FileResponse(path=str(file_path), filename=filename, media_type="video/mp4")
