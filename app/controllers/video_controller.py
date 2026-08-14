import shutil
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
    file_path = UPLOAD_DIR / f"{video_id}_{video.filename}"

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
            
        return {
            "status": "success",
            "video_id": video_id,
            "filename": video.filename,
            "saved_path": str(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể lưu video: {str(e)}")

@router.get("/api/v1/download/{filename}", dependencies=[Depends(verify_token)])
def download_result(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy file kết quả.")
    return FileResponse(path=str(file_path), filename=filename, media_type="video/mp4")
