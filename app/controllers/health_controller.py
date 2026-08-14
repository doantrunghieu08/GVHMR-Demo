import torch
from fastapi import APIRouter

router = APIRouter()

@router.get("/api/v1/health")
def health_check():
    """
    Kiểm tra trạng thái server
    """
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }
