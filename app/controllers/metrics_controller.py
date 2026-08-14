from fastapi import APIRouter, Depends
from app.security import verify_token
from app.models.schemas import CalculateMetricsRequest
from app.services.metrics_service import evaluate_metrics_logic

router = APIRouter()

@router.post("/api/v1/metrics/evaluate", dependencies=[Depends(verify_token)])
async def evaluate_metrics(request: CalculateMetricsRequest):
    """
    Tính toán sai số MPJPE (Mean Per Joint Position Error) và PA-MPJPE (Procrustes-Aligned MPJPE)
    """
    return evaluate_metrics_logic(request)
