from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, NonNegativeInt

class JobCreateRequest(BaseModel):
    video_id: str
    static_cam: bool = True
    use_dpvo: bool = False

class CalculateMetricsRequest(BaseModel):
    pred_j3d: Optional[List[Any]] = None          # (F, J, 3) hoặc (J, 3)
    target_j3d: Optional[List[Any]] = None        # (F, J, 3) hoặc (J, 3)
    target_file_path: Optional[str] = None        # Đường dẫn file GT (.npy hoặc .pt) trên server
    job_id: Optional[str] = None                  # Tùy chọn: lấy pred_j3d từ job_id đã hoàn thành
    pelvis_idxs: List[NonNegativeInt] = Field(default_factory=lambda: [1, 2], min_length=1)
    unit: Literal["mm", "m"] = "mm"
