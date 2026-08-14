from pydantic import BaseModel
from typing import List, Optional, Any

class JobCreateRequest(BaseModel):
    video_id: str
    static_cam: bool = True
    use_dpvo: bool = True

class CalculateMetricsRequest(BaseModel):
    pred_j3d: Optional[List[Any]] = None          # (F, J, 3) hoặc (J, 3)
    target_j3d: Optional[List[Any]] = None        # (F, J, 3) hoặc (J, 3)
    target_file_path: Optional[str] = None        # Đường dẫn file GT (.npy hoặc .pt) trên server
    job_id: Optional[str] = None                  # Tùy chọn: lấy pred_j3d từ job_id đã hoàn thành
    pelvis_idxs: List[int] = [1, 2]
    unit: str = "mm"
