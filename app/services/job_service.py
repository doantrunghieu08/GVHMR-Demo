from datetime import datetime

_jobs_db = {}

def get_job(job_id: str):
    job = _jobs_db.get(job_id)
    if job:
        return {
            "job_id": job.get("job_id"),
            "video_id": job.get("video_id"),
            "status": job.get("status"),
            "progress": job.get("progress"),
            "result": job.get("result"),
            "created_at": job.get("created_at")
        }
    return None

def create_job_entry(job_id: str, video_id: str = None):
    _jobs_db[job_id] = {
        "job_id": job_id,
        "video_id": video_id,
        "status": "PENDING",
        "progress": "Đang chờ xử lý",
        "result": None,
        "created_at": datetime.utcnow()
    }

def update_job(job_id: str, updates: dict):
    if job_id in _jobs_db:
        for key, value in updates.items():
            _jobs_db[job_id][key] = value
