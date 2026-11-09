import threading
from datetime import datetime
from app.config import USE_SQL
from app.database import SessionLocal
from app.models.job_model import Job

_in_memory_jobs = {}
_jobs_lock = threading.Lock()

def get_job(job_id: str):
    if not USE_SQL or SessionLocal is None or Job is None:
        with _jobs_lock:
            job = _in_memory_jobs.get(job_id)
            return dict(job) if job else None

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if not job:
            return None
        return {
            "job_id": job.job_id,
            "video_id": job.video_id,
            "status": job.status,
            "progress": job.progress,
            "result": job.result,
            "created_at": job.created_at,
        }

def create_job_entry(job_id: str, video_id: str = None):
    if not USE_SQL or SessionLocal is None or Job is None:
        with _jobs_lock:
            _in_memory_jobs[job_id] = {
                "job_id": job_id,
                "video_id": video_id,
                "status": "PENDING",
                "progress": "Đang chờ xử lý",
                "result": None,
                "created_at": datetime.utcnow(),
            }
        return

    with SessionLocal.begin() as db:
        db.add(Job(job_id=job_id, video_id=video_id))

def update_job(job_id: str, updates: dict):
    if not USE_SQL or SessionLocal is None or Job is None:
        with _jobs_lock:
            if job_id in _in_memory_jobs:
                _in_memory_jobs[job_id].update(updates)
        return

    with SessionLocal.begin() as db:
        job = db.get(Job, job_id)
        if not job:
            return
        for key, value in updates.items():
            setattr(job, key, value)
