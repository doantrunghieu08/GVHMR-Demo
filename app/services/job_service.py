from app.database import SessionLocal
from app.models.job_model import Job

def get_job(job_id: str):
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
    with SessionLocal.begin() as db:
        db.add(Job(job_id=job_id, video_id=video_id))

def update_job(job_id: str, updates: dict):
    with SessionLocal.begin() as db:
        job = db.get(Job, job_id)
        if not job:
            return
        for key, value in updates.items():
            setattr(job, key, value)
