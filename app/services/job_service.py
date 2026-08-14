from app.database import SessionLocal
from app.models.job_model import Job

def get_job(job_id: str):
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if job:
            return {
                "job_id": job.job_id,
                "video_id": job.video_id,
                "status": job.status,
                "progress": job.progress,
                "result": job.result,
                "created_at": job.created_at
            }
        return None
    finally:
        db.close()

def create_job_entry(job_id: str, video_id: str = None):
    db = SessionLocal()
    try:
        new_job = Job(
            job_id=job_id,
            video_id=video_id,
            status="PENDING",
            progress="Đang chờ xử lý",
            result=None
        )
        db.add(new_job)
        db.commit()
    finally:
        db.close()

def update_job(job_id: str, updates: dict):
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if job:
            for key, value in updates.items():
                setattr(job, key, value)
            db.commit()
    finally:
        db.close()
