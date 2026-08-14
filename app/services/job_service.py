jobs_db = {}

def get_job(job_id: str):
    return jobs_db.get(job_id)

def create_job_entry(job_id: str):
    jobs_db[job_id] = {
        "status": "PENDING",
        "progress": "Dang cho xu ly",
        "result": None
    }

def update_job(job_id: str, updates: dict):
    if job_id in jobs_db:
        jobs_db[job_id].update(updates)

def get_all_jobs_db():
    return jobs_db
