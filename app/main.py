from fastapi import FastAPI
from app.services.model_service import load_model_into_gpu
from app.controllers import health_controller, video_controller, job_controller, metrics_controller

app = FastAPI(title="GVHMR API")

from app.database import engine, Base

@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)
    load_model_into_gpu()

app.include_router(health_controller.router)
app.include_router(video_controller.router)
app.include_router(job_controller.router)
app.include_router(metrics_controller.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
