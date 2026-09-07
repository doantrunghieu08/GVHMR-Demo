from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.services.model_service import load_model_into_gpu
from app.controllers import health_controller, video_controller, job_controller, metrics_controller
from app.database import engine, Base

app = FastAPI(
    title="GVHMR API",
    servers=[
        {"url": "/", "description": "Current Host (Relative URL)"},
    ]
)

# Thêm CORS Middleware cho phép mọi domain truy cập
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

import threading

@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)
    # Tải model ở luồng ngầm để server khởi động ngay lập tức (<1s), không treo request
    threading.Thread(target=load_model_into_gpu, daemon=True).start()

app.include_router(health_controller.router)
app.include_router(video_controller.router)
app.include_router(job_controller.router)
app.include_router(metrics_controller.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
