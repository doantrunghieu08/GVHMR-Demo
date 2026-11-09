from app.database import Base

if Base is not None:
    from sqlalchemy import Column, String, JSON, DateTime
    from datetime import datetime

    class Job(Base):
        __tablename__ = "jobs"
        __table_args__ = {'mysql_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_unicode_ci'}

        job_id = Column(String(50), primary_key=True, index=True)
        video_id = Column(String(50), index=True)
        status = Column(String(50), default="PENDING")
        progress = Column(String(255), default="Đang chờ xử lý")
        result = Column(JSON, nullable=True)
        created_at = Column(DateTime, default=datetime.utcnow)
else:
    Job = None
