from app.config import DATABASE_URL, USE_SQL

engine = None
SessionLocal = None
Base = None

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import DeclarativeBase, sessionmaker

    connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    class Base(DeclarativeBase):
        pass
except ImportError:
    pass


def get_db():
    if USE_SQL and SessionLocal:
        with SessionLocal() as db:
            yield db
    else:
        yield None
