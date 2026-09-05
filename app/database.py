from app.config import DATABASE_URL

class Dummy:
    pass

engine = Dummy()
Base = Dummy
SessionLocal = Dummy

def get_db():
    yield None
