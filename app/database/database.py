from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.drone import Base

DATABASE_URL = "postgresql://aarushcb@localhost/drone_twin"

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()