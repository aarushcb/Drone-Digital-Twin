"""
WHY THIS CHANGED FROM THE ORIGINAL:
1. The DB URL now comes from Settings (see app/config.py) instead of being
   hardcoded, so the same code runs locally (SQLite) and in production
   (Postgres) without edits.
2. `Base` (the SQLAlchemy declarative base all models inherit from) now lives
   HERE instead of inside models/drone.py. Previously telemetry.py imported
   Base from drone.py, which is fragile — if drone.py ever changed import
   order, telemetry.py could break. Having one obvious source of truth for
   Base is standard practice.
3. `connect_args={"check_same_thread": False}` is required specifically for
   SQLite (not Postgres) because FastAPI can use a request from a different
   thread than the one that opened the DB connection. This only applies when
   database_url starts with "sqlite" — Postgres doesn't need it.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import settings

connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(settings.database_url, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency: opens a DB session for a request, always closes it after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
