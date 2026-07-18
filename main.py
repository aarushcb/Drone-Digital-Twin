"""
WHAT CHANGED FROM THE ORIGINAL:
- Now includes both the auth router (/auth/register, /auth/login) and the
  existing drone/telemetry router.
- Base.metadata.create_all(engine) runs on startup: this auto-creates any
  tables that don't exist yet from your models, so `python main.py` (or
  uvicorn) "just works" against a fresh SQLite file with zero manual setup.
  NOTE: this is fine for local development, but once you're on a shared
  production database, table changes should go through Alembic migrations
  instead (added in alembic/ — see the README for when to switch).
- CORS middleware added: without this, a browser-based client (or Flutter
  web build) would be blocked by the browser from calling this API at all,
  because it's a different origin. Mobile apps don't strictly need this, but
  it costs nothing to have and saves you a confusing debugging session later
  if you ever test from a browser.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.database import Base, engine
from app.models import user, drone, telemetry  # noqa: F401 — import so tables register
from app.api.routes import router as drone_router
from app.api.auth import router as auth_router
from app.api.ws import router as ws_router

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Drone Digital Twin API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your real app's origin before production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(drone_router)
app.include_router(ws_router)


@app.get("/")
def root():
    return {"message": "Drone Digital Twin Backend Running"}
