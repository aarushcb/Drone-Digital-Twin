"""
WHAT CHANGED FROM THE ORIGINAL:
- Every endpoint now depends on get_current_user (from deps.py). FastAPI
  runs that first; if there's no valid token, the request never reaches your
  endpoint code.
- Every drone lookup is scoped with owner_id=current_user.id, so one user
  can never see or modify another user's drone, even by guessing an ID.
- URLs changed from singular /drone/{id} to plural /drones/{id} to match
  standard REST convention (a detail I'm fixing now while we're touching
  this file anyway, since it's much harder to change later once a mobile
  app is calling it in production).
- Telemetry listing now supports pagination (limit/offset) and optional
  start/end date filtering — see crud/telemetry.py for why.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime

from app.database.database import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.drone import DroneCreate, DroneOut, DroneUpdate
from app.schemas.telemetry import TelemetryCreate, TelemetryOut
from app.crud import drone as drone_crud
from app.crud import telemetry as telemetry_crud
from app.services.health_monitor import get_health_status
from app.services.alerts import generate_alerts
from app.services.drone_analytics import calculate_risk_score, calculate_analytics

router = APIRouter()


def _get_owned_drone_or_404(db: Session, drone_id: int, current_user: User):
    db_drone = drone_crud.get_drone(db, drone_id, owner_id=current_user.id)
    if db_drone is None:
        raise HTTPException(status_code=404, detail="Drone not found")
    return db_drone


# ---------- Drones ----------

@router.post("/drones", response_model=DroneOut, status_code=201)
def create_drone(
    drone: DroneCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return drone_crud.create_drone(db, drone, owner_id=current_user.id)


@router.get("/drones", response_model=List[DroneOut])
def list_drones(
    skip: int = 0,
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return drone_crud.get_drones(db, owner_id=current_user.id, skip=skip, limit=limit)


@router.get("/drones/{drone_id}", response_model=DroneOut)
def read_drone(
    drone_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return _get_owned_drone_or_404(db, drone_id, current_user)


@router.patch("/drones/{drone_id}", response_model=DroneOut)
def update_drone(
    drone_id: int,
    updates: DroneUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db_drone = _get_owned_drone_or_404(db, drone_id, current_user)
    return drone_crud.update_drone(db, db_drone, updates)


@router.delete("/drones/{drone_id}", status_code=204)
def delete_drone(
    drone_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db_drone = _get_owned_drone_or_404(db, drone_id, current_user)
    drone_crud.delete_drone(db, db_drone)


# ---------- Telemetry ----------

@router.post("/drones/{drone_id}/telemetry", response_model=TelemetryOut, status_code=201)
def create_telemetry(
    drone_id: int,
    telemetry: TelemetryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_drone_or_404(db, drone_id, current_user)
    return telemetry_crud.create_telemetry(db, telemetry, drone_id=drone_id)


@router.get("/drones/{drone_id}/telemetry", response_model=List[TelemetryOut])
def read_telemetry(
    drone_id: int,
    limit: int = Query(100, le=1000),
    offset: int = 0,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_drone_or_404(db, drone_id, current_user)
    return telemetry_crud.get_telemetry(
        db, drone_id, limit=limit, offset=offset, start=start, end=end
    )


# ---------- Status / analytics (existing rule-based logic, now owner-scoped) ----------

@router.get("/drones/{drone_id}/status")
def get_drone_status(
    drone_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_drone_or_404(db, drone_id, current_user)

    latest_records = telemetry_crud.get_telemetry(db, drone_id, limit=1)
    if not latest_records:
        raise HTTPException(status_code=404, detail="No telemetry found for drone")

    latest = latest_records[0]

    health = get_health_status(latest.battery, latest.temperature)
    alerts = generate_alerts(latest.battery, latest.temperature)
    risk_score = calculate_risk_score(latest.battery, latest.temperature, latest.speed)

    return {
        "drone_id": drone_id,
        "health": health,
        "risk_score": risk_score,
        "alerts": alerts,
        "latest_telemetry": {
            "battery": latest.battery,
            "temperature": latest.temperature,
            "speed": latest.speed,
            "altitude": latest.altitude,
            "latitude": latest.latitude,
            "longitude": latest.longitude,
            "roll": latest.roll,
            "pitch": latest.pitch,
            "yaw": latest.yaw,
            "flight_state": latest.flight_state,
        },
    }


@router.get("/drones/{drone_id}/analytics")
def get_drone_analytics(
    drone_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_drone_or_404(db, drone_id, current_user)

    # limit=1000 here just bounds this to a sane amount for the current
    # threshold-based analytics; once this becomes a real predictive model
    # (later step) it'll pull from the full history in batches instead.
    records = telemetry_crud.get_telemetry(db, drone_id, limit=1000)
    if not records:
        raise HTTPException(status_code=404, detail="No telemetry found")

    return calculate_analytics(records)
