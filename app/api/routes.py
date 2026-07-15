from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.schemas.drone import DroneCreate, DroneResponse
from app.crud import drone as drone_crud
from app.crud import telemetry as telemetry_crud
from app.services.health_monitor import get_health_status
from app.services.alerts import generate_alerts
from app.services.drone_analytics import (
    calculate_risk_score,
    calculate_analytics
)
from app.schemas.telemetry import (
    TelemetryCreate,
    TelemetryResponse
)

router = APIRouter()


@router.post("/drone", response_model=DroneResponse)
def create_drone(drone: DroneCreate, db: Session = Depends(get_db)):
    return drone_crud.create_drone(db, drone)


@router.get("/drone/{id}", response_model=DroneResponse)
def read_drone(id: int, db: Session = Depends(get_db)):
    db_drone = drone_crud.get_drone(db, id)
    if db_drone is None:
        raise HTTPException(status_code=404, detail="Drone not found")
    return db_drone


@router.get("/drones", response_model=list[DroneResponse])
def read_drones(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return drone_crud.get_drones(db, skip=skip, limit=limit)


@router.post(
    "/telemetry",
    response_model=TelemetryResponse
)
def create_telemetry(
    telemetry: TelemetryCreate,
    db: Session = Depends(get_db)
):
    return telemetry_crud.create_telemetry(
        db,
        telemetry
    )


@router.get(
    "/telemetry/{drone_id}",
    response_model=list[TelemetryResponse]
)
def read_telemetry(
    drone_id: int,
    db: Session = Depends(get_db)
):
    return telemetry_crud.get_telemetry_by_drone(
        db,
        drone_id
    )
@router.get("/drone/{id}/status")
def get_drone_status(
    id: int,
    db: Session = Depends(get_db)
):
    telemetry_records = telemetry_crud.get_telemetry_by_drone(
        db,
        id
    )

    if not telemetry_records:
        raise HTTPException(
            status_code=404,
            detail="No telemetry found for drone"
        )

    latest = telemetry_records[-1]

    health = get_health_status(
        latest.battery,
        latest.temperature
    )

    alerts = generate_alerts(
        latest.battery,
        latest.temperature
    )

    risk_score = calculate_risk_score(
        latest.battery,
        latest.temperature,
        latest.speed
    )

    return {
        "drone_id": id,
        "health": health,
        "risk_score": risk_score,
        "alerts": alerts,
        "latest_telemetry": {
            "battery": latest.battery,
            "temperature": latest.temperature,
            "speed": latest.speed,
            "altitude": latest.altitude,
            "latitude": latest.latitude,
            "longitude": latest.longitude
        }
    }
@router.get("/drone/{id}/analytics")
def get_drone_analytics(
    id: int,
    db: Session = Depends(get_db)
):
    telemetry_records = (
        telemetry_crud.get_telemetry_by_drone(
            db,
            id
        )
    )

    if not telemetry_records:
        raise HTTPException(
            status_code=404,
            detail="No telemetry found"
        )

    return calculate_analytics(
        telemetry_records
    )