from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.schemas.drone import DroneCreate, DroneResponse
from app.crud import drone as drone_crud
from app.crud import telemetry as telemetry_crud
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