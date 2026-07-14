from sqlalchemy.orm import Session

from app.models.telemetry import Telemetry
from app.schemas.telemetry import TelemetryCreate


def create_telemetry(
    db: Session,
    telemetry: TelemetryCreate
):
    db_telemetry = Telemetry(
        drone_id=telemetry.drone_id,
        latitude=telemetry.latitude,
        longitude=telemetry.longitude,
        altitude=telemetry.altitude,
        battery=telemetry.battery,
        speed=telemetry.speed,
        temperature=telemetry.temperature
    )

    db.add(db_telemetry)
    db.commit()
    db.refresh(db_telemetry)

    return db_telemetry


def get_telemetry_by_drone(
    db: Session,
    drone_id: int
):
    return (
        db.query(Telemetry)
        .filter(Telemetry.drone_id == drone_id)
        .all()
    )