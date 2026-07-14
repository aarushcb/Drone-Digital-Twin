from sqlalchemy.orm import Session
from app.models.drone import Drone
from app.schemas.drone import DroneCreate


def create_drone(db: Session, drone: DroneCreate) -> Drone:
    db_drone = Drone(
        name=drone.name,
        model=drone.model,
        status=drone.status,
    )
    db.add(db_drone)
    db.commit()
    db.refresh(db_drone)
    return db_drone


def get_drone(db: Session, drone_id: int) -> Drone | None:
    return db.query(Drone).filter(Drone.id == drone_id).first()


def get_drones(db: Session, skip: int = 0, limit: int = 100) -> list[Drone]:
    return db.query(Drone).offset(skip).limit(limit).all()