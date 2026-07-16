"""
WHY EVERY FUNCTION HERE NOW TAKES owner_id:
This is what actually enforces "you only see your own drones." It would be a
security bug to let someone fetch /drones/5 and get back a drone that
belongs to a different user just because they guessed the right ID. Filtering
by owner_id at the database query level (not just in the API layer) is the
safest place to put that check.
"""

from sqlalchemy.orm import Session

from app.models.drone import Drone
from app.schemas.drone import DroneCreate, DroneUpdate


def create_drone(db: Session, drone: DroneCreate, owner_id: int) -> Drone:
    db_drone = Drone(**drone.model_dump(), owner_id=owner_id)
    db.add(db_drone)
    db.commit()
    db.refresh(db_drone)
    return db_drone


def get_drone(db: Session, drone_id: int, owner_id: int) -> Drone | None:
    return (
        db.query(Drone)
        .filter(Drone.id == drone_id, Drone.owner_id == owner_id)
        .first()
    )


def get_drones(db: Session, owner_id: int, skip: int = 0, limit: int = 100) -> list[Drone]:
    return (
        db.query(Drone)
        .filter(Drone.owner_id == owner_id)
        .offset(skip)
        .limit(limit)
        .all()
    )


def update_drone(db: Session, db_drone: Drone, updates: DroneUpdate) -> Drone:
    for field, value in updates.model_dump(exclude_unset=True).items():
        setattr(db_drone, field, value)
    db.commit()
    db.refresh(db_drone)
    return db_drone


def delete_drone(db: Session, db_drone: Drone) -> None:
    db.delete(db_drone)
    db.commit()
