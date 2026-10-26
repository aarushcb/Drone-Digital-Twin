from sqlalchemy.orm import Session

from app.models.user import User
from app.schemas.user import UserCreate
from app.core.security import hash_password


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, user: UserCreate) -> User:
    db_user = User(email=user.email, hashed_password=hash_password(user.password))
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def delete_account(db: Session, user: User) -> dict:
    """
    Deletes a user and everything they own, in one transaction, and returns
    how many rows of each kind went. Every table is deleted explicitly rather
    than left to ON DELETE CASCADE: SQLite doesn't enforce foreign keys by
    default, and the counts are what the app shows the user as confirmation.

    Raw flight logs are never stored: the server parses each upload from a
    temporary file and deletes it once analysed (services/health/store.py).
    The only log files that can exist at this moment belong to uploads still
    queued or being analysed in this process; their jobs find the flight
    gone and delete the file. Those are counted as log_files_pending.
    """
    from app.models.drone import Drone
    from app.models.health import ConfigEpoch, FeatureBaseline, Flight, FlightFeature, Pack
    from app.models.scene_object import SceneObject
    from app.models.telemetry import Telemetry
    from app.services.health import store

    drone_ids = [d for (d,) in db.query(Drone.id).filter(Drone.owner_id == user.id)]
    flight_ids = [f for (f,) in db.query(Flight.id).filter(Flight.drone_id.in_(drone_ids))] if drone_ids else []
    counts = {"log_files_pending": sum(1 for f in flight_ids if store.is_active(f))}

    def wipe(name, query):
        counts[name] = query.delete(synchronize_session=False)

    if flight_ids:
        wipe("flight_features", db.query(FlightFeature).filter(FlightFeature.flight_id.in_(flight_ids)))
    else:
        counts["flight_features"] = 0
    if drone_ids:
        wipe("feature_baselines", db.query(FeatureBaseline).filter(FeatureBaseline.drone_id.in_(drone_ids)))
        wipe("flights", db.query(Flight).filter(Flight.drone_id.in_(drone_ids)))
        wipe("config_epochs", db.query(ConfigEpoch).filter(ConfigEpoch.drone_id.in_(drone_ids)))
        wipe("packs", db.query(Pack).filter(Pack.drone_id.in_(drone_ids)))
        wipe("telemetry", db.query(Telemetry).filter(Telemetry.drone_id.in_(drone_ids)))
        wipe("scene_objects", db.query(SceneObject).filter(SceneObject.drone_id.in_(drone_ids)))
        wipe("drones", db.query(Drone).filter(Drone.id.in_(drone_ids)))
    else:
        counts.update(feature_baselines=0, flights=0, config_epochs=0, packs=0, telemetry=0, scene_objects=0, drones=0)
    wipe("user", db.query(User).filter(User.id == user.id))
    db.commit()
    return counts
