from sqlalchemy.orm import Session
from typing import Optional

from app.models.scene_object import SceneObject
from app.schemas.scene_object import SceneObjectCreate


def create_scene_object(db: Session, obj: SceneObjectCreate, drone_id: int) -> SceneObject:
    db_obj = SceneObject(**obj.model_dump(), drone_id=drone_id)
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def get_scene_objects(db: Session, drone_id: int) -> list[SceneObject]:
    # Ordered by id (== creation order), which is also tap order -- this
    # is what lets the 3D scene draw a sensible line through "path" points
    # in the sequence they were actually placed.
    return (
        db.query(SceneObject)
        .filter(SceneObject.drone_id == drone_id)
        .order_by(SceneObject.id)
        .all()
    )


def get_scene_object(db: Session, object_id: int, drone_id: int) -> Optional[SceneObject]:
    return (
        db.query(SceneObject)
        .filter(SceneObject.id == object_id, SceneObject.drone_id == drone_id)
        .first()
    )


def delete_scene_object(db: Session, db_obj: SceneObject) -> None:
    db.delete(db_obj)
    db.commit()


def delete_all_scene_objects(db: Session, drone_id: int) -> None:
    db.query(SceneObject).filter(SceneObject.drone_id == drone_id).delete()
    db.commit()
