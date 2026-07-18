"""
WHY ONE MODEL FOR THREE DIFFERENT THINGS:
Obstacles, landing points, and path waypoints are all, structurally, the
same thing: a labeled point placed somewhere in the drone's 3D scene. The
only real difference is `object_type`, which the frontend uses to decide
how to render it (a red block, a green landing pad, a blue path marker)
and what it means for future path-planning logic (Phase 8). Splitting
these into three separate tables would just be repeating the same four
columns three times for no real benefit.
"""

from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from app.database.database import Base


class SceneObject(Base):
    __tablename__ = "scene_objects"

    id = Column(Integer, primary_key=True, index=True)
    drone_id = Column(Integer, ForeignKey("drones.id"), nullable=False, index=True)

    # "obstacle", "landing", or "path" -- validated in the Pydantic schema,
    # not the database, since that's easier to extend later (e.g. adding a
    # "geofence" type) without a migration.
    object_type = Column(String, nullable=False)

    # Position in the 3D scene's coordinate space (same space the drone
    # model itself is rendered in -- see assets/three_viewer/).
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False, default=0)
    z = Column(Float, nullable=False)

    label = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    drone = relationship("Drone", back_populates="scene_objects")
