"""
WHAT CHANGED FROM THE ORIGINAL:
- owner_id: ties every drone to the User who registered it (see user.py).
    Without this, there's no way to show "my drones" vs. everyone else's.
- Physical spec fields (mass_kg, frame_type, motor_count, max_thrust_n,
    battery_capacity_mah, max_speed_mps): your vision doc calls for
    "predictive analysis for multiple environment conditions" and
    "different models of drone." You can't predict how a drone behaves in
    wind, or how long its battery will last under load, without knowing
    its physical characteristics first. These fields are the foundation
    every future physics/ML calculation will read from. They're nullable
    for now so you can register a drone with just a name and fill in specs
    later — no need to have every number ready on day one.
- registered_at: simple audit/history field, useful later for "oldest
    drone," sorting, etc.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from app.database.database import Base


class Drone(Base):
    __tablename__ = "drones"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    name = Column(String, nullable=False)
    model = Column(String, nullable=True)
    status = Column(String, default="offline")  # offline, idle, armed, flying, error
    registered_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Physical specs — used later for physics-based prediction, not required at registration.
    mass_kg = Column(Float, nullable=True)
    frame_type = Column(String, nullable=True)          # e.g. "quadcopter", "hexacopter"
    motor_count = Column(Integer, nullable=True)
    max_thrust_n = Column(Float, nullable=True)          # newtons, total across all motors
    battery_capacity_mah = Column(Float, nullable=True)
    max_speed_mps = Column(Float, nullable=True)         # meters/second

    # Added for real propeller/motor physics (app/services/motor_performance.py) --
    # these three, together, let us compute actual required RPM, real power
    # draw, and estimated flight time using dimensional propeller thrust
    # theory calibrated against published UIUC-derived coefficients --
    # instead of only a relative percentage change like before.
    propeller_diameter_in = Column(Float, nullable=True)  # inches, standard industry unit
    motor_kv = Column(Integer, nullable=True)             # RPM per volt, standard motor rating
    battery_cells = Column(Integer, nullable=True)        # LiPo "S" rating, e.g. 4 = 4S = 14.8V nominal

    owner = relationship("User", back_populates="drones")
    telemetry = relationship("Telemetry", back_populates="drone", cascade="all, delete-orphan")
    scene_objects = relationship("SceneObject", back_populates="drone", cascade="all, delete-orphan")
