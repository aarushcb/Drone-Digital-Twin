"""
WHAT CHANGED FROM THE ORIGINAL:
- roll, pitch, yaw: your original telemetry only had position (lat/long/
    altitude) and speed. But a 3D digital twin needs to know which way the
    drone is FACING to render it correctly in the scene — position alone
    only tells you where a dot is, not how the model should be rotated.
    These three angles (in degrees) are the standard way to describe a
    drone's orientation in 3D space.
- flight_state: a simple text field (idle/armed/flying/landing/error) so
    the dashboard and 3D view can show a clear status, and so alerting logic
    can react differently depending on whether the drone is actually flying.
- battery field kept as "battery" (not renamed to battery_percent) so the
    existing health_monitor/alerts/drone_analytics service functions you
    already wrote keep working without changes.
- timestamp now uses a plain Python default instead of the Postgres-only
    server_default=func.now(), because that syntax doesn't work the same way
    across SQLite (local dev) and Postgres (production) — this version works
    identically on both.
"""

from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from app.database.database import Base


class Telemetry(Base):
    __tablename__ = "telemetry"

    id = Column(Integer, primary_key=True, index=True)
    drone_id = Column(Integer, ForeignKey("drones.id"), nullable=False, index=True)

    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    altitude = Column(Float, nullable=True)
    battery = Column(Float, nullable=True)     # percentage, 0-100
    speed = Column(Float, nullable=True)       # meters/second
    temperature = Column(Float, nullable=True)  # celsius

    # Orientation, in degrees — needed to render the drone correctly in the 3D twin.
    roll = Column(Float, nullable=True)
    pitch = Column(Float, nullable=True)
    yaw = Column(Float, nullable=True)

    flight_state = Column(String, default="idle")  # idle, armed, flying, landing, error

    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    drone = relationship("Drone", back_populates="telemetry")
