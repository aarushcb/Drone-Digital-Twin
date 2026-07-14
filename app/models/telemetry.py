from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.models.drone import Base


class Telemetry(Base):
    __tablename__ = "telemetry"

    id = Column(Integer, primary_key=True, index=True)

    drone_id = Column(Integer, ForeignKey("drones.id"))

    latitude = Column(Float)
    longitude = Column(Float)

    altitude = Column(Float)
    battery = Column(Float)

    speed = Column(Float)
    temperature = Column(Float)

    timestamp = Column(DateTime(timezone=True),
                       server_default=func.now())