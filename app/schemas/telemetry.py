from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime


class TelemetryCreate(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    battery: Optional[float] = None
    speed: Optional[float] = None
    temperature: Optional[float] = None
    roll: Optional[float] = None
    pitch: Optional[float] = None
    yaw: Optional[float] = None
    flight_state: Optional[str] = "idle"


class TelemetryOut(TelemetryCreate):
    id: int
    drone_id: int
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)
