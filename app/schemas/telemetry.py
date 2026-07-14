from pydantic import BaseModel
from datetime import datetime


class TelemetryBase(BaseModel):
    drone_id: int
    latitude: float
    longitude: float
    altitude: float
    battery: float
    speed: float
    temperature: float


class TelemetryCreate(TelemetryBase):
    pass


class TelemetryResponse(TelemetryBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True