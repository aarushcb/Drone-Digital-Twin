from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime


class DroneCreate(BaseModel):
    name: str
    model: Optional[str] = None
    mass_kg: Optional[float] = None
    frame_type: Optional[str] = None
    motor_count: Optional[int] = None
    max_thrust_n: Optional[float] = None
    battery_capacity_mah: Optional[float] = None
    max_speed_mps: Optional[float] = None


class DroneUpdate(BaseModel):
    """All fields optional — lets you PATCH just the specs you want to change."""
    name: Optional[str] = None
    model: Optional[str] = None
    status: Optional[str] = None
    mass_kg: Optional[float] = None
    frame_type: Optional[str] = None
    motor_count: Optional[int] = None
    max_thrust_n: Optional[float] = None
    battery_capacity_mah: Optional[float] = None
    max_speed_mps: Optional[float] = None


class DroneOut(BaseModel):
    id: int
    name: str
    model: Optional[str] = None
    status: str
    registered_at: datetime
    mass_kg: Optional[float] = None
    frame_type: Optional[str] = None
    motor_count: Optional[int] = None
    max_thrust_n: Optional[float] = None
    battery_capacity_mah: Optional[float] = None
    max_speed_mps: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)
