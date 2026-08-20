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
    desired_roll: Optional[float] = None
    desired_pitch: Optional[float] = None
    desired_yaw: Optional[float] = None
    motor_pwm_1: Optional[float] = None
    motor_pwm_2: Optional[float] = None
    motor_pwm_3: Optional[float] = None
    motor_pwm_4: Optional[float] = None
    accel_x: Optional[float] = None
    accel_y: Optional[float] = None
    accel_z: Optional[float] = None
    gyro_x: Optional[float] = None
    gyro_y: Optional[float] = None
    gyro_z: Optional[float] = None


class TelemetryOut(TelemetryCreate):
    id: int
    drone_id: int
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)
