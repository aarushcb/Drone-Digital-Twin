from pydantic import BaseModel, ConfigDict, field_validator
from typing import Optional
from datetime import datetime

# WHY THESE SPECIFIC RANGES:
# Chosen to comfortably cover everything from tiny micro/toy drones to
# large professional/industrial ones (agricultural drones, heavy-lift
# cinema rigs), while rejecting values that are physically absurd for
# ANY real drone (e.g. 1000kg, 200 m/s -- both of which were actually
# entered during testing, which is exactly what surfaced this gap).
# These aren't meant to be precise engineering limits, just a sanity
# backstop -- generous enough not to block real hobbyist/prosumer specs.
SPEC_LIMITS = {
    "mass_kg": (0.01, 50),                  # 10g micro drone to 50kg heavy-lift
    "max_speed_mps": (0, 100),               # up to 360 km/h -- covers record-holding racing drones
    "motor_count": (1, 16),                  # single-rotor to large heavy-lift multirotors
    "max_thrust_n": (0, 3000),               # generous upper bound for large industrial drones
    "battery_capacity_mah": (50, 100000),    # tiny micro drone to large ag-drone battery packs
}


def _validate_range(field_name: str, value):
    if value is None:
        return value
    low, high = SPEC_LIMITS[field_name]
    if not (low <= value <= high):
        raise ValueError(
            f"{field_name} must be between {low} and {high} (got {value}) -- "
            f"that's outside the range of any real drone."
        )
    return value


class DroneCreate(BaseModel):
    name: str
    model: Optional[str] = None
    mass_kg: Optional[float] = None
    frame_type: Optional[str] = None
    motor_count: Optional[int] = None
    max_thrust_n: Optional[float] = None
    battery_capacity_mah: Optional[float] = None
    max_speed_mps: Optional[float] = None

    @field_validator("mass_kg")
    @classmethod
    def validate_mass(cls, v):
        return _validate_range("mass_kg", v)

    @field_validator("max_speed_mps")
    @classmethod
    def validate_speed(cls, v):
        return _validate_range("max_speed_mps", v)

    @field_validator("motor_count")
    @classmethod
    def validate_motor_count(cls, v):
        return _validate_range("motor_count", v)

    @field_validator("max_thrust_n")
    @classmethod
    def validate_thrust(cls, v):
        return _validate_range("max_thrust_n", v)

    @field_validator("battery_capacity_mah")
    @classmethod
    def validate_battery(cls, v):
        return _validate_range("battery_capacity_mah", v)


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

    @field_validator("mass_kg")
    @classmethod
    def validate_mass(cls, v):
        return _validate_range("mass_kg", v)

    @field_validator("max_speed_mps")
    @classmethod
    def validate_speed(cls, v):
        return _validate_range("max_speed_mps", v)

    @field_validator("motor_count")
    @classmethod
    def validate_motor_count(cls, v):
        return _validate_range("motor_count", v)

    @field_validator("max_thrust_n")
    @classmethod
    def validate_thrust(cls, v):
        return _validate_range("max_thrust_n", v)

    @field_validator("battery_capacity_mah")
    @classmethod
    def validate_battery(cls, v):
        return _validate_range("battery_capacity_mah", v)


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
