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
    "propeller_diameter_in": (1, 40),        # 2" micro-whoop props to 40" heavy-lift industrial props
    "motor_kv": (30, 3000),                  # 30KV large cinema/heavy-lift motors to 3000KV micro racing motors
    "battery_cells": (1, 16),                # 1S micro-whoop to 16S large industrial LiPo packs
    "wing_area_m2": (0.01, 20),              # small hand-launch fixed-wing to large fixed-wing UAS
    "lift_to_drag_ratio": (3, 60),           # draggy micro fixed-wing to high-performance sailplane-like L/D
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


# WHY ONE SHARED VALIDATOR COVERING ALL EIGHT FIELDS, INSTEAD OF ONE PER
# FIELD (the pattern used before this file was rewritten): pydantic v2
# lets a single @field_validator decorator list multiple field names --
# this does the exact same range check as before, just without repeating
# near-identical validator methods eight times across two classes.
_RANGE_CHECKED_FIELDS = list(SPEC_LIMITS.keys())


class DroneCreate(BaseModel):
    name: str
    model: Optional[str] = None
    mass_kg: Optional[float] = None
    frame_type: Optional[str] = None
    motor_count: Optional[int] = None
    max_thrust_n: Optional[float] = None
    battery_capacity_mah: Optional[float] = None
    max_speed_mps: Optional[float] = None

    # Added for real propeller/motor physics (see app/services/motor_performance.py):
    propeller_diameter_in: Optional[float] = None
    motor_kv: Optional[int] = None
    battery_cells: Optional[int] = None

    # Added for real fixed-wing cruise endurance physics (see
    # app/services/frame_comparison.py) -- only meaningful for
    # frame_type="fixed_wing"; ignored by rotorcraft calculations.
    wing_area_m2: Optional[float] = None
    lift_to_drag_ratio: Optional[float] = None

    @field_validator(*_RANGE_CHECKED_FIELDS)
    @classmethod
    def validate_spec_range(cls, v, info):
        return _validate_range(info.field_name, v)


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
    propeller_diameter_in: Optional[float] = None
    motor_kv: Optional[int] = None
    battery_cells: Optional[int] = None
    wing_area_m2: Optional[float] = None
    lift_to_drag_ratio: Optional[float] = None

    @field_validator(*_RANGE_CHECKED_FIELDS)
    @classmethod
    def validate_spec_range(cls, v, info):
        return _validate_range(info.field_name, v)


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
    propeller_diameter_in: Optional[float] = None
    motor_kv: Optional[int] = None
    battery_cells: Optional[int] = None
    wing_area_m2: Optional[float] = None
    lift_to_drag_ratio: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)
