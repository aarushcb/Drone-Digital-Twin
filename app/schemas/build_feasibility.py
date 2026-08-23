from pydantic import BaseModel, field_validator

from app.schemas.drone import SPEC_LIMITS, _validate_range

# Same range sanity-check used for a real Drone's stored specs (see
# app/schemas/drone.py's SPEC_LIMITS/_validate_range) -- reused as-is
# rather than duplicated, since a hypothetical build a student is still
# shopping for shouldn't be held to a stricter or looser standard than a
# real registered drone.
_FEASIBILITY_RANGE_FIELDS = [
    "mass_kg",
    "motor_count",
    "propeller_diameter_in",
    "motor_kv",
    "battery_cells",
    "battery_capacity_mah",
]


class BuildFeasibilityRequest(BaseModel):
    """
    A fully hypothetical spec -- no drone_id, nothing read from the
    database. Every field is required (unlike DroneCreate, where specs
    are optional and filled in later): the whole point of this endpoint
    is to answer "would this combination work", which needs every input
    up front.
    """

    frame_type: str
    mass_kg: float
    motor_count: int
    propeller_diameter_in: float
    motor_kv: float
    battery_cells: int
    battery_capacity_mah: float

    @field_validator(*_FEASIBILITY_RANGE_FIELDS)
    @classmethod
    def validate_spec_range(cls, v, info):
        return _validate_range(info.field_name, v)
