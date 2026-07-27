from pydantic import BaseModel, ConfigDict, field_validator
from typing import Optional
from datetime import datetime

VALID_OBJECT_TYPES = {"obstacle", "landing", "path", "building", "tree"}


class SceneObjectCreate(BaseModel):
    object_type: str
    x: float
    y: float = 0
    z: float
    label: Optional[str] = None

    @field_validator("object_type")
    @classmethod
    def validate_object_type(cls, v: str) -> str:
        if v not in VALID_OBJECT_TYPES:
            raise ValueError(f"object_type must be one of {sorted(VALID_OBJECT_TYPES)}")
        return v


class SceneObjectOut(SceneObjectCreate):
    id: int
    drone_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
