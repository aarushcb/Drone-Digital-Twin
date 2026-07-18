from pydantic import BaseModel
from typing import Optional, List


class PathPlanRequest(BaseModel):
    # Where the drone is starting from, in the same scene coordinate space
    # as obstacles/landing points. Defaults to the origin (0, 0) -- the
    # drone model's resting position in the 3D scene -- if not given.
    start_x: float = 0
    start_z: float = 0


class PathPoint(BaseModel):
    x: float
    z: float


class PathPlanResponse(BaseModel):
    path: List[PathPoint]
    distance_meters: float
    estimated_time_seconds: Optional[float] = None
