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


# ---------- 3D grid-based path planning (see app/services/path_planner.py's
# plan_path_3d) -- separate request/response models, not a change to the
# ones above, so the existing 2D /plan-path endpoint and its Flutter
# consumer are completely unaffected. ----------

class PathPlanRequest3D(BaseModel):
    start_x: float = 0
    start_y: float = 0
    start_z: float = 0


class PathPoint3D(BaseModel):
    x: float
    y: float
    z: float


class PathPlanResponse3D(BaseModel):
    path: List[PathPoint3D]
    distance_meters: float
    estimated_time_seconds: Optional[float] = None
