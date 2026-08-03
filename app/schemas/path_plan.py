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


# ---------- Probabilistic / risk-aware 3D path planning (see
# app/services/path_planner.py's plan_path_3d_probabilistic) -- again a
# separate request/response model, not a change to PathPlanRequest3D/
# PathPlanResponse3D above, so /plan-path-3d and its behavior are
# completely unaffected. ----------

class PathPlanRequestProbabilistic(BaseModel):
    start_x: float = 0
    start_y: float = 0
    start_z: float = 0

    # How uncertain each obstacle's REPORTED position is assumed to be
    # (meters, standard deviation, isotropic Gaussian -- see
    # path_planner.py's module docstring). 0 makes this behave like the
    # hard-radius plan_path_3d.
    obstacle_position_std: float = 0.3

    # How much a unit of collision probability costs relative to a unit
    # of distance in the search -- higher values make the planner more
    # risk-averse (willing to detour further to cut risk).
    risk_weight: float = 50.0


class PathPlanResponseProbabilistic(BaseModel):
    path: List[PathPoint3D]
    collision_probability_per_point: List[float]
    max_collision_probability: float
    mean_collision_probability: float
    distance_meters: float
    estimated_time_seconds: Optional[float] = None
