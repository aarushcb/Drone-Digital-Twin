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


# ---------- Minimum-jerk smoothed trajectory, built ON TOP OF plan_path_3d's
# own output (see app/services/path_planner.py's
# generate_minimum_jerk_trajectory) -- again a separate request/response
# model, not a change to PathPlanRequest3D/PathPlanResponse3D, so
# /plan-path-3d and its behavior are completely unaffected. ----------

class PathPlanSmoothRequest(BaseModel):
    start_x: float = 0
    start_y: float = 0
    start_z: float = 0

    # Overrides for the drone's own max_speed_mps (from its stored spec)
    # and a derived max acceleration (from motor thrust-to-weight -- see
    # routes.py) -- optional, only needed to try a different constraint
    # than the drone's own spec/defaults.
    max_speed_mps: Optional[float] = None
    max_acceleration_mps2: Optional[float] = None

    # How many trajectory samples per second of flight time to return --
    # an output-resolution knob only, doesn't affect the underlying
    # polynomial trajectory itself.
    sample_rate_hz: float = 10.0


class TrajectoryPoint(BaseModel):
    t: float
    x: float
    y: float
    z: float
    speed_mps: float
    acceleration_mps2: float


class PathPlanSmoothResponse(BaseModel):
    trajectory: List[TrajectoryPoint]
    keypoints: List[PathPoint3D]   # simplified waypoints the spline was actually fit through
    raw_waypoint_count: int        # how many points plan_path_3d's grid search returned
    keypoint_count: int            # how many survived safety-aware simplification
    distance_meters: float
    total_duration_seconds: float
    max_speed_mps_used: float
    max_acceleration_mps2_used: float
    max_realized_speed_mps: float
    max_realized_acceleration_mps2: float
