"""
WHAT CHANGED FROM THE ORIGINAL:
- Every endpoint now depends on get_current_user (from deps.py). FastAPI
  runs that first; if there's no valid token, the request never reaches your
  endpoint code.
- Every drone lookup is scoped with owner_id=current_user.id, so one user
  can never see or modify another user's drone, even by guessing an ID.
- URLs changed from singular /drone/{id} to plural /drones/{id} to match
  standard REST convention (a detail I'm fixing now while we're touching
  this file anyway, since it's much harder to change later once a mobile
  app is calling it in production).
- Telemetry listing now supports pagination (limit/offset) and optional
  start/end date filtering — see crud/telemetry.py for why.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime
from dataclasses import asdict
import tempfile
import os
import math

from app.database.database import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.drone import DroneCreate, DroneOut, DroneUpdate
from app.schemas.telemetry import TelemetryCreate, TelemetryOut
from app.schemas.scene_object import SceneObjectCreate, SceneObjectOut
from app.schemas.path_plan import (
    PathPlanRequest, PathPlanResponse, PathPoint,
    PathPlanRequest3D, PathPlanResponse3D, PathPoint3D,
    PathPlanRequestProbabilistic, PathPlanResponseProbabilistic,
    PathPlanSmoothRequest, PathPlanSmoothResponse, TrajectoryPoint,
)
from app.schemas.environment import EnvironmentSimulationRequest
from app.schemas.parameter_sweep import ParameterSweepRequest
from app.schemas.sensor_calibration import CalibrationCheckRequest
from app.crud import drone as drone_crud
from app.crud import telemetry as telemetry_crud
from app.crud import scene_object as scene_object_crud
from app.services.alerts import generate_alerts
from app.services.drone_analytics import calculate_analytics
from app.services.telemetry_broadcaster import manager
from app.services.path_planner import (
    plan_path, path_distance_meters, plan_path_3d, path_distance_meters_3d,
    plan_path_3d_probabilistic, generate_minimum_jerk_trajectory,
)
from app.services.predictive_analytics import (
    estimate_battery_remaining,
    detect_anomalies,
    calculate_predictive_risk_score,
)
from app.services.digital_twin import compute_digital_twin_stats
from app.services.environment_simulator import simulate_conditions, air_density
from app.services.motor_performance import analyze_motor_performance, has_complete_motor_specs, CT_STATIC
from app.services.bemt import analyze_bemt_hover, analyze_bemt_hover_in_wind
from app.services.monte_carlo_uq import monte_carlo_hover_uncertainty, monte_carlo_wind_endurance_uncertainty
from app.services.kalman_filter import smooth_altitude_series, detect_sensor_faults, fuse_full_state
from app.services.mavlink_import import parse_mavlink_log, MavlinkImportError
from app.services.parameter_sweep import analyze_parameter_sweep
from app.services.efficiency_landscape import compute_efficiency_landscape
from app.services.sensor_calibration import analyze_calibration
from app.services.control_loop import simulate_control_loop, build_real_control_loop_response
from app.services.frame_comparison import (
    compare_frames, VALID_FRAME_TYPES, max_available_thrust_n, GRAVITY_MPS2, DEFAULT_CRUISE_VELOCITY_MPS,
)

router = APIRouter()


def _get_owned_drone_or_404(db: Session, drone_id: int, current_user: User):
    db_drone = drone_crud.get_drone(db, drone_id, owner_id=current_user.id)
    if db_drone is None:
        raise HTTPException(status_code=404, detail="Drone not found")
    return db_drone


# ---------- Drones ----------

@router.post("/drones", response_model=DroneOut, status_code=201)
def create_drone(
    drone: DroneCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return drone_crud.create_drone(db, drone, owner_id=current_user.id)


@router.get("/drones", response_model=List[DroneOut])
def list_drones(
    skip: int = 0,
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return drone_crud.get_drones(db, owner_id=current_user.id, skip=skip, limit=limit)


@router.get("/drones/{drone_id}", response_model=DroneOut)
def read_drone(
    drone_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return _get_owned_drone_or_404(db, drone_id, current_user)


@router.patch("/drones/{drone_id}", response_model=DroneOut)
def update_drone(
    drone_id: int,
    updates: DroneUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db_drone = _get_owned_drone_or_404(db, drone_id, current_user)
    return drone_crud.update_drone(db, db_drone, updates)


@router.delete("/drones/{drone_id}", status_code=204)
def delete_drone(
    drone_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db_drone = _get_owned_drone_or_404(db, drone_id, current_user)
    drone_crud.delete_drone(db, db_drone)


# ---------- Telemetry ----------

@router.post("/drones/{drone_id}/telemetry", response_model=TelemetryOut, status_code=201)
async def create_telemetry(
    drone_id: int,
    telemetry: TelemetryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_drone_or_404(db, drone_id, current_user)
    db_telemetry = telemetry_crud.create_telemetry(db, telemetry, drone_id=drone_id)

    # Push this new reading to anyone currently watching this drone's
    # live view (see app/api/ws.py). If nobody's connected right now,
    # broadcast() just does nothing -- this never blocks or fails the
    # actual save, which already succeeded above.
    #
    # NOTE: this route is `async def` (unlike the other endpoints) purely
    # so we can `await manager.broadcast(...)`. The database calls inside
    # it are still the same synchronous SQLAlchemy calls as everywhere
    # else -- fine at this app's current scale, but worth knowing: heavy
    # concurrent traffic would eventually want a fully async DB setup.
    await manager.broadcast(drone_id, {
        "latitude": db_telemetry.latitude,
        "longitude": db_telemetry.longitude,
        "altitude": db_telemetry.altitude,
        "battery": db_telemetry.battery,
        "speed": db_telemetry.speed,
        "temperature": db_telemetry.temperature,
        "roll": db_telemetry.roll,
        "pitch": db_telemetry.pitch,
        "yaw": db_telemetry.yaw,
        "flight_state": db_telemetry.flight_state,
        "timestamp": db_telemetry.timestamp.isoformat(),
        "drone_id": drone_id,
        "id": db_telemetry.id,
    })

    return db_telemetry


@router.get("/drones/{drone_id}/telemetry", response_model=List[TelemetryOut])
def read_telemetry(
    drone_id: int,
    limit: int = Query(100, le=1000),
    offset: int = 0,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_drone_or_404(db, drone_id, current_user)
    return telemetry_crud.get_telemetry(
        db, drone_id, limit=limit, offset=offset, start=start, end=end
    )


@router.get("/drones/{drone_id}/telemetry/smoothed")
def read_smoothed_telemetry(
    drone_id: int,
    limit: int = Query(200, le=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Runs a Kalman filter (see app/services/kalman_filter.py) over this
    drone's stored altitude readings to produce a smoothed altitude and
    an estimated vertical velocity (climb rate) at each point -- a purely
    additive read-side endpoint. Does not touch the raw telemetry table,
    the WebSocket ingestion path, or the existing GET .../telemetry
    endpoint; rerunning it is always safe.
    """
    _get_owned_drone_or_404(db, drone_id, current_user)
    readings = telemetry_crud.get_telemetry(db, drone_id, limit=limit)
    readings_sorted = sorted(readings, key=lambda t: t.timestamp)

    altitude_points = [(t.timestamp, t.altitude) for t in readings_sorted if t.altitude is not None]
    smoothed = smooth_altitude_series(altitude_points)

    return {
        "drone_id": drone_id,
        "num_points": len(smoothed),
        "points": smoothed,
    }


@router.get("/drones/{drone_id}/telemetry/fused")
def read_fused_state(
    drone_id: int,
    limit: int = Query(200, le=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Runs the 9-state Extended Kalman Filter (see
    app/services/kalman_filter.py's fuse_full_state) over this drone's
    stored telemetry to produce one consistent fused estimate of position
    (local east/north/altitude), velocity, and attitude at each point --
    instead of displaying position, speed, and attitude as independent
    unfused raw readings. Purely additive read-side endpoint, same as
    /telemetry/smoothed above: doesn't touch the raw telemetry table, the
    WebSocket ingestion path, or any existing telemetry endpoint.
    """
    _get_owned_drone_or_404(db, drone_id, current_user)
    readings = telemetry_crud.get_telemetry(db, drone_id, limit=limit)
    readings_sorted = sorted(readings, key=lambda t: t.timestamp)

    reading_dicts = [
        {
            "timestamp": t.timestamp,
            "latitude": t.latitude,
            "longitude": t.longitude,
            "altitude": t.altitude,
            "speed": t.speed,
            "roll": t.roll,
            "pitch": t.pitch,
            "yaw": t.yaw,
        }
        for t in readings_sorted
    ]
    fused = fuse_full_state(reading_dicts)

    return {
        "drone_id": drone_id,
        "num_points": len(fused),
        "points": fused,
    }


@router.delete("/drones/{drone_id}/telemetry", status_code=200)
def clear_telemetry(
    drone_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Deletes ALL telemetry history for this drone -- lets you start fresh
    for testing (e.g. before recording a clean flight for Flight
    Verification) instead of accumulating scattered readings across every
    past session forever. Does not delete the drone itself, its specs, or
    any placed scene objects (obstacles/landing/path) -- only telemetry.
    """
    _get_owned_drone_or_404(db, drone_id, current_user)
    deleted_count = telemetry_crud.delete_all_telemetry(db, drone_id)
    return {"deleted_count": deleted_count}


# ---------- MAVLink flight log import ----------

@router.post("/drones/{drone_id}/import-mavlink")
async def import_mavlink_log(
    drone_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Imports a real .tlog or .bin MAVLink flight log (from a PX4 or
    ArduPilot-based drone) and inserts it as telemetry history for this
    drone -- so a REAL flight can be replayed, verified, and analyzed
    through the existing 3D replay, flight verification, and analytics
    features, not just simulated data.
    """
    _get_owned_drone_or_404(db, drone_id, current_user)

    # pymavlink's log reader needs a real file path, not an in-memory
    # stream, so the upload is written to a temp file first and cleaned
    # up afterward regardless of success or failure.
    suffix = os.path.splitext(file.filename or "")[1] or ".tlog"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        contents = await file.read()
        tmp.write(contents)

    try:
        points = parse_mavlink_log(tmp_path)
    except MavlinkImportError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)

    point_dicts = [asdict(p) for p in points]
    for pd in point_dicts:
        pd.pop("drone_id", None)  # bulk_create_telemetry sets this itself

    inserted = telemetry_crud.bulk_create_telemetry(db, point_dicts, drone_id=drone_id)

    return {
        "imported_points": inserted,
        "start_time": points[0].timestamp.isoformat(),
        "end_time": points[-1].timestamp.isoformat(),
        "filename": file.filename,
    }


# ---------- Status / analytics (existing rule-based logic, now owner-scoped) ----------

@router.get("/drones/{drone_id}/status")
def get_drone_status(
    drone_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_drone_or_404(db, drone_id, current_user)

    # Fetch the latest reading PLUS recent history -- the old version only
    # ever looked at a single snapshot in time, which is exactly why it
    # could only do flat threshold checks. Predictions and anomaly
    # detection need to see a window of recent behavior, not just "now."
    records = telemetry_crud.get_telemetry(db, drone_id, limit=51)
    if not records:
        raise HTTPException(status_code=404, detail="No telemetry found for drone")

    latest = records[0]
    history = records[1:]  # everything except the current reading itself

    battery_estimate = estimate_battery_remaining(history + [latest])
    anomalies = detect_anomalies(latest, history)

    # WHY THIS IS ADDITIVE: reuses the Kalman filter already built for
    # altitude smoothing (app/services/kalman_filter.py) to also run
    # innovation-based sensor fault detection -- a different, complementary
    # signal to the z-score anomalies above (see that module's FAULT
    # DETECTION docstring section). Feeds into risk_score as an extra,
    # capped factor; every drone with insufficient altitude history simply
    # gets an empty fault list and a 0-point contribution, same as before.
    altitude_series = [
        (t.timestamp, t.altitude) for t in sorted(history + [latest], key=lambda t: t.timestamp)
        if t.altitude is not None
    ]
    smoothed = smooth_altitude_series(altitude_series)
    sensor_faults = detect_sensor_faults(smoothed)
    sensor_fault_penalty = sum(20 if f["severity"] == "CRITICAL" else 10 for f in sensor_faults)

    risk_score = calculate_predictive_risk_score(
        latest, history, battery_estimate, sensor_fault_penalty=sensor_fault_penalty
    )

    # health label kept simple and directly tied to the same risk_score,
    # so the two never contradict each other the way two separately
    # computed values could.
    if risk_score >= 70:
        health = "CRITICAL"
    elif risk_score >= 40:
        health = "WARNING"
    else:
        health = "GOOD"

    # Legacy threshold-based alerts kept alongside the new anomaly list --
    # "battery below 20%" is still worth saying plainly even though it's
    # not novel, and the statistical anomalies add genuinely new
    # information on top rather than replacing something that still works.
    alerts = generate_alerts(latest.battery, latest.temperature)

    return {
        "drone_id": drone_id,
        "health": health,
        "risk_score": risk_score,
        "alerts": alerts,
        "anomalies": anomalies,
        "sensor_faults": sensor_faults,
        "battery_estimate": battery_estimate,
        "latest_telemetry": {
            "battery": latest.battery,
            "temperature": latest.temperature,
            "speed": latest.speed,
            "altitude": latest.altitude,
            "latitude": latest.latitude,
            "longitude": latest.longitude,
            "roll": latest.roll,
            "pitch": latest.pitch,
            "yaw": latest.yaw,
            "flight_state": latest.flight_state,
        },
    }


@router.get("/drones/{drone_id}/analytics")
def get_drone_analytics(
    drone_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_drone_or_404(db, drone_id, current_user)

    # limit=1000 here just bounds this to a sane amount for the current
    # threshold-based analytics; once this becomes a real predictive model
    # (later step) it'll pull from the full history in batches instead.
    records = telemetry_crud.get_telemetry(db, drone_id, limit=1000)
    if not records:
        raise HTTPException(status_code=404, detail="No telemetry found")

    return calculate_analytics(records)


# ---------- Scene objects (obstacles, landing points, path waypoints) ----------
# WHY THESE ARE SEPARATE FROM TELEMETRY:
# Telemetry is data ABOUT the drone (where it is, how it's doing).
# Scene objects are data about the ENVIRONMENT the drone operates in --
# set up once by the user, not streamed continuously. Different lifecycle,
# different endpoints.

@router.post("/drones/{drone_id}/objects", response_model=SceneObjectOut, status_code=201)
def create_scene_object(
    drone_id: int,
    obj: SceneObjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_drone_or_404(db, drone_id, current_user)
    return scene_object_crud.create_scene_object(db, obj, drone_id=drone_id)


@router.get("/drones/{drone_id}/objects", response_model=List[SceneObjectOut])
def list_scene_objects(
    drone_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_drone_or_404(db, drone_id, current_user)
    return scene_object_crud.get_scene_objects(db, drone_id)


@router.delete("/drones/{drone_id}/objects/{object_id}", status_code=204)
def delete_scene_object(
    drone_id: int,
    object_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_drone_or_404(db, drone_id, current_user)
    db_obj = scene_object_crud.get_scene_object(db, object_id, drone_id)
    if db_obj is None:
        raise HTTPException(status_code=404, detail="Scene object not found")
    scene_object_crud.delete_scene_object(db, db_obj)


@router.delete("/drones/{drone_id}/objects", status_code=204)
def clear_scene_objects(
    drone_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_drone_or_404(db, drone_id, current_user)
    scene_object_crud.delete_all_scene_objects(db, drone_id)


# ---------- Path planning ----------
# WHY THIS USES THE DRONE'S PLACED "landing" OBJECT AS THE GOAL:
# Rather than requiring a separate goal input, this reuses whatever
# landing point the user already placed in the 3D view (Phase 7) -- one
# consistent source of truth for "where is this drone supposed to end up."
# If more than one landing point exists, the first one placed is used.
#
# COORDINATE SYSTEM CAVEAT: this operates in the same simplified "scene
# units" as obstacles/landing points, not real-world GPS meters. Distance
# and time estimates below are only as meaningful as that scene's scale is
# consistent with the drone's real specs -- treated as an approximation
# for now, not survey-grade navigation output.

@router.post("/drones/{drone_id}/plan-path", response_model=PathPlanResponse)
def plan_drone_path(
    drone_id: int,
    request: PathPlanRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db_drone = _get_owned_drone_or_404(db, drone_id, current_user)

    scene_objects = scene_object_crud.get_scene_objects(db, drone_id)
    obstacles = [(o.x, o.z) for o in scene_objects if o.object_type == "obstacle"]
    landing_points = [o for o in scene_objects if o.object_type == "landing"]

    if not landing_points:
        raise HTTPException(
            status_code=404,
            detail="No landing point set for this drone. Place one in the 3D view first.",
        )

    goal = (landing_points[0].x, landing_points[0].z)
    start = (request.start_x, request.start_z)

    try:
        path = plan_path(start=start, goal=goal, obstacles=obstacles)
    except ValueError as e:
        # Start or goal is literally inside an obstacle's safety radius --
        # a specific, actionable error rather than a generic 500.
        raise HTTPException(status_code=422, detail=str(e))

    if path is None:
        raise HTTPException(
            status_code=422,
            detail="No valid path found — the landing point may be fully enclosed by obstacles.",
        )

    distance = path_distance_meters(path)

    estimated_time = None
    if db_drone.max_speed_mps and db_drone.max_speed_mps > 0:
        estimated_time = distance / db_drone.max_speed_mps

    return PathPlanResponse(
        path=[PathPoint(x=p[0], z=p[1]) for p in path],
        distance_meters=round(distance, 2),
        estimated_time_seconds=round(estimated_time, 1) if estimated_time else None,
    )


@router.post("/drones/{drone_id}/plan-path-3d", response_model=PathPlanResponse3D)
def plan_drone_path_3d(
    drone_id: int,
    request: PathPlanRequest3D,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Altitude-aware counterpart to /plan-path above (see
    app/services/path_planner.py's plan_path_3d) -- a separate endpoint,
    not a change to /plan-path, so the existing 2D planner and its
    Flutter consumer are unaffected. Obstacles/landing points already
    store a real y (height) coordinate (see app/models/scene_object.py);
    this is the first feature to actually use it for planning.
    """
    db_drone = _get_owned_drone_or_404(db, drone_id, current_user)

    scene_objects = scene_object_crud.get_scene_objects(db, drone_id)
    obstacles = [(o.x, o.y, o.z) for o in scene_objects if o.object_type == "obstacle"]
    landing_points = [o for o in scene_objects if o.object_type == "landing"]

    if not landing_points:
        raise HTTPException(
            status_code=404,
            detail="No landing point set for this drone. Place one in the 3D view first.",
        )

    goal = (landing_points[0].x, landing_points[0].y, landing_points[0].z)
    start = (request.start_x, request.start_y, request.start_z)

    try:
        path = plan_path_3d(start=start, goal=goal, obstacles=obstacles)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if path is None:
        raise HTTPException(
            status_code=422,
            detail="No valid path found — the landing point may be fully enclosed by obstacles.",
        )

    distance = path_distance_meters_3d(path)

    estimated_time = None
    if db_drone.max_speed_mps and db_drone.max_speed_mps > 0:
        estimated_time = distance / db_drone.max_speed_mps

    return PathPlanResponse3D(
        path=[PathPoint3D(x=p[0], y=p[1], z=p[2]) for p in path],
        distance_meters=round(distance, 2),
        estimated_time_seconds=round(estimated_time, 1) if estimated_time else None,
    )


@router.post("/drones/{drone_id}/plan-path-probabilistic", response_model=PathPlanResponseProbabilistic)
def plan_drone_path_probabilistic(
    drone_id: int,
    request: PathPlanRequestProbabilistic,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Risk-aware counterpart to /plan-path-3d above (see
    app/services/path_planner.py's plan_path_3d_probabilistic) -- a
    separate endpoint, not a change to /plan-path-3d, so that endpoint
    and its behavior are completely unaffected. Treats each obstacle's
    placed position as the MEAN of an uncertain (Gaussian) true position
    rather than an exact point, and minimizes cumulative collision
    probability (Monte Carlo-estimated, reusing the same sampling pattern
    as app/services/monte_carlo_uq.py) instead of just avoiding a fixed
    radius.
    """
    db_drone = _get_owned_drone_or_404(db, drone_id, current_user)

    scene_objects = scene_object_crud.get_scene_objects(db, drone_id)
    obstacles = [(o.x, o.y, o.z) for o in scene_objects if o.object_type == "obstacle"]
    landing_points = [o for o in scene_objects if o.object_type == "landing"]

    if not landing_points:
        raise HTTPException(
            status_code=404,
            detail="No landing point set for this drone. Place one in the 3D view first.",
        )

    goal = (landing_points[0].x, landing_points[0].y, landing_points[0].z)
    start = (request.start_x, request.start_y, request.start_z)

    try:
        result = plan_path_3d_probabilistic(
            start=start,
            goal=goal,
            obstacle_means=obstacles,
            obstacle_position_std=request.obstacle_position_std,
            risk_weight=request.risk_weight,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if result is None:
        raise HTTPException(
            status_code=422,
            detail="No valid path found — every route may pass through near-certain collision risk.",
        )

    path, risks = result
    distance = path_distance_meters_3d(path)

    estimated_time = None
    if db_drone.max_speed_mps and db_drone.max_speed_mps > 0:
        estimated_time = distance / db_drone.max_speed_mps

    return PathPlanResponseProbabilistic(
        path=[PathPoint3D(x=p[0], y=p[1], z=p[2]) for p in path],
        collision_probability_per_point=[round(r, 4) for r in risks],
        max_collision_probability=round(max(risks), 4),
        mean_collision_probability=round(sum(risks) / len(risks), 4),
        distance_meters=round(distance, 2),
        estimated_time_seconds=round(estimated_time, 1) if estimated_time else None,
    )


# ---------- Minimum-jerk smoothed trajectory (built ON TOP OF plan_path_3d's
# waypoints, not a change to /plan-path-3d) ----------

# Used only as a fallback -- see _derive_max_acceleration_mps2 below --
# when a drone doesn't have complete motor/prop/battery specs on file, so
# there's no real physics to derive an acceleration bound from. A
# moderate, gentle-camera-drone-representative figure (racing drones can
# exceed 10 m/s^2; this deliberately errs conservative since it's a
# fallback, not a measured value).
DEFAULT_TRAJECTORY_MAX_ACCELERATION_MPS2 = 2.5


def _derive_max_acceleration_mps2(db_drone) -> float:
    """
    A REAL max-acceleration bound derived from the drone's own spec, not
    an arbitrary constant, when motor/prop/battery specs are on file:
    max_available_thrust_n (frame_comparison.py -- total thrust at full
    throttle) minus the thrust already spent just hovering (=weight)
    leaves excess thrust available to accelerate. Decomposing the thrust
    vector into a component that cancels gravity and a component that
    provides horizontal acceleration (bounded by the same total thrust
    magnitude, so this is a Pythagorean relationship) gives max
    horizontal acceleration = g*sqrt(TWR^2 - 1), TWR = max_thrust/weight
    -- 0 at TWR=1 (all available thrust needed just to hover, nothing
    left over) and approaching g*TWR for large TWR, both physically
    sensible limits. Falls back to DEFAULT_TRAJECTORY_MAX_ACCELERATION_MPS2
    if motor specs are incomplete or the drone can't even hover (TWR<=1).
    """
    if not has_complete_motor_specs(db_drone):
        return DEFAULT_TRAJECTORY_MAX_ACCELERATION_MPS2

    max_thrust_n = max_available_thrust_n(
        motor_count=db_drone.motor_count,
        propeller_diameter_in=db_drone.propeller_diameter_in,
        motor_kv=db_drone.motor_kv,
        battery_cells=db_drone.battery_cells,
        air_density=1.225,  # sea-level ISA reference -- this is a planning-time capability bound, not a live-conditions estimate
    )
    weight_n = db_drone.mass_kg * GRAVITY_MPS2
    if max_thrust_n <= weight_n:
        return DEFAULT_TRAJECTORY_MAX_ACCELERATION_MPS2
    return math.sqrt(max_thrust_n ** 2 - weight_n ** 2) / db_drone.mass_kg


@router.post("/drones/{drone_id}/plan-path-smooth", response_model=PathPlanSmoothResponse)
def plan_drone_path_smooth(
    drone_id: int,
    request: PathPlanSmoothRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Smooths /plan-path-3d's blocky, one-point-per-grid-cell A* waypoint
    list into a continuous, dynamically-feasible trajectory (see
    app/services/path_planner.py's generate_minimum_jerk_trajectory) --
    runs plan_path_3d exactly like /plan-path-3d does, then feeds its
    output waypoints into the smoother as a SEPARATE step. This is a new,
    separate endpoint, not a change to /plan-path-3d -- that endpoint and
    its existing behavior/tests are completely unaffected.

    max_speed_mps/max_acceleration_mps2 default to this drone's own spec
    (max_speed_mps as already stored; max_acceleration_mps2 derived from
    real motor thrust-to-weight when the drone has complete motor specs,
    see _derive_max_acceleration_mps2) but can be overridden per-request.
    """
    db_drone = _get_owned_drone_or_404(db, drone_id, current_user)

    scene_objects = scene_object_crud.get_scene_objects(db, drone_id)
    obstacles = [(o.x, o.y, o.z) for o in scene_objects if o.object_type == "obstacle"]
    landing_points = [o for o in scene_objects if o.object_type == "landing"]

    if not landing_points:
        raise HTTPException(
            status_code=404,
            detail="No landing point set for this drone. Place one in the 3D view first.",
        )

    goal = (landing_points[0].x, landing_points[0].y, landing_points[0].z)
    start = (request.start_x, request.start_y, request.start_z)

    try:
        raw_path = plan_path_3d(start=start, goal=goal, obstacles=obstacles)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if raw_path is None:
        raise HTTPException(
            status_code=422,
            detail="No valid path found — the landing point may be fully enclosed by obstacles.",
        )

    max_speed_mps = (
        request.max_speed_mps
        or db_drone.max_speed_mps
        or DEFAULT_CRUISE_VELOCITY_MPS["quad"]
    )
    max_acceleration_mps2 = request.max_acceleration_mps2 or _derive_max_acceleration_mps2(db_drone)

    try:
        result = generate_minimum_jerk_trajectory(
            waypoints=raw_path,
            max_speed_mps=max_speed_mps,
            max_acceleration_mps2=max_acceleration_mps2,
            obstacles=obstacles,
            sample_rate_hz=request.sample_rate_hz,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    distance = path_distance_meters_3d(raw_path)

    return PathPlanSmoothResponse(
        trajectory=[
            TrajectoryPoint(
                t=round(s["t"], 4),
                x=round(s["position"][0], 4),
                y=round(s["position"][1], 4),
                z=round(s["position"][2], 4),
                speed_mps=round(s["speed_mps"], 4),
                acceleration_mps2=round(s["acceleration_mps2"], 4),
            )
            for s in result["samples"]
        ],
        keypoints=[PathPoint3D(x=p[0], y=p[1], z=p[2]) for p in result["keypoints"]],
        raw_waypoint_count=result["raw_waypoint_count"],
        keypoint_count=result["keypoint_count"],
        distance_meters=round(distance, 2),
        total_duration_seconds=result["total_duration_seconds"],
        max_speed_mps_used=max_speed_mps,
        max_acceleration_mps2_used=round(max_acceleration_mps2, 3),
        max_realized_speed_mps=result["max_realized_speed_mps"],
        max_realized_acceleration_mps2=result["max_realized_acceleration_mps2"],
    )


# ---------- Interactive parameter sweep (educational "what-if" tool) ----------

@router.post("/drones/{drone_id}/parameter-sweep")
def parameter_sweep(
    drone_id: int,
    request: ParameterSweepRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Lets a student adjust motor KV, propeller diameter, battery cell
    count, and total mass (as deltas from this drone's stored spec) and
    see the real physics consequence (see app/services/parameter_sweep.py)
    -- reuses motor_performance.py's exact dimensional thrust/power
    formulas for both the current and swept spec, so the comparison is
    apples-to-apples using one already-verified physics model.
    """
    db_drone = _get_owned_drone_or_404(db, drone_id, current_user)

    if not has_complete_motor_specs(db_drone):
        raise HTTPException(
            status_code=422,
            detail="This drone is missing motor/propeller/battery specs needed for a parameter sweep. Fill them in first.",
        )

    return analyze_parameter_sweep(
        mass_kg=db_drone.mass_kg,
        motor_count=db_drone.motor_count,
        propeller_diameter_in=db_drone.propeller_diameter_in,
        motor_kv=db_drone.motor_kv,
        battery_cells=db_drone.battery_cells,
        battery_capacity_mah=db_drone.battery_capacity_mah,
        mass_delta_kg=request.mass_delta_kg,
        propeller_diameter_delta_in=request.propeller_diameter_delta_in,
        motor_kv_delta=request.motor_kv_delta,
        battery_cells_delta=request.battery_cells_delta,
    )


@router.get("/drones/{drone_id}/efficiency-landscape")
def efficiency_landscape(
    drone_id: int,
    grid_size: int = Query(12, ge=4, le=25),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    2D (propeller diameter x motor KV) grid of hover efficiency (W/kg),
    centered on this drone's stored spec -- see
    app/services/efficiency_landscape.py. Same underlying physics as
    /parameter-sweep, evaluated over a grid instead of one point, for a
    heatmap visualization of the design space instead of a single
    before/after comparison.
    """
    db_drone = _get_owned_drone_or_404(db, drone_id, current_user)

    if not has_complete_motor_specs(db_drone):
        raise HTTPException(
            status_code=422,
            detail="This drone is missing motor/propeller/battery specs needed for the efficiency landscape. Fill them in first.",
        )

    return compute_efficiency_landscape(
        mass_kg=db_drone.mass_kg,
        motor_count=db_drone.motor_count,
        propeller_diameter_in=db_drone.propeller_diameter_in,
        motor_kv=db_drone.motor_kv,
        battery_cells=db_drone.battery_cells,
        battery_capacity_mah=db_drone.battery_capacity_mah,
        grid_size=grid_size,
    )


# ---------- Sensor calibration guide (IMU/compass/ESC) ----------

@router.post("/drones/{drone_id}/calibration-check")
def calibration_check(
    drone_id: int,
    request: CalibrationCheckRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Analyzes logged sensor readings from an IMU/compass/ESC calibration
    step (see app/services/sensor_calibration.py) -- a stateless analysis
    endpoint scoped to this drone for ownership/auth consistency with the
    rest of the API, but it doesn't read or write any stored telemetry;
    the readings to analyze come directly in the request body (captured
    from whatever calibration routine the student just ran), matching the
    step-by-step nature of the Flutter calibration guide (accel -> gyro ->
    compass -> ESC), where each step is checked independently as it's
    completed.
    """
    _get_owned_drone_or_404(db, drone_id, current_user)

    return analyze_calibration(
        accel_x=request.accel_x, accel_y=request.accel_y, accel_z=request.accel_z,
        gyro_x=request.gyro_x, gyro_y=request.gyro_y, gyro_z=request.gyro_z,
        mag_x=request.mag_x, mag_y=request.mag_y, mag_z=request.mag_z,
        esc_readings=request.esc_readings,
    )


# ---------- Attitude control loop visualization (educational simulation) ----------

@router.get("/drones/{drone_id}/telemetry/control-loop")
def control_loop(
    drone_id: int,
    scenario: str = Query("stable", pattern="^(stable|oscillating)$"),
    duration_s: float = Query(4.0, ge=1.0, le=10.0),
    sample_rate_hz: float = Query(50.0, ge=10.0, le=200.0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Shows the drone's attitude control loop (roll/pitch/yaw desired vs.
    actual). If this drone's telemetry actually has real setpoint data
    (desired_roll/pitch/yaw, populated by a MAVLink log import -- see
    app/services/mavlink_import.py's ATTITUDE_TARGET parsing), that real
    data is returned, clearly labeled "data_source": "real". Otherwise --
    the common case, since ordinary telemetry has never recorded a
    setpoint -- this falls back to an honestly-labeled SIMULATION using a
    standard 2nd-order PD-controlled system model (see
    app/services/control_loop.py), "data_source": "simulated".
    `scenario`/`duration_s`/`sample_rate_hz` only affect the simulated
    fallback. `scenario` lets a student directly compare a well-tuned
    ("stable") vs. poorly-tuned ("oscillating") controller as one real
    parameter (damping ratio) change, not two unrelated datasets.
    """
    _get_owned_drone_or_404(db, drone_id, current_user)

    real_samples = telemetry_crud.get_control_loop_samples(db, drone_id)
    sample_dicts = [
        {
            "timestamp": s.timestamp,
            "roll": s.roll,
            "pitch": s.pitch,
            "yaw": s.yaw,
            "desired_roll": s.desired_roll,
            "desired_pitch": s.desired_pitch,
            "desired_yaw": s.desired_yaw,
            "motor_pwm_1": s.motor_pwm_1,
            "motor_pwm_2": s.motor_pwm_2,
            "motor_pwm_3": s.motor_pwm_3,
            "motor_pwm_4": s.motor_pwm_4,
        }
        for s in real_samples
    ]
    real_response = build_real_control_loop_response(sample_dicts)
    if real_response is not None:
        return real_response

    simulation = simulate_control_loop(
        scenario=scenario,
        duration_s=duration_s,
        sample_rate_hz=sample_rate_hz,
    )
    simulation["data_source"] = "simulated"
    return simulation


# ---------- Frame comparison tool (educational "same motors, different airframe" tool) ----------
# WHY THIS IS NOT SCOPED UNDER /drones/{id}: unlike every other endpoint
# in this file, this one isn't about a specific stored drone -- it
# compares HYPOTHETICAL airframes for a shared motor/prop/battery choice
# a student is still deciding on, so the spec comes directly in the
# request instead of being read from a saved Drone row. Still requires
# login, for the same auth consistency as the rest of this API.

@router.get("/frames/compare")
def frames_compare(
    frame_types: str = Query(..., description="Comma-separated: quad,hex,octo,fixed_wing"),
    mass_kg: float = Query(..., gt=0),
    propeller_diameter_in: float = Query(..., gt=0),
    motor_kv: float = Query(..., gt=0),
    battery_cells: int = Query(..., gt=0),
    battery_capacity_mah: float = Query(..., gt=0),
    velocity_mps: Optional[float] = Query(None, gt=0),
    wing_area_m2: Optional[float] = Query(None, gt=0, description="Only used for fixed_wing; ignored by rotorcraft"),
    lift_to_drag_ratio: Optional[float] = Query(None, gt=0, description="Only used for fixed_wing; ignored by rotorcraft"),
    current_user: User = Depends(get_current_user),
):
    """
    Compares the requested frame types (see app/services/frame_comparison.py)
    for the SAME motor/propeller/battery/mass choice -- thrust-to-weight
    ratio, max tilt/bank angle, turn rate, hover efficiency, flight time,
    and max altitude, side by side. wing_area_m2/lift_to_drag_ratio are
    optional and only affect the fixed_wing entry's real cruise-endurance
    physics (falls back to documented representative defaults if omitted).
    """
    requested = [f.strip() for f in frame_types.split(",") if f.strip()]
    if not requested:
        raise HTTPException(status_code=422, detail="frame_types must include at least one frame type.")
    invalid = [f for f in requested if f not in VALID_FRAME_TYPES]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid frame type(s): {invalid}. Valid options: {sorted(VALID_FRAME_TYPES)}",
        )

    return compare_frames(
        frame_types=requested,
        mass_kg=mass_kg,
        propeller_diameter_in=propeller_diameter_in,
        motor_kv=motor_kv,
        battery_cells=battery_cells,
        battery_capacity_mah=battery_capacity_mah,
        velocity_mps=velocity_mps,
        wing_area_m2=wing_area_m2,
        lift_to_drag_ratio=lift_to_drag_ratio,
    )


# ---------- Digital Twin (lifetime aggregation, wear & maintenance) ----------
# WHY THIS IS SEPARATE FROM /status AND /analytics:
# Those look at a single point in time or a single recent session. This
# looks at the drone's ENTIRE recorded history to answer a different
# question: how much has this specific physical drone actually been used
# over its whole life, and is it due for maintenance -- the actual "digital
# twin" concept (a persistent virtual counterpart tracking real-world wear),
# not just a live dashboard.

@router.get("/drones/{drone_id}/digital-twin")
def get_digital_twin(
    drone_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db_drone = _get_owned_drone_or_404(db, drone_id, current_user)

    # Capped at 2000 readings -- a documented approximation for
    # extremely long-lived drones with more history than that, same
    # honest-limitation pattern used elsewhere (e.g. Flight Verification's
    # 500-reading cap for finding the "first-ever" GPS position).
    all_readings = telemetry_crud.get_telemetry(db, drone_id, limit=2000)
    all_readings.sort(key=lambda t: t.timestamp)  # oldest-first, required by compute_digital_twin_stats

    readings_as_dicts = [
        {
            "timestamp": t.timestamp,
            "battery": t.battery,
            "latitude": t.latitude,
            "longitude": t.longitude,
        }
        for t in all_readings
    ]

    stats = compute_digital_twin_stats(readings_as_dicts)

    return {
        "drone_id": drone_id,
        "specs": {
            "frame_type": db_drone.frame_type,
            "mass_kg": db_drone.mass_kg,
            "motor_count": db_drone.motor_count,
            "max_thrust_n": db_drone.max_thrust_n,
            # battery_cells (LiPo "S" rating / nominal voltage) and
            # battery_capacity_mah (capacity) are two DIFFERENT, both
            # independently optional spec fields -- e.g. "4S 3500mAh" is
            # a normal, coherent battery description, not two ways of
            # saying the same thing. Both are included here (cells was
            # previously missing from this response entirely) so a drone
            # with only one of the two set doesn't display an unrelated
            # value with no way to tell which spec it actually is.
            "battery_cells": db_drone.battery_cells,
            "battery_capacity_mah": db_drone.battery_capacity_mah,
            "max_speed_mps": db_drone.max_speed_mps,
        },
        **stats,
    }


# ---------- Environment condition simulation ----------
# WHY THIS REUSES THE DRONE'S OWN PHASE 9 BATTERY ESTIMATE AS A BASELINE:
# Rather than inventing a flight-time number from scratch (which we
# genuinely can't do precisely without real power-draw specs we don't
# collect), this takes the drone's own EMPIRICALLY OBSERVED drain rate
# and projects how established physics says it would change under
# different conditions -- grounded in real data about this specific
# drone, not a generic simulation.

@router.post("/drones/{drone_id}/simulate-environment")
def simulate_environment(
    drone_id: int,
    request: EnvironmentSimulationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db_drone = _get_owned_drone_or_404(db, drone_id, current_user)

    recent_readings = telemetry_crud.get_telemetry(db, drone_id, limit=51)
    baseline_estimate = None
    if recent_readings:
        recent_readings_sorted = sorted(recent_readings, key=lambda t: t.timestamp)
        baseline_estimate = estimate_battery_remaining(recent_readings_sorted)

    baseline_drain_rate = None
    if baseline_estimate and baseline_estimate.get("drain_rate_percent_per_min"):
        # Only meaningful if the battery is actually draining (negative
        # slope) -- a flat/charging baseline has nothing sensible to project.
        rate = baseline_estimate["drain_rate_percent_per_min"]
        if rate < 0:
            baseline_drain_rate = -rate  # store as a positive "percent per minute drained"

    result = simulate_conditions(
        altitude_m=request.altitude_m,
        temperature_c=request.temperature_c,
        baseline_drain_rate_percent_per_min=baseline_drain_rate,
    )
    result["drone_id"] = drone_id
    result["altitude_m"] = request.altitude_m
    result["temperature_c"] = request.temperature_c

    # WHY THIS IS ADDITIVE, NOT A REPLACEMENT: the relative percentage-
    # change calculation above still works for every drone, specs or not.
    # This ABSOLUTE analysis (real RPM, watts, amps, flight time) only
    # runs when the drone has the full motor/propeller/battery spec set
    # on file -- gracefully omitted (not a fake number) otherwise.
    if has_complete_motor_specs(db_drone):
        motor_analysis = analyze_motor_performance(
            mass_kg=db_drone.mass_kg,
            motor_count=db_drone.motor_count,
            propeller_diameter_in=db_drone.propeller_diameter_in,
            motor_kv=db_drone.motor_kv,
            battery_cells=db_drone.battery_cells,
            battery_capacity_mah=db_drone.battery_capacity_mah,
            air_density=air_density(request.altitude_m, request.temperature_c),
        )
        motor_analysis["static_thrust_coefficient_used"] = CT_STATIC
        result["motor_performance"] = motor_analysis

        # WHY THIS IS ADDITIVE TOO: a second, independently-derived thrust
        # model (blade-element/momentum theory, see app/services/bemt.py)
        # cross-checking the static-Ct model above -- purely a new field,
        # doesn't change or replace motor_performance.
        result["bemt_analysis"] = analyze_bemt_hover(
            mass_kg=db_drone.mass_kg,
            motor_count=db_drone.motor_count,
            propeller_diameter_in=db_drone.propeller_diameter_in,
            motor_kv=db_drone.motor_kv,
            battery_cells=db_drone.battery_cells,
            air_density=air_density(request.altitude_m, request.temperature_c),
        )

        # WHY THIS IS ADDITIVE TOO: BEMT's own docstring is explicit that
        # its blade geometry/airfoil constants are cited RANGES, not exact
        # values -- this Monte Carlo-propagates that documented uncertainty
        # through to required hover RPM, so the UI can show a confidence
        # interval next to the single bemt_analysis point estimate instead
        # of implying false precision.
        result["hover_uncertainty"] = monte_carlo_hover_uncertainty(
            mass_kg=db_drone.mass_kg,
            motor_count=db_drone.motor_count,
            propeller_diameter_in=db_drone.propeller_diameter_in,
            air_density=air_density(request.altitude_m, request.temperature_c),
        )

        # WHY THIS IS ADDITIVE TOO: extends BEMT to non-zero advance ratio
        # (see the FORWARD FLIGHT / WIND section of bemt.py) to answer
        # "what does station-keeping in this wind actually cost" instead
        # of only ever assuming still air -- only computed when the
        # request actually specifies wind (wind_speed_mps defaults to 0,
        # so every existing caller that doesn't set it gets this field as
        # a deterministic 0%-penalty no-op, never a behavior change).
        result["wind_analysis"] = analyze_bemt_hover_in_wind(
            mass_kg=db_drone.mass_kg,
            motor_count=db_drone.motor_count,
            propeller_diameter_in=db_drone.propeller_diameter_in,
            motor_kv=db_drone.motor_kv,
            battery_cells=db_drone.battery_cells,
            battery_capacity_mah=db_drone.battery_capacity_mah,
            air_density=air_density(request.altitude_m, request.temperature_c),
            wind_speed_mps=request.wind_speed_mps,
        )

        # WHY THIS IS ADDITIVE TOO: propagates gust variability around the
        # requested wind speed (see monte_carlo_uq.py's WIND-SPEED (GUST)
        # UNCERTAINTY section) to give an endurance/power RANGE under
        # realistic gusty conditions, not just the single point estimate
        # above computed at exactly the mean wind speed.
        result["wind_uncertainty"] = monte_carlo_wind_endurance_uncertainty(
            mass_kg=db_drone.mass_kg,
            motor_count=db_drone.motor_count,
            propeller_diameter_in=db_drone.propeller_diameter_in,
            battery_cells=db_drone.battery_cells,
            battery_capacity_mah=db_drone.battery_capacity_mah,
            air_density=air_density(request.altitude_m, request.temperature_c),
            mean_wind_speed_mps=request.wind_speed_mps,
        )
    else:
        result["motor_performance"] = None
        result["bemt_analysis"] = None
        result["hover_uncertainty"] = None
        result["wind_analysis"] = None
        result["wind_uncertainty"] = None

    return result