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

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime

from app.database.database import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.drone import DroneCreate, DroneOut, DroneUpdate
from app.schemas.telemetry import TelemetryCreate, TelemetryOut
from app.schemas.scene_object import SceneObjectCreate, SceneObjectOut
from app.schemas.path_plan import PathPlanRequest, PathPlanResponse, PathPoint
from app.crud import drone as drone_crud
from app.crud import telemetry as telemetry_crud
from app.crud import scene_object as scene_object_crud
from app.services.health_monitor import get_health_status
from app.services.alerts import generate_alerts
from app.services.drone_analytics import calculate_risk_score, calculate_analytics
from app.services.telemetry_broadcaster import manager
from app.services.path_planner import plan_path, path_distance_meters

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


# ---------- Status / analytics (existing rule-based logic, now owner-scoped) ----------

@router.get("/drones/{drone_id}/status")
def get_drone_status(
    drone_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_drone_or_404(db, drone_id, current_user)

    latest_records = telemetry_crud.get_telemetry(db, drone_id, limit=1)
    if not latest_records:
        raise HTTPException(status_code=404, detail="No telemetry found for drone")

    latest = latest_records[0]

    health = get_health_status(latest.battery, latest.temperature)
    alerts = generate_alerts(latest.battery, latest.temperature)
    risk_score = calculate_risk_score(latest.battery, latest.temperature, latest.speed)

    return {
        "drone_id": drone_id,
        "health": health,
        "risk_score": risk_score,
        "alerts": alerts,
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
