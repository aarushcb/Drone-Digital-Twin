"""
WHY THIS CHANGED:
The original get_telemetry_by_drone() returned EVERY record for a drone,
with no limit. That's fine with 10 test rows, but a real drone streaming
telemetry every second will have tens of thousands of rows within a day —
an unbounded query like that will eventually slow the API to a crawl and
send huge payloads to the phone app. This version adds:
- limit/offset: standard pagination, so the app requests data in pages.
- start/end: optional date-range filtering, needed for "show me last
  Tuesday's flight" style historical replay (mentioned in your vision doc).
- newest-first ordering: dashboards almost always want the latest reading
  first.
"""

from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime

from app.models.telemetry import Telemetry
from app.schemas.telemetry import TelemetryCreate


def create_telemetry(db: Session, telemetry: TelemetryCreate, drone_id: int) -> Telemetry:
    db_telemetry = Telemetry(**telemetry.model_dump(), drone_id=drone_id)
    db.add(db_telemetry)
    db.commit()
    db.refresh(db_telemetry)
    return db_telemetry


def get_telemetry(
    db: Session,
    drone_id: int,
    limit: int = 100,
    offset: int = 0,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> list[Telemetry]:
    query = db.query(Telemetry).filter(Telemetry.drone_id == drone_id)

    if start is not None:
        query = query.filter(Telemetry.timestamp >= start)
    if end is not None:
        query = query.filter(Telemetry.timestamp <= end)

    return (
        query.order_by(Telemetry.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_control_loop_samples(db: Session, drone_id: int, limit: int = 300) -> list[Telemetry]:
    """
    Returns the most recent telemetry rows for a drone that actually carry
    a REAL attitude setpoint (desired_roll/pitch/yaw) and actual attitude
    -- i.e. rows that came from a real/MAVLink-imported flight with
    ATTITUDE_TARGET data (see app/services/mavlink_import.py), not the
    ordinary actual-only telemetry every reading has always had. Used by
    the control-loop visualization (app/services/control_loop.py) to show
    real desired-vs-actual data when it exists, in chronological order.
    """
    rows = (
        db.query(Telemetry)
        .filter(
            Telemetry.drone_id == drone_id,
            Telemetry.desired_roll.isnot(None),
            Telemetry.desired_pitch.isnot(None),
            Telemetry.desired_yaw.isnot(None),
            Telemetry.roll.isnot(None),
            Telemetry.pitch.isnot(None),
            Telemetry.yaw.isnot(None),
        )
        .order_by(Telemetry.timestamp.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(rows))


def get_calibration_imu_samples(db: Session, drone_id: int, limit: int = 200) -> list[Telemetry]:
    """
    Returns the most recent telemetry rows for a drone that carry REAL
    raw IMU data (accel_x/gyro_x -- from a MAVLink log's SCALED_IMU/
    RAW_IMU messages, see app/services/mavlink_import.py) AND were logged
    while the drone was NOT flying (flight_state == "idle").

    WHY THE flight_state FILTER MATTERS (this is a correctness
    requirement, not just a nice-to-have): app/services/sensor_calibration.py's
    accelerometer/gyroscope checks assume the sensor was STATIONARY when
    the readings were captured (a perfect accelerometer reads exactly
    (0,0,+9.81) at rest; a perfect gyroscope reads exactly 0 deg/s on
    every axis at rest) -- that's the actual physical basis of a static
    IMU calibration check. Raw IMU rows captured DURING FLIGHT reflect
    real maneuvering acceleration/rotation, not sensor bias/noise, and
    running the stationary check on them would produce a meaningless (or
    actively misleading) "FAIL" on a perfectly good sensor. flight_state
    == "idle" (disarmed -- see mavlink_import.py's HEARTBEAT-derived
    flight_state) is the closest proxy this schema has for "the drone was
    genuinely sitting still," matching how real static IMU calibration is
    actually performed (disarmed, on a bench).
    """
    rows = (
        db.query(Telemetry)
        .filter(
            Telemetry.drone_id == drone_id,
            Telemetry.flight_state == "idle",
            (Telemetry.accel_x.isnot(None)) | (Telemetry.gyro_x.isnot(None)),
        )
        .order_by(Telemetry.timestamp.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(rows))


def bulk_create_telemetry(db: Session, points: list[dict], drone_id: int) -> int:
    """
    Inserts many telemetry rows in one transaction -- used by MAVLink log
    import, where a single flight log can produce hundreds or thousands of
    points. Doing this one row at a time through create_telemetry() (each
    with its own commit + refresh) would be extremely slow for a real log;
    bulk_insert_mappings does one efficient batch INSERT instead. Returns
    the number of rows inserted.
    """
    for p in points:
        p["drone_id"] = drone_id
    db.bulk_insert_mappings(Telemetry, points)
    db.commit()
    return len(points)


def delete_all_telemetry(db: Session, drone_id: int) -> int:
    """
    Deletes every telemetry reading logged for a drone -- used by the
    "Clear history" feature, so testing doesn't leave permanently
    accumulating scattered data (this is exactly what caused the tangled,
    multi-session flight paths in Flight Verification before). Returns
    the number of rows deleted, mainly useful for confirming the delete
    actually did something.
    """
    deleted_count = db.query(Telemetry).filter(Telemetry.drone_id == drone_id).delete()
    db.commit()
    return deleted_count