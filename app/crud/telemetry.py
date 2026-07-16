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
