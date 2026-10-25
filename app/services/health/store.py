"""
Database side of the health pipeline -- build step 10.

Flights are analysed by a single background worker (one at a time), never in
the request: the upload returns as soon as the log is on disk, and the app
polls the flight. Why, in short (the full reasoning is in the step 10
validation notes): on Render's free tier the slow, fragile part is the phone
holding a connection open, CPU is a fraction of a core, and two 25-60 MB
parses at once would approach the 512 MB memory limit. One worker serialises
parsing; queued uploads say "queued" instead of hanging.

Reports are not stored. They are recomputed from flight_features on every
fetch -- milliseconds for a drone's whole history -- so a change declared
after the fact (or a pack assigned later) is always reflected, and there is
no stale copy to invalidate. flight_features (with each value's admission)
and feature_baselines (latest rolling/reference per group) are written back
after every analysis, as the spec §9 step 6 source of truth and cache.
"""

import hashlib
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from app.database.database import SessionLocal
from app.models.health import ConfigEpoch, FeatureBaseline, Flight, FlightFeature
from app.services.reality_gap.parsers import LogParseError, parse_log

from . import epochs as E
from . import norms as N
from . import report as R
from .features import Feature, extract_features
from .windowing import MIN_STEADY_S, FlightWindow, window_flight

log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 100 * 1024 * 1024   # the larger real fixture is 60 MB
ALLOWED_SUFFIXES = (".ulg", ".bin")    # parse_log: ULog and DataFlash; .tlog is a tracked gap
INTERRUPTED = "interrupted: the server restarted before analysis finished; upload the log again"

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="health-worker")
_active: set = set()                    # flight ids queued or running in THIS process
_active_lock = threading.Lock()


def now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)   # naive UTC, as SQLite and Postgres return it


def _naive(dt: Optional[datetime]) -> Optional[datetime]:
    return dt.replace(tzinfo=None) if dt is not None and dt.tzinfo is not None else dt


def _jsonable(obj):
    return R._jsonable(obj)


# ---------------------------------------------------------------- epochs

def ensure_initial_epoch(db: Session, drone_id: int) -> ConfigEpoch:
    ep = (db.query(ConfigEpoch).filter_by(drone_id=drone_id, change_type="initial")
          .order_by(ConfigEpoch.id).first())
    if ep is None:
        ep = ConfigEpoch(drone_id=drone_id, started_at=now(), change_type="initial", source="initial",
                         affected_features=sorted(E.ALL_FEATURES))
        db.add(ep)
        db.flush()
    return ep


def latest_epoch(db: Session, drone_id: int) -> ConfigEpoch:
    ensure_initial_epoch(db, drone_id)
    return (db.query(ConfigEpoch).filter_by(drone_id=drone_id)
            .order_by(ConfigEpoch.started_at.desc(), ConfigEpoch.id.desc()).first())


def ordered_flights(db: Session, drone_id: int) -> List[Flight]:
    """Flight order is upload order. ponytail: log start times are often
    missing (the reference log says 1970), so a log uploaded out of order is
    judged in upload order; sort by flown_at once GPS time is reliable."""
    return db.query(Flight).filter_by(drone_id=drone_id).order_by(Flight.uploaded_at, Flight.id).all()


def declared_events(db: Session, drone_id: int, flights: List[Flight]) -> List[E.ConfigEvent]:
    """config_epochs rows -> events. A change applies from the first flight
    uploaded at or after its started_at."""
    out = []
    for ep in db.query(ConfigEpoch).filter_by(drone_id=drone_id).order_by(ConfigEpoch.started_at, ConfigEpoch.id):
        if ep.change_type == "initial":
            continue
        idx = next((i for i, f in enumerate(flights) if _naive(f.uploaded_at) >= _naive(ep.started_at)), len(flights))
        feats = set(ep.affected_features or []) if ep.change_type == "inferred_confirmed" else None
        out.append(E.ConfigEvent(idx, ep.change_type, ep.source, note=ep.note or "", features=feats, epoch_id=ep.id))
    return out


# ---------------------------------------------------------------- rows <-> pipeline

def _window_of(fl: Flight) -> FlightWindow:
    w = FlightWindow(qualified=bool(fl.qualified), reasons=list(fl.qualify_reasons or []),
                     airborne_s=fl.airborne_s or 0.0, airborne_source=fl.airborne_source or "none",
                     steady_s=fl.steady_s or 0.0, phase_seconds=dict(fl.phase_seconds or {}))
    if fl.airborne_start_s is not None:
        w.airborne_start_us, w.airborne_end_us = int(fl.airborne_start_s * 1e6), int(fl.airborne_end_s * 1e6)
    w.steady_qualified = w.qualified and w.steady_s >= MIN_STEADY_S
    return w


def _inputs(db: Session, flights: List[Flight]) -> List[R.FlightInput]:
    rows: Dict[int, List[FlightFeature]] = {}
    ids = [f.id for f in flights]
    if ids:
        for ff in db.query(FlightFeature).filter(FlightFeature.flight_id.in_(ids)):
            rows.setdefault(ff.flight_id, []).append(ff)
    out = []
    for fl in flights:
        feats = [Feature(r.feature_key, r.subsystem, r.tier, r.unit or "", value=r.value, label=r.label,
                         reason=r.reason, detail=dict(r.detail or {})) for r in rows.get(fl.id, [])]
        out.append(R.FlightInput(fl.id, fl.autopilot, feats, _window_of(fl), pack_id=fl.pack_id,
                                 watched=dict(fl.watched_params or {}), log_meta=dict(fl.log_meta or {}),
                                 warnings=[]))   # warnings are stored already merged with the window's
    return out


def drone_reports(db: Session, drone_id: int) -> Dict[int, dict]:
    """flight id -> d1.health.1 report, for every analysed flight."""
    done = [f for f in ordered_flights(db, drone_id) if f.status == "done"]
    reports = R.analyse(_inputs(db, done), declared_events(db, drone_id, done))
    by_id = {}
    for fl, rep in zip(done, reports):
        rep["warnings"] = list(fl.warnings or []) + [w for w in rep["warnings"] if w not in (fl.warnings or [])]
        by_id[fl.id] = rep
    return by_id


def persist_analysis(db: Session, drone_id: int, reports: Dict[int, dict]) -> None:
    """Write admissions back to flight_features, and rebuild the baseline
    cache: per (group, feature), the rolling and reference baselines over
    every admitted value in that group *including* the newest flight -- the
    state the next upload will be judged against."""
    from .baseline import Observation, reference_baseline, rolling_baseline
    db.query(FeatureBaseline).filter_by(drone_id=drone_id).delete()
    if not reports:
        return
    admitted: Dict[tuple, List[Observation]] = {}
    for fid, rep in reports.items():           # reports are in flight order
        by_key = {f["key"]: f for f in rep["features"]}
        for ff in db.query(FlightFeature).filter_by(flight_id=fid):
            f = by_key.get(ff.feature_key)
            if f is None:
                continue
            ff.admitted, ff.quarantine_reason, ff.baseline_group = f["admitted"], f["quarantine_reason"], f["group"]
            if f["admitted"] and f["value"] is not None:
                admitted.setdefault((f["group"], f["key"]), []).append(Observation(fid, f["value"]))
    for (group, key), obs in admitted.items():
        for kind, b in (("rolling", rolling_baseline(obs, key)), ("reference", reference_baseline(obs, key))):
            if b is not None:
                db.add(FeatureBaseline(drone_id=drone_id, baseline_group=group, feature_key=key, kind=kind,
                                       median=b.median, mad=b.mad, spread=b.spread, n=b.n, flight_ids=b.flight_ids,
                                       computed_at=now()))


# ---------------------------------------------------------------- upload job

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def enqueue(flight_id: int, path: str) -> None:
    with _active_lock:
        _active.add(flight_id)
    _executor.submit(_run, flight_id, path)


def is_active(flight_id: int) -> bool:
    with _active_lock:
        return flight_id in _active


def reconcile(db: Session, flight: Flight) -> Flight:
    """A queued/processing flight this process isn't working on was lost to a
    restart (Render redeploys and spin-downs kill in-process work). Say so,
    rather than letting the app poll forever."""
    if flight.status in ("queued", "processing") and not is_active(flight.id):
        flight.status, flight.error, flight.processed_at = "failed", INTERRUPTED, now()
        db.commit()
    return flight


def _run(flight_id: int, path: str) -> None:
    db = SessionLocal()
    try:
        fl = db.get(Flight, flight_id)
        if fl is None:
            return
        fl.status = "processing"
        db.commit()
        try:
            _analyse_log(db, fl, path)
            fl.status, fl.error = "done", None
        except LogParseError as exc:
            db.rollback()
            fl = db.get(Flight, flight_id)
            # Say it with the user's filename, not our temp file's.
            fl.status, fl.error = "failed", f"{type(exc).__name__}: {exc}".replace(os.path.basename(path), fl.log_filename)
        fl.processed_at = now()
        db.commit()
        if fl.status == "done":
            persist_analysis(db, fl.drone_id, drone_reports(db, fl.drone_id))
            db.commit()
    except Exception as exc:  # noqa: BLE001 -- a crash must end as a visible failed job, never a silent hang
        log.exception("health job %s crashed", flight_id)
        db.rollback()
        fl = db.get(Flight, flight_id)
        if fl is not None:
            fl.status, fl.error, fl.processed_at = "failed", f"internal_error: {type(exc).__name__}", now()
            db.commit()
    finally:
        db.close()
        with _active_lock:
            _active.discard(flight_id)
        try:
            os.unlink(path)
        except OSError:
            pass


def _analyse_log(db: Session, fl: Flight, path: str) -> None:
    parsed = parse_log(path)
    w = window_flight(parsed)
    feats = extract_features(parsed, w)
    m = parsed.meta
    start = m.start_time_utc
    flown = None
    if start and not start.startswith("1970"):      # epoch 0 means the log had no real clock
        flown = _naive(datetime.fromisoformat(start))
    fl.autopilot = N.autopilot_of(m.source_format)
    fl.firmware_version, fl.airframe, fl.flown_at = m.firmware_version, m.airframe, flown
    fl.qualified, fl.qualify_reasons = w.qualified, list(w.reasons)
    fl.airborne_s, fl.steady_s, fl.airborne_source = w.airborne_s, w.steady_s, w.airborne_source
    if w.airborne_start_us is not None:
        fl.airborne_start_s, fl.airborne_end_s = w.airborne_start_us / 1e6, w.airborne_end_us / 1e6
    fl.phase_seconds = _jsonable(w.phase_seconds)
    fl.topic_info, fl.motor_info = _jsonable(parsed.topic_info), _jsonable(parsed.motor_info)
    fl.watched_params = _jsonable(E.watched_params(parsed.params, m.firmware_version))
    fl.warnings = list(parsed.warnings) + list(w.warnings)
    fl.log_meta = {"filename": fl.log_filename, "sha256": m.sha256, "source_format": m.source_format,
                   "size_bytes": os.path.getsize(path), "hardware": m.hardware, "firmware_release": m.firmware_release,
                   "duration_s": round(m.duration_s, 2), "firmware_version": m.firmware_version,
                   "airframe": m.airframe, "start_time_utc": start}
    db.query(FlightFeature).filter_by(flight_id=fl.id).delete()
    for f in feats:
        db.add(FlightFeature(flight_id=fl.id, feature_key=f.key, subsystem=f.subsystem, tier=f.tier, unit=f.unit,
                             value=None if f.value is None else float(f.value), label=f.label, reason=f.reason,
                             detail=_jsonable(f.detail)))
    db.flush()
