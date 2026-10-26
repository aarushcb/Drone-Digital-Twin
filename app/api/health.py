"""
D1 Health API -- health spec §9 step 10.

Every endpoint is owner-scoped through the same _get_owned_drone_or_404 the
rest of the API uses, so one user can never reach another's flights, packs
or changes, even by guessing ids.

Uploads are a job the app polls (see services/health/store.py for why):
  POST /drones/{id}/flights            -> 202 {flight_id, status: "queued"}
  GET  /drones/{id}/flights/{fid}      -> status, plus the report once done
  GET  /drones/{id}/trends             -> spec §4.4 series and callouts
  DELETE /drones/{id}/flights/{fid}    -> one flight gone, baselines rebuilt
"""

import os
import tempfile
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.api.routes import _get_owned_drone_or_404
from app.database.database import get_db
from app.models.health import ConfigEpoch, FeatureBaseline, Flight, FlightFeature, Pack
from app.models.user import User
from app.services.health import epochs as E
from app.services.health import store

router = APIRouter()

CHUNK = 1 << 20


# ---------------------------------------------------------------- schemas

class PackIn(BaseModel):
    label: str = Field(min_length=1, max_length=80)
    cell_count: Optional[int] = Field(default=None, ge=1, le=24)
    capacity_mah: Optional[float] = Field(default=None, gt=0)
    serial_number: Optional[str] = None
    first_used_at: Optional[datetime] = None


class PackPatch(BaseModel):
    label: Optional[str] = Field(default=None, min_length=1, max_length=80)
    cell_count: Optional[int] = Field(default=None, ge=1, le=24)
    capacity_mah: Optional[float] = Field(default=None, gt=0)
    serial_number: Optional[str] = None
    first_used_at: Optional[datetime] = None


class FlightPatch(BaseModel):
    pack_id: Optional[int] = None


CHANGE_TYPES = E.CHANGE_TYPES + ("inferred_confirmed",)


class ConfigChangeIn(BaseModel):
    change_type: str
    note: Optional[str] = Field(default=None, max_length=2000)
    # The first flight the change applies to; omitted = the next upload.
    effective_from_flight_id: Optional[int] = None
    # inferred_confirmed only: the features the "Did you change something?"
    # prompt named. Their baselines reset; nothing else does.
    features: Optional[List[str]] = None

    @field_validator("change_type")
    @classmethod
    def _known(cls, v):
        if v not in CHANGE_TYPES:
            raise ValueError(f"change_type must be one of {list(CHANGE_TYPES)}")
        return v


# ---------------------------------------------------------------- helpers

def _pack_json(p: Pack) -> dict:
    return {"id": p.id, "drone_id": p.drone_id, "label": p.label, "cell_count": p.cell_count,
            "capacity_mah": p.capacity_mah, "serial_number": p.serial_number,
            "first_used_at": p.first_used_at, "created_at": p.created_at}


def _owned_pack(db: Session, drone_id: int, pack_id: int) -> Pack:
    p = db.query(Pack).filter_by(id=pack_id, drone_id=drone_id).first()
    if p is None:
        raise HTTPException(status_code=404, detail="Pack not found")
    return p


def _owned_flight(db: Session, drone_id: int, flight_id: int) -> Flight:
    f = db.query(Flight).filter_by(id=flight_id, drone_id=drone_id).first()
    if f is None:
        raise HTTPException(status_code=404, detail="Flight not found")
    return store.reconcile(db, f)


def _flight_json(f: Flight, report: Optional[dict] = None) -> dict:
    out = {"id": f.id, "drone_id": f.drone_id, "status": f.status, "error": f.error,
           "log_filename": f.log_filename, "uploaded_at": f.uploaded_at, "processed_at": f.processed_at,
           "flown_at": f.flown_at, "autopilot": f.autopilot, "pack_id": f.pack_id, "epoch_id": f.epoch_id,
           "qualified": f.qualified, "qualify_reasons": f.qualify_reasons, "airborne_s": f.airborne_s,
           "takeoff_mass_kg": f.takeoff_mass_kg}
    if report is not None:
        out.update({"flight_number": report["run"]["flight_number"], "overall_score": report["overall_score"],
                    "score_band": report["score_band"], "headline": report["verdict"]["headline"],
                    "findings_count": len(report["findings"])})
    return out


def _epoch_json(ep: ConfigEpoch, effective_from: Optional[int] = None) -> dict:
    return {"id": ep.id, "change_type": ep.change_type, "source": ep.source, "note": ep.note,
            "started_at": ep.started_at, "effective_from_flight_id": effective_from,
            "features_reset": sorted(ep.affected_features or [])}


# ---------------------------------------------------------------- flights

@router.post("/drones/{drone_id}/flights", status_code=202)
def upload_flight(
    drone_id: int,
    file: UploadFile = File(...),
    pack_id: Optional[int] = Form(default=None),
    takeoff_mass_kg: Optional[float] = Form(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Store the log and queue it for analysis. Returns at once; poll
    GET /drones/{drone_id}/flights/{flight_id} until status is done or failed."""
    _get_owned_drone_or_404(db, drone_id, current_user)
    name = file.filename or "upload"
    suffix = os.path.splitext(name)[1].lower()
    if suffix not in store.ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported log type {suffix or '(none)'}: "
                            f"upload a PX4 .ulg or ArduPilot .bin (.tlog is not supported yet)")
    if pack_id is not None:
        _owned_pack(db, drone_id, pack_id)
    if takeoff_mass_kg is not None and not (0 < takeoff_mass_kg < 1000):
        raise HTTPException(status_code=422, detail="takeoff_mass_kg must be between 0 and 1000")

    # Stream to our own temp file: the worker runs after this request has
    # closed the upload, and a 60 MB log shouldn't sit in memory twice.
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    size = 0
    try:
        with tmp:
            for chunk in iter(lambda: file.file.read(CHUNK), b""):
                size += len(chunk)
                if size > store.MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=f"Log over {store.MAX_UPLOAD_BYTES // (1 << 20)} MB")
                tmp.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        digest = store.sha256_file(tmp.name)
        existing = db.query(Flight).filter_by(drone_id=drone_id, log_sha256=digest).first()
        if existing is not None:
            store.reconcile(db, existing)
            if existing.status != "failed":
                raise HTTPException(status_code=409, detail={"message": "This log was already uploaded for this drone",
                                                             "flight_id": existing.id})
            db.delete(existing)        # a failed attempt may be retried with the same file
            db.flush()
        flight = Flight(drone_id=drone_id, epoch_id=store.latest_epoch(db, drone_id).id, pack_id=pack_id,
                        uploaded_at=store.now(), log_filename=name, log_sha256=digest, status="queued",
                        takeoff_mass_kg=takeoff_mass_kg)
        db.add(flight)
        db.commit()
    except BaseException:
        os.unlink(tmp.name)
        raise
    store.enqueue(flight.id, tmp.name)
    return {"flight_id": flight.id, "status": "queued", "size_bytes": size,
            "poll": f"/drones/{drone_id}/flights/{flight.id}"}


@router.get("/drones/{drone_id}/flights")
def list_flights(drone_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    _get_owned_drone_or_404(db, drone_id, current_user)
    flights = [store.reconcile(db, f) for f in store.ordered_flights(db, drone_id)]
    reports = store.drone_reports(db, drone_id)
    return [_flight_json(f, reports.get(f.id)) for f in reversed(flights)]   # newest first


@router.get("/drones/{drone_id}/flights/{flight_id}")
def get_flight(drone_id: int, flight_id: int, db: Session = Depends(get_db),
               current_user: User = Depends(get_current_user)):
    """The polling target: status always; the full report once done."""
    _get_owned_drone_or_404(db, drone_id, current_user)
    f = _owned_flight(db, drone_id, flight_id)
    report = store.drone_reports(db, drone_id).get(f.id) if f.status == "done" else None
    return dict(_flight_json(f, report), report=report)


@router.get("/drones/{drone_id}/flights/{flight_id}/report")
def get_report(drone_id: int, flight_id: int, db: Session = Depends(get_db),
               current_user: User = Depends(get_current_user)):
    """200 with the d1.health.1 report; 202 while queued or processing;
    422 if analysis failed (the error says why)."""
    _get_owned_drone_or_404(db, drone_id, current_user)
    f = _owned_flight(db, drone_id, flight_id)
    if f.status in ("queued", "processing"):
        return JSONResponse(status_code=202, content={"flight_id": f.id, "status": f.status})
    if f.status == "failed":
        raise HTTPException(status_code=422, detail={"flight_id": f.id, "status": "failed", "error": f.error})
    return store.drone_reports(db, drone_id)[f.id]


@router.patch("/drones/{drone_id}/flights/{flight_id}")
def update_flight(drone_id: int, flight_id: int, body: FlightPatch, db: Session = Depends(get_db),
                  current_user: User = Depends(get_current_user)):
    """Assign (or clear) the battery pack after upload; power baselines follow it."""
    _get_owned_drone_or_404(db, drone_id, current_user)
    f = _owned_flight(db, drone_id, flight_id)
    if body.pack_id is not None:
        _owned_pack(db, drone_id, body.pack_id)
    f.pack_id = body.pack_id
    db.commit()
    reports = store.drone_reports(db, drone_id)
    store.persist_analysis(db, drone_id, reports)
    db.commit()
    return _flight_json(f, reports.get(f.id))


@router.get("/drones/{drone_id}/trends")
def get_trends(drone_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Spec §4.4: raw feature values per flight against the frozen epoch
    reference, published guide lines, change boundaries and Step/Drift/Guide
    callouts."""
    _get_owned_drone_or_404(db, drone_id, current_user)
    return store.drone_trends(db, drone_id)


@router.delete("/drones/{drone_id}/flights/{flight_id}")
def delete_flight(drone_id: int, flight_id: int, db: Session = Depends(get_db),
                  current_user: User = Depends(get_current_user)):
    """Deletes one flight and its measurements, then rebuilds the drone's
    baselines and admissions from the flights that remain, so its "normal"
    no longer includes this flight. An upload still being analysed for it is
    dropped when its job finds the flight gone (and its file deleted)."""
    _get_owned_drone_or_404(db, drone_id, current_user)
    f = db.query(Flight).filter_by(id=flight_id, drone_id=drone_id).first()
    if f is None:
        raise HTTPException(status_code=404, detail="Flight not found")
    features = db.query(FlightFeature).filter_by(flight_id=flight_id).delete(synchronize_session=False)
    db.delete(f)
    db.commit()
    store.persist_analysis(db, drone_id, store.drone_reports(db, drone_id))
    db.commit()
    return {
        "flight_id": flight_id,
        "deleted": {"flights": 1, "flight_features": features},
        "flights_remaining": db.query(Flight).filter_by(drone_id=drone_id).count(),
        "baselines_rebuilt": db.query(FeatureBaseline).filter_by(drone_id=drone_id).count(),
    }


# ---------------------------------------------------------------- configuration changes

@router.post("/drones/{drone_id}/config-changes", status_code=201)
def declare_change(drone_id: int, body: ConfigChangeIn, db: Session = Depends(get_db),
                   current_user: User = Depends(get_current_user)):
    """'I changed something' (source user), or a Yes to 'Did you change
    something?' (change_type inferred_confirmed with the prompt's features)."""
    _get_owned_drone_or_404(db, drone_id, current_user)
    store.ensure_initial_epoch(db, drone_id)
    if body.change_type == "inferred_confirmed":
        if not body.features:
            raise HTTPException(status_code=422, detail="inferred_confirmed needs the prompt's features")
        unknown = sorted(set(body.features) - E.ALL_FEATURES)
        if unknown:
            raise HTTPException(status_code=422, detail=f"Unknown features: {unknown}")
        affected, source = sorted(set(body.features)), "inferred"
    else:
        if body.features:
            raise HTTPException(status_code=422, detail="features is only for inferred_confirmed")
        affected, source = sorted(E.RESETS[body.change_type]), "user"
    started = store.now()
    if body.effective_from_flight_id is not None:
        started = store._naive(_owned_flight(db, drone_id, body.effective_from_flight_id).uploaded_at)
    ep = ConfigEpoch(drone_id=drone_id, started_at=started, change_type=body.change_type, source=source,
                     note=body.note, affected_features=affected)
    db.add(ep)
    db.commit()
    store.persist_analysis(db, drone_id, store.drone_reports(db, drone_id))
    db.commit()
    return _epoch_json(ep, body.effective_from_flight_id)


@router.get("/drones/{drone_id}/config-changes")
def list_changes(drone_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    _get_owned_drone_or_404(db, drone_id, current_user)
    eps = db.query(ConfigEpoch).filter_by(drone_id=drone_id).order_by(ConfigEpoch.started_at, ConfigEpoch.id).all()
    return [_epoch_json(ep) for ep in eps]


# ---------------------------------------------------------------- packs

@router.post("/drones/{drone_id}/packs", status_code=201)
def create_pack(drone_id: int, body: PackIn, db: Session = Depends(get_db),
                current_user: User = Depends(get_current_user)):
    _get_owned_drone_or_404(db, drone_id, current_user)
    p = Pack(drone_id=drone_id, created_at=store.now(), **body.model_dump())
    db.add(p)
    db.commit()
    return _pack_json(p)


@router.get("/drones/{drone_id}/packs")
def list_packs(drone_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    _get_owned_drone_or_404(db, drone_id, current_user)
    return [_pack_json(p) for p in db.query(Pack).filter_by(drone_id=drone_id).order_by(Pack.id)]


@router.patch("/drones/{drone_id}/packs/{pack_id}")
def update_pack(drone_id: int, pack_id: int, body: PackPatch, db: Session = Depends(get_db),
                current_user: User = Depends(get_current_user)):
    _get_owned_drone_or_404(db, drone_id, current_user)
    p = _owned_pack(db, drone_id, pack_id)
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(p, k, v)
    db.commit()
    return _pack_json(p)


@router.delete("/drones/{drone_id}/packs/{pack_id}", status_code=204)
def delete_pack(drone_id: int, pack_id: int, db: Session = Depends(get_db),
                current_user: User = Depends(get_current_user)):
    """Flights keep their data; their pack becomes unknown (the DB does this
    too on Postgres via ON DELETE SET NULL; done here so SQLite agrees)."""
    _get_owned_drone_or_404(db, drone_id, current_user)
    p = _owned_pack(db, drone_id, pack_id)
    db.query(Flight).filter_by(drone_id=drone_id, pack_id=pack_id).update({"pack_id": None})
    db.delete(p)
    db.commit()
    store.persist_analysis(db, drone_id, store.drone_reports(db, drone_id))
    db.commit()
