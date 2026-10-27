"""
D1 Health, build step 10: the health API end to end.

Run:  python3 tests/test_health_api.py

FastAPI TestClient against a temporary SQLite database built by
`alembic upgrade head` (never Neon), with real registration and login.
Upload tests use the gitignored real fixtures and skip when absent.
"""

import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TMP = tempfile.mkdtemp(prefix="health_api_")
DB = os.path.join(TMP, "api.db")
os.environ["DATABASE_URL"] = f"sqlite:///{DB}"
sys.path.insert(0, ROOT)

_mig = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, capture_output=True, text=True,
                      env=dict(os.environ))
assert _mig.returncode == 0, _mig.stderr[-800:]

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402
from app.database.database import SessionLocal  # noqa: E402
from app.models.health import FeatureBaseline, Flight, FlightFeature  # noqa: E402
from app.services.health import store  # noqa: E402

REAL_DIR = os.path.join(ROOT, "tests", "fixtures", "reality_gap", "real")
REAL1 = os.path.join(REAL_DIR, "px4_777ed799.ulg")
REAL2 = os.path.join(REAL_DIR, "px4_641d242a.ulg")
client = TestClient(app)
_state = {}


def user(tag):
    email = f"{tag}_{int(time.time() * 1000)}@test.com"
    assert client.post("/auth/register", json={"email": email, "password": "pw123456"}).status_code == 201
    tok = client.post("/auth/login", data={"username": email, "password": "pw123456"}).json()["access_token"]
    h = {"Authorization": f"Bearer {tok}"}
    drone = client.post("/drones", json={"name": f"{tag} hex"}, headers=h).json()
    return h, drone["id"]


def owner():
    if "owner" not in _state:
        _state["owner"] = user("owner")
    return _state["owner"]


def upload(h, d, path, name=None, **form):
    with open(path, "rb") as fh:
        return client.post(f"/drones/{d}/flights", headers=h, files={"file": (name or os.path.basename(path), fh)},
                           data={k: str(v) for k, v in form.items()})


def wait(h, d, fid, timeout=120.0):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        r = client.get(f"/drones/{d}/flights/{fid}", headers=h).json()
        if r["status"] in ("done", "failed"):
            return r, time.perf_counter() - t0
        time.sleep(0.1)
    raise AssertionError(f"flight {fid} still {r['status']} after {timeout}s")


def garbage(name="junk.ulg", content=b"not a log at all " * 64):
    p = os.path.join(TMP, name)
    with open(p, "wb") as fh:
        fh.write(content)
    return p


# ---------------------------------------------------------------- auth and ownership

def test_every_endpoint_requires_auth():
    _, d = owner()
    calls = [("post", f"/drones/{d}/flights"), ("get", f"/drones/{d}/flights"), ("get", f"/drones/{d}/flights/1"),
             ("get", f"/drones/{d}/flights/1/report"), ("patch", f"/drones/{d}/flights/1"),
             ("post", f"/drones/{d}/config-changes"), ("get", f"/drones/{d}/config-changes"),
             ("post", f"/drones/{d}/packs"), ("get", f"/drones/{d}/packs"), ("patch", f"/drones/{d}/packs/1"),
             ("delete", f"/drones/{d}/packs/1"), ("get", f"/drones/{d}/trends"),
             ("delete", f"/drones/{d}/flights/1")]
    for method, path in calls:
        assert getattr(client, method)(path).status_code == 401, (method, path)
    print(f"PASS: all {len(calls)} health endpoints return 401 without a token")


def test_other_users_get_404_everywhere():
    h, d = owner()
    pack = client.post(f"/drones/{d}/packs", headers=h, json={"label": "A"}).json()
    hx, dx = user("intruder")
    for method, path, kw in [("get", f"/drones/{d}/flights", {}), ("get", f"/drones/{d}/packs", {}),
                             ("get", f"/drones/{d}/trends", {}),
                             ("get", f"/drones/{d}/config-changes", {}),
                             ("post", f"/drones/{d}/config-changes", {"json": {"change_type": "props"}}),
                             ("patch", f"/drones/{d}/packs/{pack['id']}", {"json": {"label": "x"}}),
                             ("get", f"/drones/{dx}/packs/{pack['id']}", {})]:
        r = getattr(client, method)(path, headers=hx, **kw)
        assert r.status_code in (404, 405), (method, path, r.status_code)
    assert client.patch(f"/drones/{dx}/packs/{pack['id']}", headers=hx, json={"label": "x"}).status_code == 404
    with open(garbage(), "rb") as fh:
        r = client.post(f"/drones/{dx}/flights", headers=hx, files={"file": ("a.ulg", fh)}, data={"pack_id": pack["id"]})
    assert r.status_code == 404                     # can't attach someone else's pack to your own flight
    print("PASS: another user gets 404 on this drone's flights, packs and changes, and can't borrow its pack")


# ---------------------------------------------------------------- packs

def test_pack_crud_and_validation():
    h, d = owner()
    r = client.post(f"/drones/{d}/packs", headers=h, json={"label": "Pack B", "cell_count": 12, "capacity_mah": 25000})
    assert r.status_code == 201 and r.json()["cell_count"] == 12
    pid = r.json()["id"]
    assert client.post(f"/drones/{d}/packs", headers=h, json={"label": "", "cell_count": 0}).status_code == 422
    r = client.patch(f"/drones/{d}/packs/{pid}", headers=h, json={"label": "Pack B (old)"})
    assert r.status_code == 200 and r.json()["label"] == "Pack B (old)" and r.json()["cell_count"] == 12
    assert any(p["id"] == pid for p in client.get(f"/drones/{d}/packs", headers=h).json())
    assert client.delete(f"/drones/{d}/packs/{pid}", headers=h).status_code == 204
    assert client.delete(f"/drones/{d}/packs/{pid}", headers=h).status_code == 404
    print("PASS: packs create (201) / list / patch (partial) / delete (204, then 404); bad input -> 422")


# ---------------------------------------------------------------- upload validation and jobs

def test_upload_rejections():
    h, d = owner()
    assert upload(h, d, garbage("x.tlog")).status_code == 400            # tlog: tracked gap
    assert upload(h, d, garbage("x.csv")).status_code == 400
    assert upload(h, d, garbage("empty.ulg", b"")).status_code == 400
    assert upload(h, d, garbage(), pack_id=999999).status_code == 404
    assert upload(h, d, garbage(), takeoff_mass_kg=-1).status_code == 422
    old = store.MAX_UPLOAD_BYTES
    store.MAX_UPLOAD_BYTES = 100
    try:
        assert upload(h, d, garbage("big.ulg")).status_code == 413
    finally:
        store.MAX_UPLOAD_BYTES = old
    print("PASS: .tlog/.csv -> 400, empty -> 400, foreign pack -> 404, bad mass -> 422, oversize -> 413")


def test_unparseable_log_fails_visibly_and_can_be_retried():
    h, d = owner()
    p = garbage("broken.ulg")
    r = upload(h, d, p)
    assert r.status_code == 202 and r.json()["status"] == "queued"
    f, _ = wait(h, d, r.json()["flight_id"])
    assert f["status"] == "failed" and f["error"] and f["report"] is None, f
    assert "broken.ulg" in f["error"] and "tmp" not in f["error"], f["error"]
    rep = client.get(f"/drones/{d}/flights/{f['id']}/report", headers=h)
    assert rep.status_code == 422 and rep.json()["detail"]["error"] == f["error"]
    r2 = upload(h, d, p)                                                   # same bytes: a failed attempt may retry
    assert r2.status_code == 202 and r2.json()["status"] == "queued"      # (SQLite may reuse the freed id)
    again, _ = wait(h, d, r2.json()["flight_id"])
    assert again["status"] == "failed"
    print(f"PASS: unparseable log -> failed with \"{f['error'][:60]}...\", report 422; same file may be retried")


def test_restart_mid_job_is_reported_not_hung():
    h, d = owner()
    db = SessionLocal()
    ep = store.latest_epoch(db, d)
    fl = Flight(drone_id=d, epoch_id=ep.id, uploaded_at=store.now(), log_filename="lost.ulg", log_sha256="f" * 64,
                status="processing")
    db.add(fl)
    db.commit()
    fid = fl.id
    db.close()
    r = client.get(f"/drones/{d}/flights/{fid}", headers=h).json()
    assert r["status"] == "failed" and r["error"] == store.INTERRUPTED
    print("PASS: a job lost to a restart (processing, but not running here) reads as failed/interrupted")


# ---------------------------------------------------------------- the real log, end to end

def _skip():
    if not os.path.exists(REAL1):
        print("SKIP: real fixture absent")
        return True
    return False


def test_real_log_upload_poll_report():
    if _skip():
        return
    h, d = user("pilot")
    pack = client.post(f"/drones/{d}/packs", headers=h, json={"label": "Pack A", "cell_count": 12}).json()
    t0 = time.perf_counter()
    r = upload(h, d, REAL1, pack_id=pack["id"])
    t_upload = time.perf_counter() - t0
    assert r.status_code == 202, r.text
    fid = r.json()["flight_id"]
    f, t_job = wait(h, d, fid)
    assert f["status"] == "done", f
    # The flight's real date, from the GPS clock: Flight Review lists this log's
    # "Logging Start" as 1788582711 (2026-09-05 04:31:51 UTC).
    assert str(f["flown_at"]).startswith("2026-09-05T04:31:5"), f["flown_at"]
    assert f["report"]["run"]["uploaded_at"].endswith("+00:00")            # the fallback date, marked UTC
    rep = f["report"]
    assert rep["schema_version"] == "d1.health.1" and rep["run"]["flight_number"] == 1 and rep["run"]["pack_id"] == pack["id"]
    assert rep["overall_score"] is None and rep["verdict"]["headline"] == "First flight logged: 8 of 8 checks OK"
    assert rep["real_log_meta"]["filename"] == "px4_777ed799.ulg" and rep["flight_window"]["airborne_source"] == "land_detector"
    for gone in ("sim_log_meta", "alignment", "actuator_lag", "resource_usage", "hints", "signals"):
        assert gone not in rep, gone                                       # §4.3: retired fields stay retired
    for new in ("features", "findings", "checks", "baseline", "logging_gaps", "flight_window", "coverage"):
        assert new in rep, new
    same = client.get(f"/drones/{d}/flights/{fid}/report", headers=h)
    assert same.status_code == 200 and same.json()["findings"] == rep["findings"]
    listed = client.get(f"/drones/{d}/flights", headers=h).json()
    assert listed[0]["id"] == fid and listed[0]["headline"] == rep["verdict"]["headline"]
    dup = upload(h, d, REAL1)
    assert dup.status_code == 409 and dup.json()["detail"]["flight_id"] == fid
    db = SessionLocal()
    rows = db.query(FlightFeature).filter_by(flight_id=fid).all()
    assert len(rows) == 26 and any(r.admitted for r in rows)
    db.close()
    _state["pilot"] = (h, d, fid, pack["id"])
    print(f"PASS: real 26.7 MB log -> 202 in {t_upload:.2f}s, done after {t_job:.2f}s of polling; report "
          f"d1.health.1, \"{rep['verdict']['headline']}\"; retired §4.3 fields absent; duplicate -> 409")


def test_second_flight_change_declaration_and_pack_reassignment():
    if _skip() or "pilot" not in _state:
        return
    if not os.path.exists(REAL2):
        print("SKIP: second real fixture absent")
        return
    h, d, f1, pack = _state["pilot"]
    r = upload(h, d, REAL2)
    f2, t_job = wait(h, d, r.json()["flight_id"])
    assert f2["status"] == "done" and f2["report"]["run"]["flight_number"] == 2
    detected = f2["report"]["epoch_events"]
    # A different airframe's log: the watched parameters differ, so change detection fires.
    assert detected and all(e["source"] == "params" for e in detected), detected

    bad = client.post(f"/drones/{d}/config-changes", headers=h, json={"change_type": "new_wings"})
    assert bad.status_code == 422
    assert client.post(f"/drones/{d}/config-changes", headers=h,
                       json={"change_type": "inferred_confirmed"}).status_code == 422
    assert client.post(f"/drones/{d}/config-changes", headers=h,
                       json={"change_type": "inferred_confirmed", "features": ["vib.nope"]}).status_code == 422
    ch = client.post(f"/drones/{d}/config-changes", headers=h,
                     json={"change_type": "props", "note": "new 28-inch props", "effective_from_flight_id": f2["id"]})
    assert ch.status_code == 201 and "vib.metric_p50" in ch.json()["features_reset"] and "gps.eph_p50" not in ch.json()["features_reset"]
    rep2 = client.get(f"/drones/{d}/flights/{f2['id']}/report", headers=h).json()
    assert any(e["source"] == "user" and e["change_type"] == "props" for e in rep2["epoch_events"])
    vib = next(x for x in rep2["features"] if x["key"] == "vib.metric_p50")
    assert vib["n"] == 1 and vib["baseline"] is None                      # reset by the declared change
    inf = client.post(f"/drones/{d}/config-changes", headers=h,
                      json={"change_type": "inferred_confirmed", "features": ["act.hover"]})
    assert inf.status_code == 201 and inf.json()["source"] == "inferred" and inf.json()["features_reset"] == ["act.hover"]
    kinds = [c["change_type"] for c in client.get(f"/drones/{d}/config-changes", headers=h).json()]
    assert kinds[0] == "initial" and "props" in kinds and "inferred_confirmed" in kinds

    p = client.patch(f"/drones/{d}/flights/{f2['id']}", headers=h, json={"pack_id": pack})
    assert p.status_code == 200 and p.json()["pack_id"] == pack
    rep2 = client.get(f"/drones/{d}/flights/{f2['id']}/report", headers=h).json()
    r_cell = next(x for x in rep2["features"] if x["key"] == "pwr.r_cell")
    assert r_cell["group"].startswith(f"pack:{pack}/")   # same pack, but the detected pack/frame change starts a new group
    assert client.patch(f"/drones/{d}/flights/{f2['id']}", headers=h, json={"pack_id": 999999}).status_code == 404
    tr = client.get(f"/drones/{d}/trends", headers=h)
    assert tr.status_code == 200 and tr.json()["flights"] == 2
    assert [c["key"] for c in tr.json()["cards"]] == ["vibration", "motors", "power"]
    assert client.get(f"/drones/{d}/trends").status_code == 401
    db = SessionLocal()
    cached = db.query(FeatureBaseline).filter_by(drone_id=d).count()
    db.close()
    assert cached > 0
    assert client.delete(f"/drones/{d}/packs/{pack}", headers=h).status_code == 204
    assert client.get(f"/drones/{d}/flights/{f1}", headers=h).json()["pack_id"] is None
    print(f"PASS: 60 MB second log done in {t_job:.2f}s; param change detected ({detected[0]['change_type']}); "
          f"retroactive props change resets vibration (n back to 1); inferred_confirmed resets only act.hover; "
          f"pack reassigned then deleted (flights keep data, pack -> null); baseline cache {cached} rows")


def test_deleting_one_flight_removes_it_and_rebuilds_the_baselines():
    if _skip() or not os.path.exists(REAL2):
        print("SKIP: real fixtures absent")
        return
    h, d = user("pruner")
    f1, _ = wait(h, d, upload(h, d, REAL1).json()["flight_id"])
    f2, _ = wait(h, d, upload(h, d, REAL2).json()["flight_id"])
    assert f1["status"] == f2["status"] == "done"
    db = SessionLocal()
    used_f2 = [b for b in db.query(FeatureBaseline).filter_by(drone_id=d) if f2["id"] in b.flight_ids]
    db.close()
    assert used_f2, "precondition: the second flight is part of some baseline"

    hx, _ = user("snoop")
    assert client.delete(f"/drones/{d}/flights/{f2['id']}", headers=hx).status_code == 404
    r = client.delete(f"/drones/{d}/flights/{f2['id']}", headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["deleted"] == {"flights": 1, "flight_features": 26} and body["flights_remaining"] == 1

    db = SessionLocal()
    assert db.query(FlightFeature).filter_by(flight_id=f2["id"]).count() == 0
    rebuilt = db.query(FeatureBaseline).filter_by(drone_id=d).all()
    db.close()
    assert rebuilt and all(f2["id"] not in b.flight_ids for b in rebuilt)       # the normal no longer includes it
    assert all(b.flight_ids == [f1["id"]] for b in rebuilt if b.kind == "rolling")
    assert client.get(f"/drones/{d}/flights/{f2['id']}", headers=h).status_code == 404
    listed = client.get(f"/drones/{d}/flights", headers=h).json()
    assert [f["id"] for f in listed] == [f1["id"]] and listed[0]["flight_number"] == 1
    assert client.get(f"/drones/{d}/trends", headers=h).json()["flights"] == 1
    assert client.delete(f"/drones/{d}/flights/{f2['id']}", headers=h).status_code == 404
    print(f"PASS: deleting flight 2 removes it and its 26 features; {len(rebuilt)} baseline rows rebuilt from "
          f"flight 1 alone; another user gets 404")


# ---------------------------------------------------------------- schema guard

def test_create_all_never_creates_health_tables():
    fresh = os.path.join(TMP, "fresh.db")
    code = ("import main, sqlite3, sys; t={r[0] for r in sqlite3.connect(sys.argv[1]).execute("
            "\"select name from sqlite_master where type='table'\")}; print(sorted(t))")
    out = subprocess.run([sys.executable, "-c", code, fresh], cwd=ROOT, capture_output=True, text=True,
                         env=dict(os.environ, DATABASE_URL=f"sqlite:///{fresh}"))
    assert out.returncode == 0, out.stderr[-500:]
    tables = eval(out.stdout.strip().splitlines()[-1])
    assert "drones" in tables and not ({"flights", "flight_features", "packs", "config_epochs", "feature_baselines"} & set(tables))
    print("PASS: app startup's create_all builds the legacy tables but none of the 5 health tables (Alembic-only)")


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"FAIL: {t.__name__} -- {exc!r}")
    print(f"\n{passed}/{len(tests)} D1 health step 10 API tests passed.")
    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
