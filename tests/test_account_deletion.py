"""
Account deletion (Google Play requirement): DELETE /auth/account and the
public web page at /account/delete.

Run:  python3 tests/test_account_deletion.py

TestClient against a temporary SQLite database built by `alembic upgrade
head` (never Neon). Uses the real fixture log when present, so flights carry
features and baselines; falls back to an unparseable log otherwise.
"""

import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TMP = tempfile.mkdtemp(prefix="account_del_")
os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(TMP, 'acct.db')}"
sys.path.insert(0, ROOT)
_mig = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, capture_output=True, text=True,
                      env=dict(os.environ))
assert _mig.returncode == 0, _mig.stderr[-800:]

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402
from app.database.database import SessionLocal  # noqa: E402
from app.models.drone import Drone  # noqa: E402
from app.models.health import ConfigEpoch, FeatureBaseline, Flight, FlightFeature, Pack  # noqa: E402
from app.models.scene_object import SceneObject  # noqa: E402
from app.models.telemetry import Telemetry  # noqa: E402
from app.models.user import User  # noqa: E402

REAL = os.path.join(ROOT, "tests", "fixtures", "reality_gap", "real", "px4_777ed799.ulg")
client = TestClient(app)
PW = "pw123456"


def make_user(tag, with_flight=True):
    email = f"{tag}_{int(time.time() * 1000)}@test.com"
    assert client.post("/auth/register", json={"email": email, "password": PW}).status_code == 201
    h = {"Authorization": f"Bearer {client.post('/auth/login', data={'username': email, 'password': PW}).json()['access_token']}"}
    drones = [client.post("/drones", json={"name": f"{tag} {i}"}, headers=h).json()["id"] for i in range(2)]
    d = drones[0]
    for i in range(3):
        assert client.post(f"/drones/{d}/telemetry", json={"battery": 90 - i, "altitude": 10}, headers=h).status_code == 201
    assert client.post(f"/drones/{d}/objects", json={"object_type": "obstacle", "x": 1, "z": 2}, headers=h).status_code == 201
    assert client.post(f"/drones/{d}/packs", json={"label": "A"}, headers=h).status_code == 201
    assert client.post(f"/drones/{d}/config-changes", json={"change_type": "props"}, headers=h).status_code == 201
    if with_flight:
        path = REAL if os.path.exists(REAL) else os.path.join(TMP, "junk.ulg")
        if not os.path.exists(path):
            open(path, "wb").write(b"not a log " * 50)
        with open(path, "rb") as fh:
            fid = client.post(f"/drones/{d}/flights", headers=h, files={"file": (os.path.basename(path), fh)}).json()["flight_id"]
        for _ in range(600):
            if client.get(f"/drones/{d}/flights/{fid}", headers=h).json()["status"] in ("done", "failed"):
                break
            time.sleep(0.1)
    return email, h, drones


def owned_rows(email):
    db = SessionLocal()
    try:
        u = db.query(User).filter_by(email=email).first()
        if u is None:
            return {"user": 0}
        ds = [d.id for d in db.query(Drone).filter_by(owner_id=u.id)]
        fs = [f.id for f in db.query(Flight).filter(Flight.drone_id.in_(ds))]
        return {
            "user": 1, "drones": len(ds),
            "telemetry": db.query(Telemetry).filter(Telemetry.drone_id.in_(ds)).count(),
            "scene_objects": db.query(SceneObject).filter(SceneObject.drone_id.in_(ds)).count(),
            "flights": len(fs),
            "flight_features": db.query(FlightFeature).filter(FlightFeature.flight_id.in_(fs)).count(),
            "feature_baselines": db.query(FeatureBaseline).filter(FeatureBaseline.drone_id.in_(ds)).count(),
            "config_epochs": db.query(ConfigEpoch).filter(ConfigEpoch.drone_id.in_(ds)).count(),
            "packs": db.query(Pack).filter(Pack.drone_id.in_(ds)).count(),
        }
    finally:
        db.close()


def test_wrong_password_deletes_nothing():
    email, h, _ = make_user("wrongpw", with_flight=False)
    before = owned_rows(email)
    r = client.request("DELETE", "/auth/account", headers=h, json={"password": "nope"})
    assert r.status_code == 403 and owned_rows(email) == before
    assert client.request("DELETE", "/auth/account", json={"password": PW}).status_code == 401   # needs the token too
    print("PASS: wrong password -> 403 and nothing deleted; no token -> 401")


def test_deletes_everything_and_reports_it():
    email, h, drones = make_user("leaver")
    other_email, oh, _ = make_user("stayer", with_flight=False)
    before, other_before = owned_rows(email), owned_rows(other_email)
    assert before["telemetry"] == 3 and before["flights"] == 1 and before["packs"] == 1
    r = client.request("DELETE", "/auth/account", headers=h, json={"password": PW})
    assert r.status_code == 200, r.text
    deleted = r.json()["deleted"]
    for k, v in before.items():
        assert deleted[k] == v, (k, deleted[k], v)
    assert deleted["log_files_pending"] == 0
    assert owned_rows(email) == {"user": 0}
    assert owned_rows(other_email) == other_before                            # nobody else touched
    assert client.post("/auth/login", data={"username": email, "password": PW}).status_code == 401
    assert client.get(f"/drones/{drones[0]}/flights", headers=h).status_code == 401   # the old token is dead
    print(f"PASS: deleted and reported {deleted}; the other user's data is untouched; the account can't sign in")


def test_public_web_page_works_without_the_app():
    email, _, _ = make_user("webleaver", with_flight=False)
    page = client.get("/account/delete")
    assert page.status_code == 200 and "<form" in page.text and "cannot be undone" in page.text
    assert "<script" not in page.text
    bad = client.post("/account/delete", data={"email": email, "password": "wrong"})
    assert bad.status_code == 401 and "Nothing was deleted" in bad.text and owned_rows(email)["user"] == 1
    ok = client.post("/account/delete", data={"email": email, "password": PW})
    assert ok.status_code == 200 and "have been deleted" in ok.text and owned_rows(email) == {"user": 0}
    print("PASS: GET /account/delete shows a plain form; a wrong password deletes nothing; the right one deletes all")


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
    print(f"\n{passed}/{len(tests)} account deletion tests passed.")
    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
