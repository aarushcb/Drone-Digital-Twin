"""
run_full_regression.py

Runnable, self-contained end-to-end regression check against the LIVE
Render backend -- exercises every backend-provable feature through the
real HTTP API (not local function calls, not TestClient), the same way
the Flutter app or a manual curl session actually would. Registers a
fresh throwaway user and creates ONE disposable test drone at the start,
runs every check against it, and deletes the drone (cascading away all
its telemetry/scene objects) in a `finally` block -- repeated runs never
leave the real database littered with test data, even if a check fails
partway through.

Run with: python3 run_full_regression.py
Needs only httpx, which is already a project dependency (requirements.txt).

Each check is independent and self-contained wherever the feature allows
it (posts its own dedicated telemetry rather than relying on another
check's data still being in whatever window a later query happens to
read) -- the two exceptions, noted inline, are Control Loop (needs the
MAVLink import's real desired-attitude data to exist first) and Fault
Detection (run deliberately last, back-to-back with no interleaved
network calls, for reasons explained at that check -- see FAULT
DETECTION'S TIMING SENSITIVITY below).

One check failing does not stop the others from running -- every check
is caught individually, so a single regression is reported clearly
without hiding the pass/fail status of everything else. Exit code is 0
only if every check passed.
"""
import math
import os
import sys
import time

import httpx

BASE_URL = os.environ.get("REGRESSION_BASE_URL", "https://drone-digital-twin-api.onrender.com")
TLOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_flight_synthetic.tlog")
REQUEST_TIMEOUT = 30.0


class CheckFailure(AssertionError):
    """Raised inside a check function to report a specific, readable failure."""


RETRY_ATTEMPTS = 3
RETRY_DELAY_S = 3.0


def _with_retry(fn):
    """
    Retries a request up to RETRY_ATTEMPTS times on a 5xx response or a
    connection-level error, with a short fixed delay between attempts --
    found to be necessary empirically: Render's free tier can respond
    200 to a lightweight wake-up ping (GET /docs) before its database
    connection pool is actually ready, so the very first real
    (DB-touching) request afterward can transiently 500 even though the
    backend is completely healthy a few seconds later (reproduced this
    exact sequence directly: one register call 500'd right after a
    successful wake-up ping, then the identical call succeeded twice in
    a row immediately after). Does NOT retry on 4xx -- a 401/422/etc. is
    a real result to report, not a transient infra hiccup to paper over.
    """
    last_exc = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            r = fn()
            if r.status_code < 500:
                return r
            last_exc = None
            print(f"    (attempt {attempt}/{RETRY_ATTEMPTS} got HTTP {r.status_code}, retrying...)")
        except httpx.HTTPError as e:
            last_exc = e
            print(f"    (attempt {attempt}/{RETRY_ATTEMPTS} network error: {e}, retrying...)")
        if attempt < RETRY_ATTEMPTS:
            time.sleep(RETRY_DELAY_S)
    if last_exc is not None:
        raise last_exc
    return r  # last (still-failing) response, so the caller's status check reports it normally


class Ctx:
    """Thin wrapper around an httpx.Client that carries the auth token and
    test drone id, so check functions don't have to pass headers/ids
    around by hand. Every request retries transient 5xx/network errors --
    see _with_retry."""

    def __init__(self, base_url: str):
        self.client = httpx.Client(base_url=base_url, timeout=REQUEST_TIMEOUT)
        self.token: str | None = None
        self.drone_id: int | None = None

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    def get(self, path, **kw):
        return _with_retry(lambda: self.client.get(path, headers=self.headers, **kw))

    def post(self, path, **kw):
        return _with_retry(lambda: self.client.post(path, headers=self.headers, **kw))

    def patch(self, path, **kw):
        return _with_retry(lambda: self.client.patch(path, headers=self.headers, **kw))

    def delete(self, path, **kw):
        return _with_retry(lambda: self.client.delete(path, headers=self.headers, **kw))


def expect_status(response: httpx.Response, expected: int, what: str):
    if response.status_code != expected:
        raise CheckFailure(
            f"{what}: expected HTTP {expected}, got {response.status_code} -- body: {response.text[:500]}"
        )


# ============================================================================
# CHECKS -- each takes `ctx` and either returns normally (pass, may print
# extra detail lines) or raises CheckFailure (fail, message is printed by
# the runner). Ordered so each check's own data dependencies are already
# satisfied by the time it runs -- see the module docstring.
# ============================================================================


def check_auth_and_drone_setup(ctx: Ctx):
    """Registers a throwaway user, logs in, and creates the one
    disposable test drone every other check runs against -- with a
    complete motor/propeller/battery spec so every spec-dependent
    feature (motor physics, parameter sweep, efficiency landscape) has
    what it needs from the very start."""
    email = f"regression_{int(time.time())}@test.com"
    password = "regressiontest123"

    r = _with_retry(lambda: ctx.client.post("/auth/register", json={"email": email, "password": password}))
    expect_status(r, 201, "POST /auth/register")

    r = _with_retry(lambda: ctx.client.post(
        "/auth/login",
        data={"username": email, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    ))
    expect_status(r, 200, "POST /auth/login")
    ctx.token = r.json()["access_token"]
    if not ctx.token:
        raise CheckFailure("login returned an empty access_token")

    r = ctx.post(
        "/drones",
        json={
            "name": "RegressionTestDrone",
            "frame_type": "quad",
            "mass_kg": 1.5,
            "motor_count": 4,
            "propeller_diameter_in": 10.0,
            "motor_kv": 920,
            "battery_cells": 4,
            "battery_capacity_mah": 5000.0,
            "max_speed_mps": 6.0,
        },
    )
    expect_status(r, 201, "POST /drones")
    drone = r.json()
    ctx.drone_id = drone["id"]
    if drone["name"] != "RegressionTestDrone" or drone["mass_kg"] != 1.5:
        raise CheckFailure(f"created drone doesn't reflect the spec it was created with: {drone}")

    print(f"  registered {email}, created drone id={ctx.drone_id}")


def check_drone_crud(ctx: Ctx):
    r = ctx.get("/drones")
    expect_status(r, 200, "GET /drones")
    ids = [d["id"] for d in r.json()]
    if ctx.drone_id not in ids:
        raise CheckFailure(f"test drone {ctx.drone_id} missing from GET /drones list ({ids})")

    r = ctx.get(f"/drones/{ctx.drone_id}")
    expect_status(r, 200, f"GET /drones/{ctx.drone_id}")
    if r.json()["name"] != "RegressionTestDrone":
        raise CheckFailure(f"GET single drone returned wrong name: {r.json()}")

    r = ctx.patch(f"/drones/{ctx.drone_id}", json={"max_speed_mps": 7.5})
    expect_status(r, 200, f"PATCH /drones/{ctx.drone_id}")
    patched = r.json()
    if patched["max_speed_mps"] != 7.5:
        raise CheckFailure(f"PATCH didn't update max_speed_mps: {patched}")
    if patched["mass_kg"] != 1.5:
        raise CheckFailure(f"PATCH changed a field it shouldn't have (mass_kg): {patched}")

    print(f"  list/get/patch all consistent (max_speed_mps now {patched['max_speed_mps']})")


def check_telemetry_posting(ctx: Ctx):
    posted = []
    for i in range(5):
        r = ctx.post(
            f"/drones/{ctx.drone_id}/telemetry",
            json={"battery": 95.0 - i, "altitude": 10.0, "flight_state": "flying"},
        )
        expect_status(r, 201, "POST telemetry")
        posted.append(r.json()["id"])

    r = ctx.get(f"/drones/{ctx.drone_id}/telemetry?limit=20")
    expect_status(r, 200, "GET telemetry")
    rows = r.json()
    ids_seen = {row["id"] for row in rows}
    missing = [pid for pid in posted if pid not in ids_seen]
    if missing:
        raise CheckFailure(f"posted telemetry ids missing from GET list: {missing}")

    print(f"  posted {len(posted)} readings, all present in GET (total in window: {len(rows)})")


def check_mavlink_import(ctx: Ctx):
    if not os.path.exists(TLOG_PATH):
        raise CheckFailure(
            f"synthetic test log not found at {TLOG_PATH} -- run generate_test_flight_log.py first"
        )

    with open(TLOG_PATH, "rb") as f:
        tlog_bytes = f.read()

    # Rebuilds the multipart payload fresh on each attempt (a retry can't
    # reuse an already-consumed file/bytes stream from a prior attempt).
    r = _with_retry(lambda: ctx.client.post(
        f"/drones/{ctx.drone_id}/import-mavlink",
        headers=ctx.headers,
        files={"file": ("test_flight_synthetic.tlog", tlog_bytes, "application/octet-stream")},
    ))
    expect_status(r, 200, "POST import-mavlink")
    summary = r.json()
    if summary.get("imported_points", 0) < 100:
        raise CheckFailure(f"expected ~131 imported points, got {summary}")

    print(f"  imported {summary['imported_points']} points from {summary['filename']} "
          f"({summary['start_time']} .. {summary['end_time']})")


def check_digital_twin(ctx: Ctx):
    r = ctx.get(f"/drones/{ctx.drone_id}/digital-twin")
    expect_status(r, 200, "GET digital-twin")
    d = r.json()

    if d["reading_count"] <= 0:
        raise CheckFailure(f"digital-twin reports zero readings despite prior telemetry: {d}")
    if d["specs"]["battery_cells"] != 4 or d["specs"]["battery_capacity_mah"] != 5000.0:
        raise CheckFailure(f"digital-twin specs don't match the drone's stored spec: {d['specs']}")
    if d["motor_wear_percent"] < 0 or d["motor_wear_percent"] > 100:
        raise CheckFailure(f"motor_wear_percent out of [0,100] range: {d['motor_wear_percent']}")

    print(f"  reading_count={d['reading_count']}, flight_hours={d['total_flight_hours']}, "
          f"motor_wear={d['motor_wear_percent']}%, specs.battery_cells={d['specs']['battery_cells']}")


def check_ekf_state_estimate(ctx: Ctx):
    # Dedicated, freshly-timestamped flight segment -- NOT reusing the
    # earlier MAVLink import's data. Found empirically while building
    # this script: the synthetic .tlog's telemetry carries FIXED
    # timestamps from whenever that file was generated, which drifts
    # further into the past every day this script is run afterward; once
    # that gap reaches real multi-day scale, the EKF's process noise
    # (which scales with dt^4 -- see kalman_filter.py's predict(), no
    # upper cap on dt) explodes numerically -- confirmed directly: a
    # 7-day-old gap produced a dt^4 factor of ~1.4e23 and a resulting
    # fused state with a 162-BILLION-meter position uncertainty. That's
    # a real EKF robustness gap worth its own follow-up (a drone idle
    # for a week between real flights would hit this too), but it isn't
    # this check's job to exercise -- this check verifies the EKF
    # FEATURE works, with its own clean, realistic-cadence data, so its
    # pass/fail signal stays meaningful and stable run over run instead
    # of silently degrading as the fixture file's timestamp ages.
    for i in range(10):
        r = ctx.post(
            f"/drones/{ctx.drone_id}/telemetry",
            json={
                "latitude": 37.7749 + i * 0.00001,
                "longitude": -122.4194 + i * 0.00001,
                "altitude": 20.0 + i * 0.2,
                "speed": 3.0,
                "roll": 0.0, "pitch": 2.0, "yaw": 45.0,
                "flight_state": "flying",
            },
        )
        expect_status(r, 201, "POST telemetry (EKF fixture)")

    r = ctx.get(f"/drones/{ctx.drone_id}/telemetry/fused?limit=10")
    expect_status(r, 200, "GET telemetry/fused")
    d = r.json()
    if d["num_points"] <= 0 or not d["points"]:
        raise CheckFailure(f"EKF fused state returned no points: {d}")

    last = d["points"][-1]
    required_keys = {"x_east", "y_north", "z_alt", "vx", "vy", "vz", "roll", "pitch", "yaw",
                      "speed_estimate", "position_uncertainty_m"}
    missing = required_keys - set(last.keys())
    if missing:
        raise CheckFailure(f"EKF point missing expected keys {missing}: {last}")
    # Physically-sane bounds for THIS fixture (a ~20-22m altitude, ~3m/s
    # flight near the origin) -- a real divergence (like the dt^4
    # blowup above) would produce numbers many orders of magnitude
    # outside these, so this is a meaningful regression signal, not an
    # arbitrarily loose placeholder.
    if not (-1000 < last["z_alt"] < 1000):
        raise CheckFailure(f"z_alt outside a physically sane range for this fixture: {last}")
    if not (0 <= last["position_uncertainty_m"] < 1000):
        raise CheckFailure(f"position_uncertainty_m outside a physically sane range: {last}")
    if not (0 <= last["speed_estimate"] < 100):
        raise CheckFailure(f"speed_estimate outside a physically sane range: {last}")

    print(f"  {d['num_points']} fused points, last: z_alt={last['z_alt']}, "
          f"speed_estimate={last['speed_estimate']}, uncertainty={last['position_uncertainty_m']}m")


def check_control_loop(ctx: Ctx):
    # Depends on check_mavlink_import having already run -- that's what
    # populates real desired_roll/pitch/yaw + motor_pwm data (from the
    # synthetic log's ATTITUDE_TARGET/SERVO_OUTPUT_RAW messages), which
    # is what makes this endpoint return data_source="real" instead of
    # falling back to the simulated demo.
    r = ctx.get(f"/drones/{ctx.drone_id}/telemetry/control-loop")
    expect_status(r, 200, "GET telemetry/control-loop")
    d = r.json()

    if d.get("data_source") != "real":
        raise CheckFailure(
            f"expected data_source='real' (from the MAVLink import's desired-attitude data), "
            f"got '{d.get('data_source')}' -- full response: {d}"
        )
    for axis in ("roll", "pitch", "yaw"):
        if axis not in d["axes"]:
            raise CheckFailure(f"missing '{axis}' axis in control-loop response: {d['axes'].keys()}")
    roll_axis = d["axes"]["roll"]
    if roll_axis["rms_error_deg"] is None or roll_axis["rms_error_deg"] < 0:
        raise CheckFailure(f"invalid roll rms_error_deg: {roll_axis}")

    print(f"  data_source=real, sample_count={d.get('sample_count')}, "
          f"roll rms_error={roll_axis['rms_error_deg']}deg, motor_pwm present={d.get('motor_pwm') is not None}")


def check_battery_prediction(ctx: Ctx):
    # Dedicated, monotonically-declining battery sequence posted
    # immediately before checking. GET /status reads the last 51
    # readings (app/api/routes.py), so this posts 55 -- comfortably
    # over that window -- guaranteeing the regression is computed ONLY
    # over this check's own clean data, regardless of whatever trend
    # earlier checks (e.g. the MAVLink import's own, much shallower,
    # battery decline) left sitting in that window. An earlier version
    # of this check posted only 8 points and got its expected negative
    # slope diluted down to essentially flat (-0.001%/min, read as "not
    # draining") by the ~43 older, flatter readings still in the window
    # -- a real lesson about this endpoint's fixed window size, not a
    # backend bug.
    for i in range(55):
        r = ctx.post(f"/drones/{ctx.drone_id}/telemetry", json={"battery": 95.0 - i, "flight_state": "flying"})
        expect_status(r, 201, "POST telemetry (battery sequence)")

    r = ctx.get(f"/drones/{ctx.drone_id}/status")
    expect_status(r, 200, "GET status (battery prediction)")
    d = r.json()
    est = d.get("battery_estimate")
    if not est:
        raise CheckFailure(f"expected a battery_estimate for a clearly-draining battery, got: {d}")
    if est["drain_rate_percent_per_min"] >= 0:
        raise CheckFailure(f"expected a negative drain rate for declining battery, got: {est}")
    if est["minutes_remaining"] is None or est["minutes_remaining"] < 0:
        raise CheckFailure(f"invalid minutes_remaining: {est}")

    # drain_rate will look extreme (often several hundred %/min) --
    # correct, not a bug: 55 requests land within a couple of real
    # seconds with no artificial delay between them, so a 54%
    # battery drop happens over a genuinely tiny elapsed time.
    print(f"  drain_rate={est['drain_rate_percent_per_min']}%/min (steep because these 55 posts landed "
          f"within ~seconds, not a bug), minutes_remaining={est['minutes_remaining']}, confidence={est['confidence']}")


def check_anomaly_detection(ctx: Ctx):
    # Dedicated batch: 8 near-constant temperature readings (the
    # z-score baseline this detector needs, MIN_READINGS_FOR_ANOMALY_-
    # DETECTION=8 in predictive_analytics.py), then one wildly hot
    # outlier as the final/"current" reading. No other check in this
    # script ever sets `temperature`, so this batch is the only source
    # of non-null temperature readings -- keeps the z-score baseline
    # clean regardless of what other telemetry exists by this point.
    for i in range(8):
        temp = 24.5 if i % 2 == 0 else 25.5
        r = ctx.post(f"/drones/{ctx.drone_id}/telemetry", json={"temperature": temp, "flight_state": "flying"})
        expect_status(r, 201, "POST telemetry (temperature baseline)")

    r = ctx.post(f"/drones/{ctx.drone_id}/telemetry", json={"temperature": 95.0, "flight_state": "flying"})
    expect_status(r, 201, "POST telemetry (temperature outlier)")

    r = ctx.get(f"/drones/{ctx.drone_id}/status")
    expect_status(r, 200, "GET status (anomaly detection)")
    anomalies = r.json().get("anomalies", [])
    if not anomalies:
        raise CheckFailure("expected at least one anomaly for a 95C reading against a ~25C baseline, got none")
    if not any("Temperature" in a for a in anomalies):
        raise CheckFailure(f"anomaly list doesn't reference Temperature: {anomalies}")

    print(f"  anomalies detected: {anomalies}")


def check_environment_simulator_and_motor_physics(ctx: Ctx):
    r = ctx.post(f"/drones/{ctx.drone_id}/simulate-environment", json={"altitude_m": 2000, "temperature_c": 5})
    expect_status(r, 200, "POST simulate-environment")
    d = r.json()

    if "power_change_percent" not in d and "relative_power_change_percent" not in d:
        # Field name sanity check without hard-pinning the exact key --
        # what matters is SOME relative-change figure is present.
        if not any("percent" in k for k in d.keys()):
            raise CheckFailure(f"no relative power-change field found in response: {list(d.keys())}")

    motor = d.get("motor_performance")
    if motor is None:
        raise CheckFailure(f"expected motor_performance to be populated (drone has complete specs): {d}")
    if not motor["feasible"]:
        raise CheckFailure(f"expected this realistic spec to be hover-feasible: {motor}")
    if motor["required_rpm"] <= 0 or motor["total_hover_power_watts"] <= 0:
        raise CheckFailure(f"nonsensical motor physics numbers: {motor}")

    print(f"  environment: altitude=2000m temp=5C -> motor feasible={motor['feasible']}, "
          f"required_rpm={motor['required_rpm']}, hover_power={motor['total_hover_power_watts']}W")


def check_parameter_sweep(ctx: Ctx):
    r = ctx.post(
        f"/drones/{ctx.drone_id}/parameter-sweep",
        json={"propeller_diameter_delta_in": 1.0, "battery_cells_delta": 0},
    )
    expect_status(r, 200, "POST parameter-sweep")
    d = r.json()
    current, swept = d["current"], d["swept"]
    current_spec, swept_spec = d["current_spec"], d["swept_spec"]
    if swept_spec["propeller_diameter_in"] <= current_spec["propeller_diameter_in"]:
        raise CheckFailure(f"expected swept propeller diameter > current: {d}")

    print(f"  propeller {current_spec['propeller_diameter_in']}in -> {swept_spec['propeller_diameter_in']}in, "
          f"hover efficiency {current.get('hover_efficiency_w_per_kg')} -> {swept.get('hover_efficiency_w_per_kg')} W/kg")


def check_efficiency_landscape(ctx: Ctx):
    r = ctx.get(f"/drones/{ctx.drone_id}/efficiency-landscape?grid_size=6")
    expect_status(r, 200, "GET efficiency-landscape")
    d = r.json()
    grid = d.get("efficiency_grid") or d.get("grid")
    if not grid:
        raise CheckFailure(f"no grid field found in response: {list(d.keys())}")
    if len(grid) != 6 or any(len(row) != 6 for row in grid):
        raise CheckFailure(f"expected a 6x6 grid, got shape {len(grid)}x{len(grid[0]) if grid else 0}")

    print(f"  {len(grid)}x{len(grid[0])} efficiency grid returned")


def check_astar_path_planning(ctx: Ctx):
    r = ctx.post(f"/drones/{ctx.drone_id}/objects", json={"object_type": "obstacle", "x": 5, "y": 0, "z": 5})
    expect_status(r, 201, "POST scene object (obstacle)")
    r = ctx.post(f"/drones/{ctx.drone_id}/objects", json={"object_type": "landing", "x": 10, "y": 0, "z": 10})
    expect_status(r, 201, "POST scene object (landing)")

    r = ctx.post(f"/drones/{ctx.drone_id}/plan-path-3d", json={})
    expect_status(r, 200, "POST plan-path-3d")
    d = r.json()
    path = d["path"]
    if len(path) < 2:
        raise CheckFailure(f"expected a multi-point path, got: {d}")

    obstacle_radius = 0.6
    min_clearance = min(math.dist((p["x"], p["y"], p["z"]), (5, 0, 5)) for p in path)
    if min_clearance < obstacle_radius - 1e-6:
        raise CheckFailure(f"path passes within {min_clearance}m of the obstacle (radius {obstacle_radius}): {path}")

    print(f"  {len(path)} waypoints, distance={d['distance_meters']}m, "
          f"min obstacle clearance={round(min_clearance, 3)}m")


def check_trajectory_smoothing(ctx: Ctx):
    # Depends on the obstacle/landing point placed by check_astar_path_planning.
    r = ctx.post(f"/drones/{ctx.drone_id}/plan-path-smooth", json={})
    expect_status(r, 200, "POST plan-path-smooth")
    d = r.json()

    if d["keypoint_count"] > d["raw_waypoint_count"]:
        raise CheckFailure(f"keypoint_count should be <= raw_waypoint_count: {d}")
    traj = d["trajectory"]
    if len(traj) < 2:
        raise CheckFailure(f"expected multiple trajectory samples: {d}")
    if traj[0]["speed_mps"] > 0.01 or traj[-1]["speed_mps"] > 0.01:
        raise CheckFailure(f"expected trajectory to start/end at rest: first={traj[0]}, last={traj[-1]}")
    tolerance = 1.05
    if d["max_realized_speed_mps"] > d["max_speed_mps_used"] * tolerance:
        raise CheckFailure(f"realized speed exceeds requested max: {d}")

    print(f"  {d['raw_waypoint_count']} raw waypoints -> {d['keypoint_count']} keypoints, "
          f"{len(traj)} samples, duration={d['total_duration_seconds']}s, "
          f"max_realized_speed={d['max_realized_speed_mps']}m/s (limit {d['max_speed_mps_used']})")


def check_frame_comparison(ctx: Ctx):
    r = ctx.get(
        "/frames/compare",
        params={
            "frame_types": "quad,hex,octo,fixed_wing",
            "mass_kg": 1.5,
            "propeller_diameter_in": 10,
            "motor_kv": 920,
            "battery_cells": 4,
            "battery_capacity_mah": 5000,
            "wing_area_m2": 0.4,
            "lift_to_drag_ratio": 12,
        },
    )
    expect_status(r, 200, "GET frames/compare")
    d = r.json()
    frames = d["frames"]
    for frame_type in ("quad", "hex", "octo", "fixed_wing"):
        if frame_type not in frames:
            raise CheckFailure(f"missing frame type '{frame_type}' in response: {list(frames.keys())}")

    quad, fixed_wing = frames["quad"], frames["fixed_wing"]
    if quad["hover_efficiency_w_per_kg"] is None:
        raise CheckFailure(f"quad should have a real hover efficiency: {quad}")
    if fixed_wing["hover_efficiency_w_per_kg"] is not None:
        raise CheckFailure(f"fixed_wing hover_efficiency_w_per_kg should be None (hover physics doesn't apply): {fixed_wing}")
    if fixed_wing.get("cruise_analysis") is None:
        raise CheckFailure(f"fixed_wing should have real cruise_analysis: {fixed_wing}")
    if quad.get("cruise_analysis") is not None:
        raise CheckFailure(f"quad cruise_analysis should be None (rotorcraft-only field misuse): {quad}")

    print(f"  quad hover_eff={quad['hover_efficiency_w_per_kg']}W/kg, "
          f"fixed_wing cruise_endurance={fixed_wing['cruise_analysis']['estimated_endurance_minutes']}min "
          f"at {fixed_wing['cruise_analysis']['cruise_velocity_mps']}m/s")


def check_fault_detection(ctx: Ctx):
    """
    FAULT DETECTION'S TIMING SENSITIVITY (found and root-caused during
    manual live verification of this exact feature): the Kalman filter's
    process-noise growth scales with the elapsed time (dt) between
    readings, so a long real-world gap between posts (e.g. from
    interleaving diagnostic GET calls between each POST) widens the
    filter's uncertainty enough that even a 50m+ jump can produce a LOW
    normalized-innovation-squared (NIS) value -- a long gap makes a big
    real altitude change statistically plausible, which is correct
    filter behavior, not a bug, but it means this check must post its
    entire baseline+spike+recovery sequence BACK TO BACK with no
    interleaved network calls, exactly like this, or the spike may not
    register. Run deliberately LAST for the same reason: it's the most
    timing-sensitive check, so it isn't slowed by any other check's
    round-trips first.
    """
    for _ in range(3):
        r = ctx.post(f"/drones/{ctx.drone_id}/telemetry", json={"altitude": 30.0, "flight_state": "flying"})
        expect_status(r, 201, "POST telemetry (fault baseline)")
    r = ctx.post(f"/drones/{ctx.drone_id}/telemetry", json={"altitude": 82.0, "flight_state": "flying"})
    expect_status(r, 201, "POST telemetry (fault spike)")
    for i in range(4):
        alt = 30.1 if i % 2 == 0 else 29.9
        r = ctx.post(f"/drones/{ctx.drone_id}/telemetry", json={"altitude": alt, "flight_state": "flying"})
        expect_status(r, 201, "POST telemetry (fault recovery)")

    r = ctx.get(f"/drones/{ctx.drone_id}/status")
    expect_status(r, 200, "GET status (fault detection)")
    faults = r.json().get("sensor_faults", [])
    if not faults:
        raise CheckFailure(
            "expected a sensor fault after a 52m instantaneous altitude jump, got none -- "
            "either the injection sequence wasn't tight enough in time (see this check's docstring) "
            "or the detector genuinely regressed"
        )
    fault = faults[0]
    if fault["num_consecutive_points"] < 3:
        raise CheckFailure(f"fault run shorter than the required minimum of 3: {fault}")
    if fault["peak_nis"] <= 3.841:
        raise CheckFailure(f"peak_nis doesn't exceed the 95% chi-square threshold: {fault}")

    print(f"  fault detected: severity={fault['severity']}, peak_nis={fault['peak_nis']}, "
          f"run_length={fault['num_consecutive_points']} points")


CHECKS = [
    ("Auth + drone setup", check_auth_and_drone_setup),
    ("Drone CRUD", check_drone_crud),
    ("Telemetry posting", check_telemetry_posting),
    ("MAVLink import", check_mavlink_import),
    ("Digital twin panel", check_digital_twin),
    ("EKF state estimate", check_ekf_state_estimate),
    ("Control loop data", check_control_loop),
    ("Battery prediction", check_battery_prediction),
    ("Anomaly detection", check_anomaly_detection),
    ("Environment simulator + motor physics", check_environment_simulator_and_motor_physics),
    ("Parameter sweep", check_parameter_sweep),
    ("Efficiency landscape", check_efficiency_landscape),
    ("A* path planning", check_astar_path_planning),
    ("Trajectory smoothing", check_trajectory_smoothing),
    ("Frame comparison (incl. fixed-wing)", check_frame_comparison),
    ("Fault detection", check_fault_detection),
]


def wake_server(base_url: str):
    print(f"Waking server at {base_url} ...")
    try:
        r = httpx.get(f"{base_url}/docs", timeout=90.0)
        print(f"  server responded {r.status_code}")
    except httpx.HTTPError as e:
        print(f"  WARNING: could not reach server yet ({e}); continuing anyway")


def main() -> int:
    wake_server(BASE_URL)
    ctx = Ctx(BASE_URL)

    results = []  # (name, passed, detail_or_error)
    try:
        for i, (name, fn) in enumerate(CHECKS):
            print(f"\n[{name}]")
            try:
                fn(ctx)
                results.append((name, True, None))
                print(f"  PASS: {name}")
            except CheckFailure as e:
                results.append((name, False, str(e)))
                print(f"  FAIL: {name} -- {e}")
            except httpx.HTTPError as e:
                results.append((name, False, f"network error: {e}"))
                print(f"  FAIL: {name} -- network error: {e}")
            except Exception as e:  # noqa: BLE001 -- a check crashing is still just a FAIL, not a script crash
                results.append((name, False, f"unexpected error: {e!r}"))
                print(f"  FAIL: {name} -- unexpected error: {e!r}")

            # The very first check (auth + drone setup) is a hard
            # prerequisite for literally everything else -- if it failed,
            # every remaining check would just be a guaranteed, noisy
            # 401 (no token) or crash (no drone_id), telling us nothing
            # new. Stop here instead; cleanup still runs via `finally`.
            if i == 0 and not results[-1][1]:
                print("\nAuth + drone setup failed -- skipping all remaining checks (nothing else can run without it).")
                break
    finally:
        if ctx.drone_id is not None:
            print(f"\nCleaning up: deleting test drone {ctx.drone_id} ...")
            try:
                r = ctx.delete(f"/drones/{ctx.drone_id}")
                print(f"  delete returned HTTP {r.status_code}")
            except httpx.HTTPError as e:
                print(f"  WARNING: could not delete test drone {ctx.drone_id} -- clean it up manually: {e}")
        ctx.client.close()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    print(f"\n{passed}/{len(results)} checks passed.")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
