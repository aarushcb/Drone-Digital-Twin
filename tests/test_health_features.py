"""
D1 Health, build steps 1-3: single-log parsing, windowing, feature extraction.

Run:  python3 tests/test_health_features.py

Standalone script, same convention as test_reality_gap_metrics.py (no pytest
in this repo); plain test_* functions with bare asserts, so pytest would
collect them unchanged. Real-log tests skip when the gitignored fixtures in
tests/fixtures/reality_gap/real/ are absent.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.services.health.features import FEATURE_CATALOGUE, extract_features  # noqa: E402
from app.services.health.windowing import window_flight  # noqa: E402
from app.services.reality_gap.parsers import LogMeta, ParsedLog, Signal, parse_log  # noqa: E402

REAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "reality_gap", "real", "px4_777ed799.ulg")
HAS_REAL = os.path.exists(REAL)
_real_cache = {}

# Hex geometry from the real log's CA_ROTOR* parameters (PX forward, PY right).
HEX_PARAMS = {
    "CA_ROTOR_COUNT": 6.0,
    "CA_ROTOR0_PX": 0.0, "CA_ROTOR0_PY": 0.62, "CA_ROTOR0_KM": -0.05,
    "CA_ROTOR1_PX": 0.0, "CA_ROTOR1_PY": -0.62, "CA_ROTOR1_KM": 0.05,
    "CA_ROTOR2_PX": 0.57, "CA_ROTOR2_PY": -0.31, "CA_ROTOR2_KM": -0.05,
    "CA_ROTOR3_PX": -0.57, "CA_ROTOR3_PY": 0.31, "CA_ROTOR3_KM": 0.05,
    "CA_ROTOR4_PX": 0.57, "CA_ROTOR4_PY": 0.31, "CA_ROTOR4_KM": 0.05,
    "CA_ROTOR5_PX": -0.57, "CA_ROTOR5_PY": -0.31, "CA_ROTOR5_KM": -0.05,
    "BAT1_N_CELLS": 12.0,
}


def real():
    if "p" not in _real_cache:
        p = parse_log(REAL)
        _real_cache["p"] = p
        _real_cache["w"] = window_flight(p)
        _real_cache["f"] = {f.key: f for f in extract_features(p, _real_cache["w"])}
    return _real_cache["p"], _real_cache["w"], _real_cache["f"]


# ---------------------------------------------------------------- synthetic builder

def sig(key, t_s, values):
    t_s = np.asarray(t_s, dtype=np.float64)
    return Signal(key=key, t_us=(t_s * 1e6).astype(np.int64), values=np.asarray(values, dtype=np.float32))


def flight(duration=110.0, airborne=(10.0, 100.0), manoeuvre=(40.0, 45.0), land=True, extra=None, params=None):
    """A synthetic flight with known times: climb for 5 s after takeoff,
    descend for the last 5 s, a roll manoeuvre in between, level otherwise."""
    t50 = np.arange(0.0, duration, 0.02)
    up, down = airborne
    vz = np.zeros_like(t50)
    vz[(t50 >= up) & (t50 < up + 5)] = -1.0       # climbing (NED)
    vz[(t50 >= down - 5) & (t50 < down)] = 1.0    # descending
    roll_sp = np.zeros_like(t50)
    if manoeuvre:
        roll_sp[(t50 >= manoeuvre[0]) & (t50 < manoeuvre[1])] = 0.6
    signals = {
        "vel.z": sig("vel.z", t50, vz),
        "rate_sp.roll": sig("rate_sp.roll", t50, roll_sp),
        "rate_sp.pitch": sig("rate_sp.pitch", t50, np.zeros_like(t50)),
        "rate_sp.yaw": sig("rate_sp.yaw", t50, np.zeros_like(t50)),
    }
    if land:
        t10 = np.arange(0.0, duration, 0.1)
        signals["land.landed"] = sig("land.landed", t10, ((t10 < up) | (t10 >= down)).astype(float))
    for s in (extra or []):
        signals[s.key] = s
    meta = LogMeta(source_format="ulog", filename="synthetic.ulg", sha256="0" * 64, imu_accel_cutoff=30.0)
    return ParsedLog(signals=signals, params=dict(params or HEX_PARAMS), meta=meta,
                     motor_info={"motor_count": 6, "motor_count_source": "CA_ROTOR_COUNT"})


def feats(parsed):
    return {f.key: f for f in extract_features(parsed, window_flight(parsed))}


# ---------------------------------------------------------------- step 1: parsing (real log)

def test_real_log_all_motors_by_motor_number():
    if not HAS_REAL:
        print("SKIP: real fixture absent"); return
    p, _, _ = real()
    assert [k for k in sorted(p.signals) if k.startswith("act.motor")] == [f"act.motor{i}" for i in range(6)]
    assert [k for k in sorted(p.signals) if k.startswith("act.out")] == [f"act.out{i}" for i in range(6)]
    assert p.motor_info["motor_count"] == 6 and p.motor_info["motor_count_source"] == "CA_ROTOR_COUNT"
    assert p.motor_info["mapping_source"] == "output_functions"
    assert p.motor_info["channel_of_motor"][0] == "PWM_MAIN1"
    assert p.motor_info["actuator_outputs_instance"] == 0   # instance 1 is the AUX bank
    assert p.meta.actuator_pwm_range == (1155.0, 1900.0)   # per-channel limits, not 1000-2000
    assert max(float(p.signals[f"act.out{i}"].values.max()) for i in range(6)) <= 1.0
    print("PASS: hex -> 6 motors by motor number via PWM_MAIN_FUNC, instance 0, 1155-1900 us limits")


def test_real_log_selects_primary_instances():
    if not HAS_REAL:
        print("SKIP: real fixture absent"); return
    p, _, _ = real()
    imu, est = p.topic_info["vehicle_imu_status"], p.topic_info["estimator_status"]
    assert imu["instances"] == 3 and imu["selected_instance"] == 0, imu
    assert est["instances"] == 3 and est["selected_instance"] == 0, est
    # The old last-instance read gave a secondary IMU whose vibration median was 3.14.
    assert abs(float(np.median(p.signals["vibe.x"].values)) - 5.42) < 0.2
    print("PASS: primary EKF (selector) and its IMU (accel_device_id) selected, not the last instance")


def test_real_log_new_signals_and_rates():
    if not HAS_REAL:
        print("SKIP: real fixture absent"); return
    p, _, _ = real()
    for key in ["att.qw", "att_sp.qw", "fd.imbalance", "est.vel_tr", "est.pos_tr", "est.hgt_tr", "est.mag_tr",
                "batt.voltage_raw", "batt.current_raw", "batt.discharged_mah", "batt.cell_delta",
                "gps.sacc", "gps.noise", "gps.jamming", "imu.clip0", "imu.clip1", "imu.clip2", "land.landed"]:
        assert key in p.signals, key
    ti = p.topic_info
    assert abs(ti["actuator_outputs"]["rate_hz"] - 10.0) < 0.5, ti["actuator_outputs"]
    assert abs(ti["battery_status"]["rate_hz"] - 5.0) < 0.3
    assert abs(ti["vehicle_imu_status"]["rate_hz"] - 1.0) < 0.1
    assert ti["sensor_accel_fifo"] == {"present": False}
    print("PASS: every new signal present; topic rates recorded (outputs 10 Hz, battery 5 Hz); FIFO recorded absent")


# ---------------------------------------------------------------- step 2: windowing

def test_window_known_flight():
    w = window_flight(flight())
    assert w.qualified and not w.reasons
    assert w.airborne_source == "land_detector"
    assert abs(w.airborne_s - 90.0) < 0.2, w.airborne_s
    # Steady: 15-40 s and 45-95 s (climb, manoeuvre and descent excluded).
    assert len(w.steady_segments) == 2, w.steady_segments
    assert abs(w.steady_s - 75.0) < 0.2, w.steady_s
    assert w.steady_qualified
    print(f"PASS: synthetic flight -> airborne {w.airborne_s} s, 2 steady segments, {w.steady_s} s steady")


def test_window_rejects_short_flight():
    w = window_flight(flight(duration=60.0, airborne=(10.0, 50.0), manoeuvre=None))
    assert not w.qualified and w.reasons[0].startswith("airborne_too_short"), w.reasons
    assert not w.steady_qualified
    print(f"PASS: 40 s airborne -> not qualified ({w.reasons[0]})")


def test_window_falls_back_to_altitude():
    t = np.arange(0.0, 110.0, 0.02)
    alt = np.where((t >= 10) & (t < 100), 20.0, 0.0)
    armed = np.where(t >= 5, 1.0, 0.0)
    w = window_flight(flight(land=False, extra=[sig("alt.rel", t, alt), sig("status.armed", t, armed)]))
    assert w.airborne_source == "altitude_threshold" and w.qualified, (w.airborne_source, w.reasons)
    print("PASS: no land detector -> altitude-threshold airborne window, still qualifies")


def test_window_steady_needs_rate_setpoints():
    p = flight()
    del p.signals["rate_sp.yaw"]
    w = window_flight(p)
    assert w.qualified and not w.steady_qualified and w.steady_reason == "signal_absent:rate_sp.yaw"
    print("PASS: missing rate setpoint -> steady detection reports why instead of guessing")


# ---------------------------------------------------------------- step 3: features (synthetic)

def test_clip_rate_uses_cumulative_delta():
    t = np.arange(0.0, 110.0, 1.0)
    clip0 = np.where(t < 10, 5.0, 5.0 + np.minimum(t - 10, 90) * 0.2)   # +18 over the airborne 90 s
    extra = [sig("imu.clip0", t, clip0), sig("imu.clip1", t, np.zeros_like(t)), sig("imu.clip2", t, np.zeros_like(t))]
    f = feats(flight(extra=extra))["vib.clip_rate"]
    assert abs(f.value - 12.0) < 0.3, f.value   # 18 events / 1.5 min, pre-takeoff 5 excluded
    reset = [sig("imu.clip0", t, np.where(t < 50, 10.0, 2.0)), extra[1], extra[2]]
    assert feats(flight(extra=reset))["vib.clip_rate"].reason == "counter_reset"
    print(f"PASS: clip rate {f.value:.2f}/min from the airborne delta; a counter reset -> null")


def battery(current, t):
    mah = np.cumsum(current) * (t[1] - t[0]) / 3.6
    rng = np.random.default_rng(1)
    v = 50.4 - 0.0003 * mah - 0.060 * current + rng.normal(0, 0.005, t.size)
    return [sig("batt.voltage_raw", t, v), sig("batt.current_raw", t, current), sig("batt.discharged_mah", t, mah)]


def test_pack_resistance_recovered_within_5_percent():
    t = np.arange(0.0, 110.0, 0.2)
    current = 40 + 15 * np.sin(2 * np.pi * t / 20)
    f = feats(flight(extra=battery(current, t)))["pwr.r_cell"]
    assert f.value is not None, f.reason
    assert abs(f.value - 5.0) / 5.0 < 0.05, f.value     # 0.060 ohm / 12 cells = 5.0 mOhm
    print(f"PASS: V = a + b*mAh - R*I recovers {f.value:.3f} mOhm/cell vs true 5.000 (within 5%)")


def test_pack_resistance_refuses_flat_current():
    t = np.arange(0.0, 110.0, 0.2)
    f = feats(flight(extra=battery(np.full(t.size, 40.0) + 0.5 * np.sin(t), t)))["pwr.r_cell"]
    assert f.value is None and f.reason == "insufficient_current_range", (f.value, f.reason)
    print("PASS: current varying < 30% of its mean -> pwr.r_cell null (insufficient_current_range)")


def accel(prefix, fs, amp=0.5, freq=10.0):
    t = np.arange(0.0, 110.0, 1.0 / fs)
    rng = np.random.default_rng(2)
    z = -9.81 + amp * np.sin(2 * np.pi * freq * t)
    return [sig(f"{prefix}.x", t, rng.normal(0, 1e-3, t.size)), sig(f"{prefix}.y", t, rng.normal(0, 1e-3, t.size)),
            sig(f"{prefix}.z", t, z)]


def test_vibration_bands_filtered_source():
    f = feats(flight(extra=accel("accel", 200.0)))
    expected = 10 * math.log10(0.5 ** 2 / 2)   # a 10 Hz sine of amplitude 0.5 carries A^2/2
    assert abs(f["vib.band_struct"].value - expected) < 0.5, (f["vib.band_struct"].value, expected)
    assert f["vib.band_struct"].detail["source"] == "filtered"
    assert f["vib.band_hi"].value is None and f["vib.band_hi"].reason == "requires_fifo_accel"
    print(f"PASS: 10 Hz tone -> band_struct {f['vib.band_struct'].value:.2f} dB vs analytic {expected:.2f}; "
          "band_hi null (requires_fifo_accel)")


def test_vibration_bands_prefer_fifo():
    f = feats(flight(extra=accel("accel", 200.0) + accel("accel_fifo", 800.0, freq=60.0)))
    assert f["vib.band_hi"].detail["source"] == "fifo" and f["vib.band_hi"].value is not None
    expected = 10 * math.log10(0.5 ** 2 / 2)
    assert abs(f["vib.band_hi"].value - expected) < 0.5, f["vib.band_hi"].value
    print(f"PASS: FIFO present -> used; 60 Hz tone lands in band_hi ({f['vib.band_hi'].value:.2f} dB)")


def motors(offsets, base=0.5):
    t = np.arange(0.0, 110.0, 0.1)
    return [sig(f"act.motor{i}", t, np.full(t.size, base + o)) for i, o in enumerate(offsets)]


def test_motor_pattern_classification():
    km = [HEX_PARAMS[f"CA_ROTOR{i}_KM"] for i in range(6)]
    px = [HEX_PARAMS[f"CA_ROTOR{i}_PX"] for i in range(6)]
    single = feats(flight(extra=motors([0, 0, 0, 0.06, 0, 0])))["act.pattern"]
    assert single.label == "single_motor_high" and single.detail["motor_number"] == 4, single
    side = feats(flight(extra=motors([-0.04 * x for x in px])))["act.pattern"]
    assert side.label == "one_side_high" and side.detail["toward"] == "rear", side
    spin = feats(flight(extra=motors([0.02 * math.copysign(1, k) for k in km])))["act.pattern"]
    assert spin.label == "same_spin_set_high", spin
    balanced = feats(flight(extra=motors([0.002, -0.002, 0.001, -0.001, 0, 0])))["act.pattern"]
    assert balanced.label == "balanced"
    no_geo = flight(extra=motors([0, 0, 0, 0.06, 0, 0]), params={"BAT1_N_CELLS": 12.0})
    assert feats(no_geo)["act.pattern"].reason == "geometry_unavailable"
    print("PASS: pattern -> single (motor 4), one side (rear), same spin set, balanced; no geometry -> null")


def test_motor_spread_and_saturation():
    f = feats(flight(extra=motors([0, 0.02, -0.02, 0, 0, 0.04])))
    assert abs(f["act.spread"].value - 0.06 / f["act.mean"].value) < 1e-6
    assert f["act.spread"].detail["highest_motor_number"] == 6
    assert f["act.sat_frac"].value == 0.0
    sat = feats(flight(extra=motors([0.49, 0, 0, 0, 0, 0])))["act.sat_frac"]   # motor 1 pinned at 0.99
    assert sat.value > 0.99, sat.value
    print("PASS: spread = (max-min)/mean exactly; a pinned motor -> saturation fraction ~1")


def test_attitude_error_known_rotation():
    t = np.arange(0.0, 110.0, 0.05)
    half = math.radians(5.0) / 2
    one, zero = np.ones_like(t), np.zeros_like(t)
    extra = [sig("att_sp.qw", t, one), sig("att_sp.qx", t, zero), sig("att_sp.qy", t, zero), sig("att_sp.qz", t, zero),
             sig("att.qw", t, one * math.cos(half)), sig("att.qx", t, one * math.sin(half)),
             sig("att.qy", t, zero), sig("att.qz", t, zero)]
    f = feats(flight(extra=extra))["trk.att_err_p95"]
    assert abs(f.value - 5.0) < 1e-3, f.value
    print(f"PASS: actual rotated 5 deg from setpoint -> attitude error p95 {f.value:.4f} deg")


def test_rate_tracking_only_over_manoeuvres():
    p = flight(manoeuvre=(30.0, 60.0))
    t = p.signals["rate_sp.roll"].t_us / 1e6
    sp = p.signals["rate_sp.roll"].values.astype(np.float64)
    rng = np.random.default_rng(3)
    gyro = sp + rng.normal(0, 0.06, t.size)     # 0.06 rad/s error on a 0.6 rad/s command
    p.signals["gyro.x"] = sig("gyro.x", t, gyro)
    for k in ("gyro.y", "gyro.z"):
        p.signals[k] = sig(k, t, rng.normal(0, 0.06, t.size))
    f = feats(p)
    assert abs(f["trk.rate_rms.roll"].value - 0.1) < 0.01, f["trk.rate_rms.roll"].value
    assert f["trk.rate_rms.pitch"].reason == "insufficient_manoeuvring"
    print(f"PASS: roll ratio {f['trk.rate_rms.roll'].value:.3f} over 30 s of manoeuvring (true 0.1); "
          "pitch never manoeuvred -> null")


def test_unqualified_flight_returns_every_feature_null():
    fs = extract_features(flight(duration=60.0, airborne=(10.0, 50.0), manoeuvre=None),
                          window_flight(flight(duration=60.0, airborne=(10.0, 50.0), manoeuvre=None)))
    assert [f.key for f in fs] == [k for k, *_ in FEATURE_CATALOGUE]
    assert all(f.value is None and f.reason.startswith("flight_not_qualified") for f in fs)
    print("PASS: unqualified flight -> every catalogue feature null with the windowing reason")


# ---------------------------------------------------------------- step 3: the real default-profile log

def test_real_log_every_scored_feature_computes_and_r3_is_null():
    if not HAS_REAL:
        print("SKIP: real fixture absent"); return
    _, w, f = real()
    assert w.qualified and w.steady_qualified
    assert list(f) == [k for k, *_ in FEATURE_CATALOGUE]
    failures = []
    for feat in f.values():
        has = feat.value is not None or feat.label is not None
        if feat.tier in ("R1", "R2") and not has:
            failures.append(f"{feat.key}: {feat.reason}")
        if feat.tier == "R3" and (has or not feat.reason):
            failures.append(f"{feat.key} (R3) should be null with a reason")
    assert not failures, failures
    assert f["vib.band_hi"].reason == "requires_fifo_accel"
    assert f["pwr.cell_delta"].reason == "no_per_cell_sensing"
    print(f"PASS: all {sum(x.tier != 'R3' for x in f.values())} R1/R2 features compute on px4_777ed799.ulg; "
          "both R3 features null with reasons")


def test_ardupilot_firmware_from_real_vehicle_banners():
    """Real DataFlash boot banners name the vehicle ("ArduCopter V4.5.7 (2ba6b7aa)"),
    never "ArduPilot", so the firmware was always missing on real logs."""
    import tempfile
    from app.services.reality_gap.parsers import ardupilot_firmware, parse_log
    from tests.fixtures.reality_gap.make_fixtures import FIXTURE_DIR, _df_data_message, _df_fmt_message
    for banner in ("ArduCopter V4.5.7 (2ba6b7aa)", "ArduPlane V4.5.1 (f2d4a5e8)", "ArduRover V4.4.0 (c4b9e3b2)",
                   "ArduSub V4.1.2 (8f9f1d3c)", "APM:Copter V3.6.12 (5ae3c4f0)", "ArduPilot Blimp V4.5.0"):
        assert ardupilot_firmware(["ChibiOS: 6d2ba67f", banner, "Frame: HEXA/X"]) == banner, banner
    assert ardupilot_firmware(["ChibiOS: 6d2ba67f", "Frame: HEXA/X", "EKF3 IMU0 is using GPS"]) is None
    # Not anchored mid-message: a later text that mentions a vehicle isn't a banner.
    assert ardupilot_firmware(["PreArm: ArduCopter needs GPS"]) is None
    # End to end: a MSG record appended to the committed DataFlash fixture.
    with open(os.path.join(FIXTURE_DIR, "synthetic_real.bin"), "rb") as fh:
        blob = fh.read()
    blob += _df_fmt_message(0x8C, "MSG", "QZ", ["TimeUS", "Message"])
    blob += _df_data_message(0x8C, "QZ", [12_000_000, "ArduCopter V4.5.7 (2ba6b7aa)"])
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
        fh.write(blob)
    try:
        assert parse_log(fh.name).meta.firmware_version == "ArduCopter V4.5.7 (2ba6b7aa)"
    finally:
        os.unlink(fh.name)
    print("PASS: ArduCopter/Plane/Rover/Sub banners read as firmware; mid-message mentions don't")


REAL_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "reality_gap", "real")


def test_log_start_time_comes_from_a_clock_not_the_timezone_field():
    """PX4's time_ref_utc is SDLOG_UTC_OFFSET in seconds (28800 = UTC+8 on
    px4_641d242a), not a date. The date comes from boot_time_utc_us, else the
    first GPS sample with a 3D fix, wound back to the start of the log."""
    import types
    from app.services.reality_gap.parsers import _dataflash_start_utc, _ulog_start_utc, parse_log
    ulog = types.SimpleNamespace(start_timestamp=10_000_000)
    assert _ulog_start_utc(ulog, None, {"time_ref_utc": 28800}) is None
    gps = types.SimpleNamespace(data={
        "timestamp": np.array([12_000_000, 13_000_000, 14_000_000]),
        "fix_type": np.array([0, 3, 3]),
        "time_utc_usec": np.array([0, 1_788_582_711_000_000, 1_788_582_712_000_000]),
    })
    # The first 3D-fix sample is 3 s into the log: start = its UTC minus 3 s.
    assert _ulog_start_utc(ulog, gps, {"time_ref_utc": 28800}) == "2026-09-05T04:31:48+00:00"
    assert _ulog_start_utc(ulog, gps, {"boot_time_utc_us": 1_788_582_700_000_000}) == "2026-09-05T04:31:50+00:00"
    no_fix = types.SimpleNamespace(data=dict(gps.data, fix_type=np.array([0, 2, 2])))
    assert _ulog_start_utc(ulog, no_fix, {}) is None
    # DataFlash: GPS week 2438 + 4 days is 2026-10-01 00:00:00 GPS time, which is
    # 23:59:42 UTC on 30 Sep (18 leap seconds); the fix came 5 s into the log.
    collected = {"IMU": {"_t": [1_000_000.0]},
                 "GPS": {"_t": [2_000_000.0, 6_000_000.0], "Status": [1.0, 3.0], "GWk": [0.0, 2438.0],
                         "GMS": [0.0, 345_600_000.0]}}
    assert _dataflash_start_utc(collected) == "2026-09-30T23:59:37+00:00"
    real = os.path.join(REAL_FIXTURES, "px4_777ed799.ulg")
    if os.path.exists(real):
        assert parse_log(real).meta.start_time_utc.startswith("2026-09-05T04:31:5")
    print("PASS: dates from boot_time_utc_us or a GPS fix; time_ref_utc (a timezone) never read as one")


def main():
    tests = [v for k, v in sorted(globals().items(), key=lambda kv: list(globals()).index(kv[0]))
             if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: {t.__name__} -- {exc!r}")
    print(f"\n{passed}/{len(tests)} D1 health tests passed.")
    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
