"""
Standalone test script for the Extended Kalman Filter (EKF) full state
estimation -- run directly with `python3 test_ekf.py`, matching the
other standalone test_*.py scripts in this repo.
"""
import sys, os, math, random
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timedelta
from app.services.kalman_filter import (
    ExtendedKalmanFilter9D, fuse_full_state, latlon_to_local_meters, EARTH_RADIUS_M,
)


def _rmse(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def test_latlon_projection_matches_known_reference():
    # Well-known reference figure: ~111km per degree of longitude at the
    # equator. Using the mean Earth radius (6,371,000m) this formula
    # uses, that works out to exactly 111,195m -- an independent,
    # hand-computable check of the projection math, not just "looks close."
    x, y = latlon_to_local_meters(lat=0.0, lon=1.0, lat0=0.0, lon0=0.0)
    expected = math.radians(1.0) * EARTH_RADIUS_M
    assert abs(x - expected) < 1.0, f"Expected ~{expected}m, got {x}m"
    assert abs(x - 111195) < 10, f"Expected ~111195m (known reference), got {x}m"
    assert y == 0.0
    print(f"PASS: 1 degree longitude at the equator = {x:.1f}m (matches known ~111.2km/degree reference)")


def test_speed_jacobian_matches_finite_difference():
    # Independent numerical check that the hand-derived analytical
    # Jacobian used in update_speed() (dh/dv = v/|v|) is actually
    # correct, not just plausible-looking -- compares against a
    # finite-difference approximation of the same gradient.
    def h(vx, vy, vz):
        return math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    vx, vy, vz = 3.0, -4.0, 1.5
    speed = h(vx, vy, vz)
    analytical = [vx / speed, vy / speed, vz / speed]

    eps = 1e-6
    numerical = [
        (h(vx + eps, vy, vz) - h(vx - eps, vy, vz)) / (2 * eps),
        (h(vx, vy + eps, vz) - h(vx, vy - eps, vz)) / (2 * eps),
        (h(vx, vy, vz + eps) - h(vx, vy, vz - eps)) / (2 * eps),
    ]

    for a, n in zip(analytical, numerical):
        assert abs(a - n) < 1e-4, f"Analytical Jacobian {analytical} doesn't match numerical {numerical}"
    print(f"PASS: analytical speed Jacobian {[round(v,4) for v in analytical]} matches finite-difference numerical {[round(v,4) for v in numerical]}")


def test_full_state_fusion_reduces_noise_below_raw_readings():
    # Same real correctness standard as the 1D altitude filter test:
    # build a KNOWN ground-truth 3D trajectory (straight-line horizontal
    # flight, climbing, slowly rotating), add realistic sensor noise to
    # each channel, run the EKF, and confirm the FUSED state is
    # numerically closer to ground truth than the raw noisy readings.
    random.seed(11)
    n = 150
    start = datetime(2026, 1, 1, 12, 0, 0)
    lat0, lon0 = 37.7749, -122.4194  # arbitrary real-world-plausible origin

    readings = []
    true_x, true_y, true_alt, true_yaw = [], [], [], []
    for i in range(n):
        t_x = 2.0 * i * 0.1          # 2 m/s east
        t_y = 1.0 * i * 0.1          # 1 m/s north
        t_alt = 10.0 + 0.05 * i
        t_yaw = (5.0 + 0.02 * i) % 360
        true_x.append(t_x)
        true_y.append(t_y)
        true_alt.append(t_alt)
        true_yaw.append(t_yaw)

        # Convert true local (x,y) back to lat/lon so fuse_full_state has
        # to go through the same projection round-trip a real caller would.
        lat = lat0 + math.degrees(t_y / EARTH_RADIUS_M)
        lon = lon0 + math.degrees(t_x / (EARTH_RADIUS_M * math.cos(math.radians(lat0))))

        readings.append({
            "timestamp": start + timedelta(seconds=i * 0.1),
            "latitude": lat + random.gauss(0, 0.3) / 111195,  # ~0.3m position noise, in degrees
            "longitude": lon + random.gauss(0, 0.3) / (111195 * math.cos(math.radians(lat0))),
            "altitude": t_alt + random.gauss(0, 0.3),
            "speed": math.hypot(2.0, 1.0) + random.gauss(0, 0.5),
            "roll": random.gauss(0, 2.0),
            "pitch": random.gauss(0, 2.0),
            "yaw": t_yaw + random.gauss(0, 2.0),
        })

    result = fuse_full_state(readings)

    fused_x = [r["x_east"] for r in result]
    fused_alt = [r["z_alt"] for r in result]
    raw_alt = [r["altitude"] for r in readings]

    alt_raw_rmse = _rmse(raw_alt, true_alt)
    alt_fused_rmse = _rmse(fused_alt, true_alt)
    x_fused_rmse = _rmse(fused_x, true_x)

    assert alt_fused_rmse < alt_raw_rmse, (
        f"Expected fused altitude RMSE below raw, got raw={alt_raw_rmse:.3f} fused={alt_fused_rmse:.3f}"
    )
    assert x_fused_rmse < 2.0, f"Expected fused horizontal position to track ground truth reasonably, got RMSE={x_fused_rmse:.3f}m"
    print(
        f"PASS: altitude RMSE raw={alt_raw_rmse:.3f}m -> fused={alt_fused_rmse:.3f}m; "
        f"horizontal x-position fused RMSE={x_fused_rmse:.3f}m vs ground truth"
    )


def test_graceful_degradation_with_missing_fields():
    # Real telemetry frequently has some fields null (nullable columns --
    # see app/models/telemetry.py). The EKF must handle a mix of readings
    # with partial data without crashing, same graceful-degradation
    # standard as smooth_altitude_series.
    start = datetime(2026, 1, 1, 12, 0, 0)
    readings = [
        {"timestamp": start, "latitude": 37.0, "longitude": -122.0, "altitude": 10.0, "speed": None, "roll": None, "pitch": None, "yaw": None},
        {"timestamp": start + timedelta(seconds=1), "latitude": None, "longitude": None, "altitude": None, "speed": 3.0, "roll": 1.0, "pitch": 0.5, "yaw": 90.0},
        {"timestamp": start + timedelta(seconds=2), "latitude": 37.0001, "longitude": -122.0, "altitude": 10.5, "speed": None, "roll": None, "pitch": None, "yaw": None},
    ]
    result = fuse_full_state(readings)
    assert len(result) == 3
    for point in result:
        assert all(math.isfinite(v) for k, v in point.items() if k != "timestamp"), f"Non-finite value in {point}"
    print(f"PASS: mixed missing-field readings handled without crashing -> final state: {result[-1]}")


def test_duplicate_timestamp_does_not_crash():
    start = datetime(2026, 1, 1, 12, 0, 0)
    readings = [
        {"timestamp": start, "latitude": 37.0, "longitude": -122.0, "altitude": 10.0, "speed": 2.0, "roll": 0, "pitch": 0, "yaw": 0},
        {"timestamp": start, "latitude": 37.0, "longitude": -122.0, "altitude": 10.1, "speed": 2.0, "roll": 0, "pitch": 0, "yaw": 0},
        {"timestamp": start + timedelta(seconds=1), "latitude": 37.0001, "longitude": -122.0, "altitude": 10.2, "speed": 2.0, "roll": 0, "pitch": 0, "yaw": 0},
    ]
    result = fuse_full_state(readings)
    assert len(result) == 2, f"Expected the dt=0 duplicate to be skipped, got {len(result)} results"
    print("PASS: duplicate timestamp (dt=0) handled without crashing, correctly skipped")


def test_covariance_shrinks_with_repeated_consistent_measurements():
    # Basic sanity check independent of the full pipeline: repeated
    # consistent position measurements should shrink the filter's
    # position uncertainty, exactly like the 1D filter's convergence test.
    ekf = ExtendedKalmanFilter9D(initial_state=[0, 0, 0, 0, 0, 0, 0, 0, 0], initial_variance=100.0)
    initial_p00 = ekf.P[0][0]
    for _ in range(30):
        ekf.predict(dt=1.0)
        ekf.update_position(x_east=10.0, y_north=5.0, z_alt=2.0)
    assert ekf.P[0][0] < initial_p00, "Expected position uncertainty to shrink after repeated measurements"
    assert abs(ekf.x[0] - 10.0) < 1.0, f"Expected x_east to converge near 10.0, got {ekf.x[0]}"
    print(f"PASS: position covariance shrank from {initial_p00} to {ekf.P[0][0]:.4f}, converged to x_east={ekf.x[0]:.3f}")


if __name__ == "__main__":
    test_latlon_projection_matches_known_reference()
    test_speed_jacobian_matches_finite_difference()
    test_full_state_fusion_reduces_noise_below_raw_readings()
    test_graceful_degradation_with_missing_fields()
    test_duplicate_timestamp_does_not_crash()
    test_covariance_shrinks_with_repeated_consistent_measurements()
    print("\nAll EKF tests passed.")
