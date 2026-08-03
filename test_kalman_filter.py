"""
Standalone test script for the Kalman filter sensor fusion module -- run
directly with `python3 test_kalman_filter.py`, matching the other
standalone test_*.py scripts in this repo.
"""
import sys, os, math, random
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timedelta
from app.services.kalman_filter import (
    smooth_altitude_series, KalmanFilter1D, detect_sensor_faults,
    NIS_THRESHOLD_95,
)


def _rmse(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def test_filter_reduces_noise_below_raw_readings():
    # The real correctness check: build a KNOWN ground-truth altitude
    # profile (a slow climb, matching a realistic drone flight), add
    # random noise matching the assumed measurement noise std, run the
    # filter, and confirm the FILTERED output is numerically closer to
    # the known ground truth than the raw noisy readings are -- not just
    # "looks smoother."
    random.seed(42)
    measurement_noise_std = 0.3
    n = 200
    dt_seconds = 1.0

    start = datetime(2026, 1, 1, 12, 0, 0)
    ground_truth = [5.0 + 0.05 * i for i in range(n)]  # slow steady climb, 0.05 m/s
    readings = []
    for i, true_alt in enumerate(ground_truth):
        noisy = true_alt + random.gauss(0, measurement_noise_std)
        readings.append((start + timedelta(seconds=i * dt_seconds), noisy))

    # A steady climb has essentially zero real acceleration variability,
    # so a tighter process-noise assumption than the general-purpose
    # default (which has to account for real in-flight maneuvering) is
    # the physically appropriate tuning for this specific scenario --
    # a looser process noise would correctly trust the noisy measurements
    # more and smooth less, which isn't a bug, just a different tuning
    # tradeoff (see the module docstring's note on what each knob does).
    result = smooth_altitude_series(
        readings, measurement_noise_std=measurement_noise_std, process_noise_accel_std=0.02
    )

    raw_values = [r[1] for r in readings]
    filtered_values = [r["filtered_altitude"] for r in result]

    raw_rmse = _rmse(raw_values, ground_truth)
    filtered_rmse = _rmse(filtered_values, ground_truth)

    assert filtered_rmse < raw_rmse * 0.6, (
        f"Expected filtered RMSE well below raw RMSE, got raw={raw_rmse:.3f} filtered={filtered_rmse:.3f}"
    )
    print(f"PASS: raw RMSE={raw_rmse:.3f}m, filtered RMSE={filtered_rmse:.3f}m (vs ground truth)")


def test_filter_recovers_a_reasonable_climb_rate():
    # The steady 0.05 m/s climb used above should be recoverable (roughly)
    # from the filter's estimated velocity state once it's had time to
    # converge, even though velocity was never directly measured.
    random.seed(7)
    measurement_noise_std = 0.3
    n = 300
    start = datetime(2026, 1, 1, 12, 0, 0)
    true_climb_rate = 0.05
    readings = [
        (start + timedelta(seconds=i), 5.0 + true_climb_rate * i + random.gauss(0, measurement_noise_std))
        for i in range(n)
    ]
    result = smooth_altitude_series(readings, measurement_noise_std=measurement_noise_std)

    # Average the estimated velocity over the second half (after the
    # filter has converged past its initial transient).
    late_velocities = [r["vertical_velocity_mps"] for r in result[n // 2:]]
    avg_estimated_rate = sum(late_velocities) / len(late_velocities)

    assert abs(avg_estimated_rate - true_climb_rate) < 0.02, (
        f"Expected estimated climb rate near {true_climb_rate}, got {avg_estimated_rate:.4f}"
    )
    print(f"PASS: true climb rate={true_climb_rate} m/s, filter-estimated (converged avg)={avg_estimated_rate:.4f} m/s")


def test_duplicate_timestamp_does_not_crash_or_corrupt_state():
    # Real-world case this app already knows about (see CLAUDE.md's
    # "duplicate telemetry occasionally arrives over WebSocket" open
    # item) -- two readings with the identical timestamp (dt=0) must be
    # handled gracefully (skipped), not crash on a division by zero in
    # the process-noise calculation.
    start = datetime(2026, 1, 1, 12, 0, 0)
    readings = [
        (start, 10.0),
        (start, 10.2),  # duplicate timestamp -- dt = 0
        (start + timedelta(seconds=1), 10.1),
    ]
    result = smooth_altitude_series(readings)
    assert len(result) == 2, f"Expected the dt=0 duplicate to be skipped, got {len(result)} results"
    print("PASS: duplicate timestamp (dt=0) handled without crashing, correctly skipped")


def test_filter_state_converges_toward_a_constant_measurement():
    # Simple sanity check independent of the full pipeline: feeding the
    # same constant value repeatedly should make the filter's position
    # estimate converge to that value and its uncertainty shrink.
    kf = KalmanFilter1D(initial_position=0.0, initial_variance=100.0)
    for _ in range(50):
        kf.predict(dt=1.0)
        kf.update(measurement=42.0)
    assert abs(kf.position - 42.0) < 0.5, f"Expected convergence near 42.0, got {kf.position}"
    assert kf.position_std_dev < 1.0, f"Expected uncertainty to shrink, got std_dev={kf.position_std_dev}"
    print(f"PASS: converged to position={kf.position:.3f} (std_dev={kf.position_std_dev:.3f}) after 50 constant measurements")


def test_healthy_noisy_sensor_produces_no_false_fault_flags():
    # A normal, healthy sensor stream (same synthetic scenario as
    # test_filter_reduces_noise_below_raw_readings) WILL occasionally
    # cross the 95% NIS threshold by pure chance (~5% of points, that's
    # what "95th percentile" means) -- but should essentially never
    # produce 3 CONSECUTIVE exceedances, which is what detect_sensor_faults
    # actually requires before calling something a fault. This is the key
    # false-positive-rate check: real, ordinary sensor noise should not
    # spam fault alerts.
    random.seed(123)
    measurement_noise_std = 0.3
    n = 300
    start = datetime(2026, 1, 1, 12, 0, 0)
    readings = [
        (start + timedelta(seconds=i), 5.0 + 0.05 * i + random.gauss(0, measurement_noise_std))
        for i in range(n)
    ]
    result = smooth_altitude_series(readings, measurement_noise_std=measurement_noise_std)

    nis_values = [p["nis"] for p in result if p["nis"] is not None]
    exceedance_rate = sum(1 for v in nis_values if v > NIS_THRESHOLD_95) / len(nis_values)
    faults = detect_sensor_faults(result)

    # Sanity-check the premise itself: some individual exceedances should
    # exist (otherwise this test wouldn't be testing anything meaningful).
    assert exceedance_rate > 0, "Expected at least some individual NIS exceedances in 300 healthy samples"
    assert faults == [], f"Expected zero fault flags on a healthy sensor stream, got {faults}"
    print(
        f"PASS: healthy sensor stream had {exceedance_rate:.1%} individual NIS exceedances "
        f"(expected, ~5% by chance) but zero flagged faults (requires 3 consecutive)"
    )


def test_injected_sensor_fault_is_detected():
    # A sensor that goes stuck/degraded partway through a flight -- here,
    # a sudden persistent 5m offset starting at sample 100, simulating a
    # stuck-at-wrong-value or badly miscalibrated altimeter -- should
    # trigger a detected fault run, unlike the healthy-noise case above.
    random.seed(99)
    measurement_noise_std = 0.3
    n = 200
    start = datetime(2026, 1, 1, 12, 0, 0)
    readings = []
    for i in range(n):
        true_alt = 5.0 + 0.05 * i
        if i >= 100:
            true_alt += 5.0  # sudden persistent sensor fault injected here
        readings.append((start + timedelta(seconds=i), true_alt + random.gauss(0, measurement_noise_std)))

    result = smooth_altitude_series(readings, measurement_noise_std=measurement_noise_std)
    faults = detect_sensor_faults(result)

    assert len(faults) >= 1, "Expected the injected sensor fault to be detected"
    fault_start = faults[0]["start_timestamp"]
    injected_at = start + timedelta(seconds=100)
    # Detection should fire at or shortly after the fault is injected (the
    # filter needs a few consecutive bad readings before it's confident
    # this isn't just noise -- some lag is expected and correct).
    delay_seconds = (fault_start - injected_at).total_seconds()
    assert 0 <= delay_seconds <= 10, f"Expected detection within ~10s of the injected fault, got {delay_seconds}s delay"
    print(f"PASS: injected fault at t=100s detected at t={100 + delay_seconds:.0f}s -> {faults[0]}")


if __name__ == "__main__":
    test_filter_reduces_noise_below_raw_readings()
    test_filter_recovers_a_reasonable_climb_rate()
    test_duplicate_timestamp_does_not_crash_or_corrupt_state()
    test_filter_state_converges_toward_a_constant_measurement()
    test_healthy_noisy_sensor_produces_no_false_fault_flags()
    test_injected_sensor_fault_is_detected()
    print("\nAll Kalman filter tests passed.")
