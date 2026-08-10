"""
Standalone test script for the attitude control loop simulation -- run
directly with `python3 test_control_loop.py`, matching the other
standalone test_*.py scripts in this repo.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timedelta, timezone

from app.services.control_loop import (
    simulate_step_response, closed_form_metrics, simulate_control_loop,
    build_real_control_loop_response, MIN_REAL_CONTROL_LOOP_SAMPLES,
    STABLE_DAMPING_RATIO, OSCILLATING_DAMPING_RATIO, DEFAULT_NATURAL_FREQUENCY_RAD_S,
)


def _closed_form_theta(t, step_deg, zeta, omega_n, step_start):
    if t < step_start:
        return 0.0
    tau = t - step_start
    omega_d = omega_n * math.sqrt(1 - zeta ** 2)
    phi = math.acos(zeta)
    return step_deg * (1 - (math.exp(-zeta * omega_n * tau) / math.sqrt(1 - zeta ** 2)) * math.sin(omega_d * tau + phi))


def test_numerical_simulation_matches_closed_form_solution():
    # The real correctness check: the RK4-integrated numerical simulation
    # should match the KNOWN ANALYTICAL solution of the same ODE at
    # several time points, not just "produces a smooth-looking curve."
    step_deg = 15.0
    zeta = STABLE_DAMPING_RATIO
    omega_n = DEFAULT_NATURAL_FREQUENCY_RAD_S
    step_start = 0.5

    result = simulate_step_response(
        step_deg=step_deg, damping_ratio=zeta, natural_frequency_rad_s=omega_n,
        duration_s=3.0, sample_rate_hz=50, step_start_s=step_start,
    )

    max_error = 0.0
    for t, actual in zip(result["timestamps_s"], result["actual_deg"]):
        expected = _closed_form_theta(t, step_deg, zeta, omega_n, step_start)
        max_error = max(max_error, abs(actual - expected))

    assert max_error < 0.05, f"Expected numerical simulation to match closed-form solution closely, max error={max_error}"
    print(f"PASS: numerical (RK4) simulation matches closed-form analytical solution (max error={max_error:.5f} deg)")


def test_peak_time_metric_matches_the_simulations_own_actual_peak():
    # Independent check: find the ACTUAL peak in the simulated time series
    # and confirm it lands within one sample of the closed-form
    # peak_time_s formula -- not just trusting the formula in isolation.
    step_deg = 15.0
    zeta = STABLE_DAMPING_RATIO
    omega_n = DEFAULT_NATURAL_FREQUENCY_RAD_S
    step_start = 0.5
    sample_rate = 100

    result = simulate_step_response(
        step_deg=step_deg, damping_ratio=zeta, natural_frequency_rad_s=omega_n,
        duration_s=3.0, sample_rate_hz=sample_rate, step_start_s=step_start,
    )
    metrics = closed_form_metrics(zeta, omega_n)

    peak_idx = max(range(len(result["actual_deg"])), key=lambda i: result["actual_deg"][i])
    simulated_peak_time = result["timestamps_s"][peak_idx] - step_start
    sample_period = 1 / sample_rate

    assert abs(simulated_peak_time - metrics["peak_time_s"]) <= sample_period * 1.5, (
        f"Expected simulated peak time ({simulated_peak_time}) near closed-form peak_time_s "
        f"({metrics['peak_time_s']}), within ~{sample_period*1.5}s sample resolution"
    )
    print(f"PASS: simulated peak at t={simulated_peak_time:.3f}s matches closed-form peak_time_s={metrics['peak_time_s']}s")


def test_overshoot_metric_matches_the_simulations_own_actual_maximum():
    step_deg = 15.0
    zeta = STABLE_DAMPING_RATIO
    omega_n = DEFAULT_NATURAL_FREQUENCY_RAD_S

    result = simulate_step_response(
        step_deg=step_deg, damping_ratio=zeta, natural_frequency_rad_s=omega_n,
        duration_s=3.0, sample_rate_hz=100, step_start_s=0.5,
    )
    metrics = closed_form_metrics(zeta, omega_n)

    simulated_max = max(result["actual_deg"])
    simulated_overshoot_percent = ((simulated_max - step_deg) / step_deg) * 100

    assert abs(simulated_overshoot_percent - metrics["overshoot_percent"]) < 1.0, (
        f"Expected simulated overshoot ({simulated_overshoot_percent:.2f}%) near closed-form "
        f"overshoot_percent ({metrics['overshoot_percent']}%)"
    )
    print(f"PASS: simulated overshoot={simulated_overshoot_percent:.2f}% matches closed-form={metrics['overshoot_percent']}%")


def test_oscillating_scenario_actually_overshoots_more_than_stable():
    # The core pedagogical claim this feature makes: the "oscillating"
    # preset should show MEASURABLY more overshoot and a longer settling
    # time than "stable" -- not just a different arbitrary dataset.
    stable_metrics = closed_form_metrics(STABLE_DAMPING_RATIO, DEFAULT_NATURAL_FREQUENCY_RAD_S)
    oscillating_metrics = closed_form_metrics(OSCILLATING_DAMPING_RATIO, DEFAULT_NATURAL_FREQUENCY_RAD_S)

    assert oscillating_metrics["overshoot_percent"] > stable_metrics["overshoot_percent"]
    assert oscillating_metrics["settling_time_s"] > stable_metrics["settling_time_s"]
    print(
        f"PASS: oscillating preset overshoot={oscillating_metrics['overshoot_percent']}% "
        f"(vs stable={stable_metrics['overshoot_percent']}%), "
        f"settling time={oscillating_metrics['settling_time_s']}s (vs stable={stable_metrics['settling_time_s']}s)"
    )


def test_control_input_is_zero_at_steady_state():
    # Physical sanity check: once the actual attitude has settled onto
    # the desired step value with zero rate, the PD control law's output
    # (commanded angular acceleration) should itself settle near zero --
    # a system at its setpoint with no residual error needs no more
    # correction.
    result = simulate_step_response(
        step_deg=15.0, damping_ratio=STABLE_DAMPING_RATIO, natural_frequency_rad_s=DEFAULT_NATURAL_FREQUENCY_RAD_S,
        duration_s=5.0, sample_rate_hz=50, step_start_s=0.5,
    )
    final_control_input = result["control_input_deg_s2"][-1]
    assert abs(final_control_input) < 0.5, f"Expected near-zero control input at steady state, got {final_control_input}"
    print(f"PASS: control input settles near zero at steady state ({final_control_input} deg/s^2)")


def test_full_control_loop_simulation_returns_all_three_axes():
    result = simulate_control_loop(scenario="stable", duration_s=3.0, sample_rate_hz=50)
    assert set(result["axes"].keys()) == {"roll", "pitch", "yaw"}
    for axis, data in result["axes"].items():
        assert len(data["timestamps_s"]) == len(data["actual_deg"]) == len(data["control_input_deg_s2"])
        assert data["step_command_deg"] > 0
    print(f"PASS: full simulation returns all 3 axes with matching-length time series -> {list(result['axes'].keys())}")


def _make_sample(t, roll, pitch, yaw, desired_roll, desired_pitch, desired_yaw, pwm=None):
    s = {
        "timestamp": t, "roll": roll, "pitch": pitch, "yaw": yaw,
        "desired_roll": desired_roll, "desired_pitch": desired_pitch, "desired_yaw": desired_yaw,
    }
    for i in range(1, 5):
        s[f"motor_pwm_{i}"] = pwm[i - 1] if pwm else None
    return s


def test_too_few_real_samples_returns_none_so_caller_falls_back_to_simulation():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    samples = [_make_sample(t0 + timedelta(seconds=i), 1, 1, 1, 2, 2, 2) for i in range(MIN_REAL_CONTROL_LOOP_SAMPLES - 1)]
    result = build_real_control_loop_response(samples)
    assert result is None
    print(f"PASS: {len(samples)} real samples (below the {MIN_REAL_CONTROL_LOOP_SAMPLES}-sample minimum) returns None")


def test_real_response_reports_data_source_and_honest_tracking_error():
    # A perfect controller (actual == desired always) should show zero
    # tracking error -- the simplest possible correctness check for the
    # RMS/max error formulas.
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    n = 10
    samples = [
        _make_sample(t0 + timedelta(seconds=0.1 * i), 5.0, -3.0, 90.0, 5.0, -3.0, 90.0)
        for i in range(n)
    ]
    result = build_real_control_loop_response(samples)
    assert result["data_source"] == "real"
    assert result["sample_count"] == n
    for axis in ("roll", "pitch", "yaw"):
        assert result["axes"][axis]["rms_error_deg"] == 0.0
        assert result["axes"][axis]["max_error_deg"] == 0.0
    assert result["motor_pwm"] is None
    print("PASS: perfect-tracking real samples give zero RMS/max error and data_source='real'")


def test_real_response_tracking_error_matches_hand_computed_value():
    # A constant 2-degree roll offset for every sample -> RMS and max
    # error should both be exactly 2.0 (not some formula-dependent
    # approximation) -- the simplest non-trivial hand-computed check.
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    samples = [
        _make_sample(t0 + timedelta(seconds=0.1 * i), 3.0, 0.0, 0.0, 5.0, 0.0, 0.0)
        for i in range(MIN_REAL_CONTROL_LOOP_SAMPLES)
    ]
    result = build_real_control_loop_response(samples)
    assert result["axes"]["roll"]["rms_error_deg"] == 2.0
    assert result["axes"]["roll"]["max_error_deg"] == 2.0
    print("PASS: constant 2deg roll offset gives RMS error=2.0 and max error=2.0, hand-computed")


def test_real_response_includes_motor_pwm_when_present():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    samples = [
        _make_sample(t0 + timedelta(seconds=0.1 * i), 0, 0, 0, 0, 0, 0, pwm=[1500, 1500, 1500, 1500])
        for i in range(MIN_REAL_CONTROL_LOOP_SAMPLES)
    ]
    result = build_real_control_loop_response(samples)
    assert result["motor_pwm"] is not None
    assert result["motor_pwm"]["motor_pwm_1"] == [1500] * MIN_REAL_CONTROL_LOOP_SAMPLES
    print("PASS: motor_pwm sub-object populated when telemetry has PWM data")


def test_real_response_timestamps_are_relative_seconds_from_first_sample():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    samples = [
        _make_sample(t0 + timedelta(seconds=2 * i), 0, 0, 0, 0, 0, 0)
        for i in range(MIN_REAL_CONTROL_LOOP_SAMPLES)
    ]
    result = build_real_control_loop_response(samples)
    assert result["axes"]["roll"]["timestamps_s"] == [0.0, 2.0, 4.0, 6.0, 8.0]
    print("PASS: timestamps_s are relative seconds from the first real sample")


if __name__ == "__main__":
    test_numerical_simulation_matches_closed_form_solution()
    test_peak_time_metric_matches_the_simulations_own_actual_peak()
    test_overshoot_metric_matches_the_simulations_own_actual_maximum()
    test_oscillating_scenario_actually_overshoots_more_than_stable()
    test_control_input_is_zero_at_steady_state()
    test_full_control_loop_simulation_returns_all_three_axes()
    test_too_few_real_samples_returns_none_so_caller_falls_back_to_simulation()
    test_real_response_reports_data_source_and_honest_tracking_error()
    test_real_response_tracking_error_matches_hand_computed_value()
    test_real_response_includes_motor_pwm_when_present()
    test_real_response_timestamps_are_relative_seconds_from_first_sample()
    print("\nAll control loop tests passed.")
