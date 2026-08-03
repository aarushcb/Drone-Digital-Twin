"""
Standalone test script for the sensor calibration guide -- run directly
with `python3 test_sensor_calibration.py`, matching the other standalone
test_*.py scripts in this repo.
"""
import sys, os, random, math
sys.path.insert(0, os.path.dirname(__file__))

from app.services.sensor_calibration import (
    check_accelerometer, check_gyroscope, check_compass, check_esc, analyze_calibration,
    GRAVITY_MPS2,
)


def test_accelerometer_recovers_known_injected_bias_and_noise():
    # The real correctness check: build synthetic readings from a KNOWN
    # injected bias and noise std dev, and confirm the computed bias/noise
    # numerically match what was actually injected -- not just "some
    # plausible-looking number comes out."
    random.seed(1)
    injected_bias_z = 0.15
    injected_noise_std = 0.05
    n = 200
    accel_x = [random.gauss(0, injected_noise_std) for _ in range(n)]
    accel_y = [random.gauss(0, injected_noise_std) for _ in range(n)]
    accel_z = [random.gauss(GRAVITY_MPS2 + injected_bias_z, injected_noise_std) for _ in range(n)]

    result = check_accelerometer(accel_x, accel_y, accel_z)
    assert abs(result["z"]["bias"] - injected_bias_z) < 0.02, (
        f"Expected recovered bias near {injected_bias_z}, got {result['z']['bias']}"
    )
    assert abs(result["z"]["noise_std_dev"] - injected_noise_std) < 0.02, (
        f"Expected recovered noise near {injected_noise_std}, got {result['z']['noise_std_dev']}"
    )
    assert result["passed"] is True, f"Expected this well-calibrated sensor to pass, got {result}"
    print(f"PASS: injected bias={injected_bias_z}, recovered={result['z']['bias']}; injected noise={injected_noise_std}, recovered={result['z']['noise_std_dev']}")


def test_accelerometer_with_large_bias_fails():
    # A poorly-calibrated accelerometer with a large systematic offset
    # should be correctly flagged as failing, not silently passed.
    random.seed(2)
    n = 100
    accel_x = [random.gauss(0, 0.05) for _ in range(n)]
    accel_y = [random.gauss(0, 0.05) for _ in range(n)]
    accel_z = [random.gauss(GRAVITY_MPS2 + 1.5, 0.05) for _ in range(n)]  # 1.5 m/s^2 bias -- clearly miscalibrated

    result = check_accelerometer(accel_x, accel_y, accel_z)
    assert result["passed"] is False, f"Expected a large bias to fail calibration, got {result}"
    print(f"PASS: large injected bias (1.5 m/s^2) correctly fails -> z bias={result['z']['bias']}")


def test_gyroscope_at_rest_recovers_known_bias():
    random.seed(3)
    injected_bias = 0.3
    n = 150
    gyro_x = [random.gauss(injected_bias, 0.1) for _ in range(n)]
    gyro_y = [random.gauss(0, 0.1) for _ in range(n)]
    gyro_z = [random.gauss(0, 0.1) for _ in range(n)]

    result = check_gyroscope(gyro_x, gyro_y, gyro_z)
    assert abs(result["x"]["bias"] - injected_bias) < 0.05
    assert result["passed"] is True
    print(f"PASS: gyro injected bias={injected_bias}, recovered={result['x']['bias']}, passed={result['passed']}")


def test_compass_passes_with_constant_field_magnitude():
    # Simulates a WELL-calibrated magnetometer: field magnitude stays
    # constant across many different headings, even though the raw x/y/z
    # components (and implied heading) change completely with each sample
    # -- exactly the real-world signature of a good calibration.
    random.seed(4)
    true_magnitude = 45.0  # microtesla-ish, arbitrary units -- only the CONSISTENCY matters
    n = 100
    mag_x, mag_y, mag_z = [], [], []
    for i in range(n):
        heading = 2 * math.pi * i / n
        # Small measurement noise on top of a perfectly constant magnitude.
        noisy_magnitude = true_magnitude + random.gauss(0, 0.3)
        mag_x.append(noisy_magnitude * math.cos(heading))
        mag_y.append(noisy_magnitude * math.sin(heading))
        mag_z.append(random.gauss(0, 0.1))

    result = check_compass(mag_x, mag_y, mag_z)
    assert result["passed"] is True, f"Expected a well-calibrated compass to pass, got {result}"
    print(f"PASS: well-calibrated compass (constant field magnitude across headings) -> CV={result['coefficient_of_variation']}")


def test_compass_fails_with_heading_dependent_magnitude():
    # Simulates UNCORRECTED hard-iron distortion: field magnitude visibly
    # varies with heading (a real, standard signature of bad compass
    # calibration) -- should be flagged as failing.
    random.seed(5)
    n = 100
    mag_x, mag_y, mag_z = [], [], []
    for i in range(n):
        heading = 2 * math.pi * i / n
        # Magnitude itself swings between 30 and 60 depending on heading --
        # exactly what hard-iron distortion looks like.
        distorted_magnitude = 45.0 + 15.0 * math.sin(heading)
        mag_x.append(distorted_magnitude * math.cos(heading))
        mag_y.append(distorted_magnitude * math.sin(heading))
        mag_z.append(0.0)

    result = check_compass(mag_x, mag_y, mag_z)
    assert result["passed"] is False, f"Expected heading-dependent magnitude to fail calibration, got {result}"
    print(f"PASS: hard-iron-distorted compass correctly fails -> CV={result['coefficient_of_variation']}")


def test_esc_symmetric_response_passes_and_asymmetric_fails():
    good_motors = [4500, 4520, 4480, 4510]  # RPM, all motors close together at the same throttle
    result_good = check_esc(good_motors)
    assert result_good["passed"] is True, f"Expected symmetric motor response to pass, got {result_good}"

    bad_motors = [4500, 3800, 4520, 4510]  # one motor way out of line
    result_bad = check_esc(bad_motors)
    assert result_bad["passed"] is False, f"Expected asymmetric motor response to fail, got {result_bad}"
    print(f"PASS: symmetric motors pass (CV={result_good['coefficient_of_variation']}), asymmetric motors fail (CV={result_bad['coefficient_of_variation']})")


def test_analyze_calibration_only_reports_steps_with_data():
    # Step-by-step UI reality: a student may only have completed the
    # accelerometer step so far -- the other three should be honestly
    # reported as not-yet-checked (None), not fabricated results, and
    # overall_passed should reflect only what was actually checked.
    result = analyze_calibration(
        accel_x=[0.01] * 50, accel_y=[-0.01] * 50, accel_z=[9.80] * 50,
    )
    assert result["accelerometer"] is not None
    assert result["gyroscope"] is None
    assert result["compass"] is None
    assert result["esc"] is None
    assert result["steps_checked"] == 1
    assert result["overall_passed"] is True
    print(f"PASS: partial calibration report only includes the accelerometer step -> steps_checked={result['steps_checked']}")


def test_analyze_calibration_overall_fails_if_any_step_fails():
    result = analyze_calibration(
        accel_x=[0.01] * 50, accel_y=[-0.01] * 50, accel_z=[9.80] * 50,  # passes
        gyro_x=[5.0] * 50, gyro_y=[0.0] * 50, gyro_z=[0.0] * 50,  # large bias -- fails
    )
    assert result["accelerometer"]["passed"] is True
    assert result["gyroscope"]["passed"] is False
    assert result["overall_passed"] is False
    assert result["steps_checked"] == 2
    print("PASS: overall_passed correctly reflects a failing step even when another step passes")


if __name__ == "__main__":
    test_accelerometer_recovers_known_injected_bias_and_noise()
    test_accelerometer_with_large_bias_fails()
    test_gyroscope_at_rest_recovers_known_bias()
    test_compass_passes_with_constant_field_magnitude()
    test_compass_fails_with_heading_dependent_magnitude()
    test_esc_symmetric_response_passes_and_asymmetric_fails()
    test_analyze_calibration_only_reports_steps_with_data()
    test_analyze_calibration_overall_fails_if_any_step_fails()
    print("\nAll sensor calibration tests passed.")
