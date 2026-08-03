"""
WHY THIS EXISTS:
A student setting up a real flight controller has to calibrate the IMU
accelerometer, IMU gyroscope, compass, and ESCs before a drone is safe to
fly -- but "calibrate it and hope" isn't very educational. This gives a
concrete, numeric answer to "did that calibration actually work": given a
batch of logged sensor readings captured during (or right after) a
calibration step, it computes the same two things every real IMU
calibration procedure checks for -- BIAS (a systematic offset from the
true/expected value) and NOISE (how much the reading jitters around its
own average) -- and compares both against typical, documented acceptance
thresholds for consumer MEMS sensors used in hobby/educational flight
controllers.

WHY BIAS + NOISE, NOT SOMETHING FANCIER:
This is the standard first-level IMU calibration diagnostic -- the
"static test": hold the sensor still (or, for the compass, rotate slowly
through headings) and check the resulting readings against what a
PERFECT sensor would report. More sophisticated approaches exist (e.g.
Allan variance analysis for characterizing noise across many timescales,
used in real navigation-grade IMU datasheets) but require much longer
logged sessions and are overkill for an educational calibration-health
check -- bias+noise from a single static batch is exactly what popular
calibration guides (ArduPilot's and PX4's own accelerometer/gyro/compass
calibration wizards, and general hobbyist IMU calibration tutorials) walk
a user through checking by hand.

WHERE EACH THRESHOLD/EXPECTED VALUE COMES FROM:

1. ACCELEROMETER -- while stationary in a known, level orientation, a
   PERFECT accelerometer reads exactly (0, 0, +9.81) m/s^2 (0 on the
   level axes, standard gravity -- the same 9.81 m/s^2 constant already
   used throughout this app, e.g. motor_performance.py) on the vertical
   axis. Bias/noise thresholds (0.5 m/s^2 bias, 0.3 m/s^2 noise) are
   representative of the typical post-calibration tolerance commonly
   used in hobby/educational flight-controller calibration guides for
   consumer MEMS accelerometers (e.g. the InvenSense MPU-6xxx/ICM-2xxxx
   family used on common ArduPilot/PX4 flight controllers) -- documented
   representative values, not a specific chip's datasheet spec, flagged
   honestly the same way this app's other "typical published value"
   constants are (see motor_performance.py's CT_STATIC).

2. GYROSCOPE -- while stationary, a PERFECT gyroscope reads exactly 0
   deg/s on all three axes (no rotation happening). Bias/noise
   thresholds (1.0 deg/s bias, 0.5 deg/s noise) are representative
   typical values for the same class of consumer MEMS gyros.

3. COMPASS/MAGNETOMETER -- this one is NOT a bias-from-a-fixed-value
   check (heading is SUPPOSED to change as you rotate through a
   calibration swing) -- it's a FIELD MAGNITUDE CONSISTENCY check: at a
   single physical location, Earth's magnetic field has one fixed total
   STRENGTH regardless of which way the sensor is pointed -- only the
   DIRECTION of the reading should change as you rotate, not its
   magnitude sqrt(mx^2+my^2+mz^2). A magnetometer with uncorrected hard-
   iron/soft-iron distortion will show the magnitude visibly changing
   with heading; the standard calibration validation check (used in
   ArduPilot/PX4 compass calibration writeups and general magnetometer
   calibration tutorials) is exactly this: rotate through many headings
   and confirm the field MAGNITUDE stays consistent (low coefficient of
   variation) even though the raw x/y/z components and heading angle
   themselves are constantly changing by design.

4. ESC (electronic speed controller) -- not a bias/noise-from-rest check
   at all; the standard ESC calibration validation is a SYMMETRIC
   RESPONSE check: command the same throttle to every motor and confirm
   they all respond consistently (similar RPM or current draw) -- large
   discrepancies between motors at the same commanded throttle is the
   standard hobbyist diagnostic for an ESC that still needs (re)calibrating
   or has drifted out of sync with its siblings. Implemented here via the
   coefficient of variation (std dev / mean) across the per-motor
   readings, with 5% used as the representative acceptable threshold
   commonly cited in ESC calibration/motor-matching guides.

WHAT WAS VERIFIED BEFORE SHIPPING:
- Fed synthetic accel/gyro readings built from KNOWN injected bias and
  noise values and confirmed the computed bias/noise numerically match
  what was injected (not just "some number comes out") -- see
  test_sensor_calibration.py.
- Confirmed the compass magnitude-consistency check correctly PASSES for
  synthetic readings with a constant field magnitude (direction varying,
  magnitude fixed -- simulating a well-calibrated magnetometer) and FAILS
  when magnitude is made to visibly vary with heading (simulating
  uncorrected hard-iron distortion).
"""

import math
from typing import List, Optional

GRAVITY_MPS2 = 9.81  # standard gravity, same constant already used elsewhere in this app

ACCEL_BIAS_THRESHOLD_MPS2 = 0.5
ACCEL_NOISE_THRESHOLD_MPS2 = 0.3

GYRO_BIAS_THRESHOLD_DEG_S = 1.0
GYRO_NOISE_THRESHOLD_DEG_S = 0.5

COMPASS_MAGNITUDE_CV_THRESHOLD = 0.05  # 5% coefficient of variation in field magnitude across headings

ESC_RESPONSE_CV_THRESHOLD = 0.05  # 5% coefficient of variation in per-motor response at the same commanded throttle


def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _std_dev(values: List[float], mean: Optional[float] = None) -> float:
    m = mean if mean is not None else _mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return variance ** 0.5


def _bias_noise_check(
    readings: List[float], expected_value: float, bias_threshold: float, noise_threshold: float
) -> dict:
    mean = _mean(readings)
    bias = mean - expected_value
    noise_std_dev = _std_dev(readings, mean)
    passed = abs(bias) <= bias_threshold and noise_std_dev <= noise_threshold
    return {
        "bias": round(bias, 4),
        "noise_std_dev": round(noise_std_dev, 4),
        "sample_size": len(readings),
        "passed": passed,
    }


def check_accelerometer(accel_x: List[float], accel_y: List[float], accel_z: List[float]) -> dict:
    """Stationary, level orientation: expected (0, 0, +9.81) m/s^2."""
    x = _bias_noise_check(accel_x, 0.0, ACCEL_BIAS_THRESHOLD_MPS2, ACCEL_NOISE_THRESHOLD_MPS2)
    y = _bias_noise_check(accel_y, 0.0, ACCEL_BIAS_THRESHOLD_MPS2, ACCEL_NOISE_THRESHOLD_MPS2)
    z = _bias_noise_check(accel_z, GRAVITY_MPS2, ACCEL_BIAS_THRESHOLD_MPS2, ACCEL_NOISE_THRESHOLD_MPS2)
    return {
        "x": x, "y": y, "z": z,
        "passed": x["passed"] and y["passed"] and z["passed"],
        "expected_bias_threshold_mps2": ACCEL_BIAS_THRESHOLD_MPS2,
        "expected_noise_threshold_mps2": ACCEL_NOISE_THRESHOLD_MPS2,
    }


def check_gyroscope(gyro_x: List[float], gyro_y: List[float], gyro_z: List[float]) -> dict:
    """Stationary: expected 0 deg/s on all three axes."""
    x = _bias_noise_check(gyro_x, 0.0, GYRO_BIAS_THRESHOLD_DEG_S, GYRO_NOISE_THRESHOLD_DEG_S)
    y = _bias_noise_check(gyro_y, 0.0, GYRO_BIAS_THRESHOLD_DEG_S, GYRO_NOISE_THRESHOLD_DEG_S)
    z = _bias_noise_check(gyro_z, 0.0, GYRO_BIAS_THRESHOLD_DEG_S, GYRO_NOISE_THRESHOLD_DEG_S)
    return {
        "x": x, "y": y, "z": z,
        "passed": x["passed"] and y["passed"] and z["passed"],
        "expected_bias_threshold_deg_s": GYRO_BIAS_THRESHOLD_DEG_S,
        "expected_noise_threshold_deg_s": GYRO_NOISE_THRESHOLD_DEG_S,
    }


def check_compass(mag_x: List[float], mag_y: List[float], mag_z: List[float]) -> dict:
    """
    Field-magnitude consistency check across a rotation through many
    headings -- see module docstring for why this (not a fixed-value
    bias check) is the correct compass calibration validation.
    """
    magnitudes = [
        math.sqrt(x ** 2 + y ** 2 + z ** 2) for x, y, z in zip(mag_x, mag_y, mag_z)
    ]
    mean_magnitude = _mean(magnitudes)
    noise_std_dev = _std_dev(magnitudes, mean_magnitude)
    coefficient_of_variation = noise_std_dev / mean_magnitude if mean_magnitude > 0 else float("inf")
    passed = coefficient_of_variation <= COMPASS_MAGNITUDE_CV_THRESHOLD
    return {
        "mean_field_magnitude": round(mean_magnitude, 4),
        "magnitude_noise_std_dev": round(noise_std_dev, 4),
        "coefficient_of_variation": round(coefficient_of_variation, 4),
        "sample_size": len(magnitudes),
        "passed": passed,
        "expected_cv_threshold": COMPASS_MAGNITUDE_CV_THRESHOLD,
    }


def check_esc(motor_readings: List[float]) -> dict:
    """
    Symmetric-response check -- motor_readings is one reading per motor
    (e.g. RPM or current draw) at the SAME commanded throttle.
    """
    mean_response = _mean(motor_readings)
    std_dev = _std_dev(motor_readings, mean_response)
    coefficient_of_variation = std_dev / mean_response if mean_response > 0 else float("inf")
    passed = coefficient_of_variation <= ESC_RESPONSE_CV_THRESHOLD
    return {
        "mean_response": round(mean_response, 2),
        "response_std_dev": round(std_dev, 2),
        "coefficient_of_variation": round(coefficient_of_variation, 4),
        "num_motors": len(motor_readings),
        "passed": passed,
        "expected_cv_threshold": ESC_RESPONSE_CV_THRESHOLD,
    }


def analyze_calibration(
    accel_x: Optional[List[float]] = None,
    accel_y: Optional[List[float]] = None,
    accel_z: Optional[List[float]] = None,
    gyro_x: Optional[List[float]] = None,
    gyro_y: Optional[List[float]] = None,
    gyro_z: Optional[List[float]] = None,
    mag_x: Optional[List[float]] = None,
    mag_y: Optional[List[float]] = None,
    mag_z: Optional[List[float]] = None,
    esc_readings: Optional[List[float]] = None,
) -> dict:
    """
    Runs whichever calibration steps have data supplied (an educational
    step-by-step UI is expected to call this once per completed step, not
    necessarily all four at once) -- steps without data are reported as
    None rather than a fabricated result, and overall_passed only
    considers the steps actually checked.
    """
    result = {"accelerometer": None, "gyroscope": None, "compass": None, "esc": None}

    if accel_x and accel_y and accel_z:
        result["accelerometer"] = check_accelerometer(accel_x, accel_y, accel_z)
    if gyro_x and gyro_y and gyro_z:
        result["gyroscope"] = check_gyroscope(gyro_x, gyro_y, gyro_z)
    if mag_x and mag_y and mag_z:
        result["compass"] = check_compass(mag_x, mag_y, mag_z)
    if esc_readings:
        result["esc"] = check_esc(esc_readings)

    checked_steps = [v for v in result.values() if v is not None]
    result["steps_checked"] = len(checked_steps)
    result["overall_passed"] = (
        all(step["passed"] for step in checked_steps) if checked_steps else None
    )
    return result
