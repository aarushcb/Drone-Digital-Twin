"""
WHY THIS EXISTS:
Real sensor readings are noisy -- a barometric altimeter jitters around
the true altitude reading to reading, even when the drone is perfectly
still. This app currently stores and displays every telemetry reading
as-is, raw noise included (and CLAUDE.md notes duplicate telemetry
occasionally arriving over the WebSocket, adding even more raw jitter to
what's displayed). A Kalman filter is the standard technique real flight
controllers (ArduPilot, PX4) use to fuse a noisy sensor measurement with a
predictive motion model to produce a smoothed, more trustworthy state
estimate -- both an altitude AND a vertical velocity (climb rate) it
didn't directly measure, inferred from how the filtered altitude is
changing. This is a read-side addition: a new endpoint that smooths
already-stored telemetry on request. It does NOT touch the WebSocket
ingestion path, the raw telemetry table, or the existing GET
/drones/{id}/telemetry endpoint -- rerunning this filter over the same
stored readings is always safe and repeatable.

THE MODEL -- CONSTANT-VELOCITY 1D KALMAN FILTER:
State vector x = [altitude, vertical_velocity]. This is the textbook
"discretized continuous white noise acceleration" (DWNA) model:

  Predict:  x_pred = F @ x        F = [[1, dt], [0, 1]]
            P_pred = F @ P @ F^T + Q
  Update:   y = z - H @ x_pred    H = [1, 0]   (only altitude is measured)
            S = H @ P_pred @ H^T + R
            K = P_pred @ H^T / S
            x = x_pred + K * y
            P = (I - K @ H) @ P_pred

These are the standard discrete Kalman filter predict/update equations --
see Welch & Bishop, "An Introduction to the Kalman Filter" (UNC Chapel
Hill TR 95-041, the most widely cited practical KF tutorial) for the
general form, which this directly follows.

The process noise covariance Q uses the standard "discrete white noise
acceleration" model (constant, unknown vertical acceleration between
steps, with variance q):

  Q = q * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]

This exact form is derived in Bar-Shalom, Li & Kirubarajan, "Estimation
with Applications to Tracking and Navigation" (Wiley, 2001), Ch. 6 -- the
standard reference for this specific process noise model.

WHERE THE NOISE NUMBERS COME FROM:
- Measurement noise std (default 0.3m): typical MEMS barometric
  altimeters used on small flight controllers (e.g. the MS5611/BMP280
  family used on ArduPilot/PX4 boards) report altitude RMS noise in the
  documented 0.1-0.5m range depending on onboard filtering -- 0.3m is a
  representative mid-range value, the same "honest documented range,
  not a measured-for-this-app number" standard as bemt.py and
  motor_performance.py. If the underlying sensor is actually GPS-derived
  altitude rather than barometric, real noise would be meaningfully
  higher (commonly several meters) -- this module assumes barometric-
  quality altitude and says so, rather than silently guessing.
- Process noise (accel std, default 0.5 m/s^2): a representative
  assumption for typical hover/slow-cruise vertical acceleration
  variability on a small multirotor (not aggressive aerobatic
  maneuvering) -- higher values make the filter trust new measurements
  more (track fast changes better but smooth less); lower values smooth
  more aggressively but lag behind real altitude changes.

WHAT WAS VERIFIED BEFORE SHIPPING:
Ran the filter against synthetic altitude data with KNOWN ground truth
(a smooth climb/descent profile) plus added random noise matching the
assumed measurement noise std, and confirmed the filtered output's RMSE
against the known ground truth is substantially lower than the raw noisy
readings' RMSE -- see test_kalman_filter.py. This is a real, numeric
noise-reduction check, not a "looks smoother" eyeball test.
"""

import math
from datetime import datetime
from typing import List, Optional, Tuple


class KalmanFilter1D:
    """Constant-velocity 1D Kalman filter -- state = [position, velocity].
    Pure Python 2x2 matrix math (no numpy), matching this app's existing
    convention (see predictive_analytics.py) of not pulling in numpy for
    something this small."""

    def __init__(
        self,
        initial_position: float,
        initial_velocity: float = 0.0,
        initial_variance: float = 10.0,
        process_noise_accel_std: float = 0.5,
        measurement_noise_std: float = 0.3,
    ):
        self.x = [initial_position, initial_velocity]
        # State covariance P, as a flat 2x2: [[p00, p01], [p10, p11]]
        self.P = [[initial_variance, 0.0], [0.0, initial_variance]]
        self.q = process_noise_accel_std ** 2
        self.r = measurement_noise_std ** 2
        self.last_innovation: Optional[float] = None
        self.last_nis: Optional[float] = None

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        f00, f01 = 1.0, dt
        f10, f11 = 0.0, 1.0

        x0, x1 = self.x
        self.x = [f00 * x0 + f01 * x1, f10 * x0 + f11 * x1]

        p00, p01 = self.P[0]
        p10, p11 = self.P[1]
        # FP = F @ P
        fp00 = f00 * p00 + f01 * p10
        fp01 = f00 * p01 + f01 * p11
        fp10 = f10 * p00 + f11 * p10
        fp11 = f10 * p01 + f11 * p11
        # FPFt = FP @ F^T
        fpft00 = fp00 * f00 + fp01 * f01
        fpft01 = fp00 * f10 + fp01 * f11
        fpft10 = fp10 * f00 + fp11 * f01
        fpft11 = fp10 * f10 + fp11 * f11

        q = self.q
        q00 = q * (dt ** 4) / 4
        q01 = q * (dt ** 3) / 2
        q10 = q01
        q11 = q * (dt ** 2)

        self.P = [[fpft00 + q00, fpft01 + q01], [fpft10 + q10, fpft11 + q11]]

    def update(self, measurement: float) -> None:
        p00, p01 = self.P[0]
        p10, p11 = self.P[1]

        innovation = measurement - self.x[0]  # y = z - H@x_pred, H = [1, 0]
        s = p00 + self.r  # S = H@P@H^T + R -- the innovation covariance
        if s == 0:
            return

        # WHY THIS IS SAVED (used by detect_sensor_faults below): the
        # innovation y and its covariance S are exactly what innovation-
        # based fault detection needs -- see the FAULT DETECTION section
        # of this file's module docstring for the statistical reasoning.
        self.last_innovation = innovation
        self.last_nis = (innovation ** 2) / s  # normalized innovation squared

        k0 = p00 / s  # K = P @ H^T / S
        k1 = p10 / s

        self.x = [self.x[0] + k0 * innovation, self.x[1] + k1 * innovation]

        # P = (I - K@H) @ P_pred
        new_p00 = (1 - k0) * p00
        new_p01 = (1 - k0) * p01
        new_p10 = p10 - k1 * p00
        new_p11 = p11 - k1 * p01
        self.P = [[new_p00, new_p01], [new_p10, new_p11]]

    @property
    def position(self) -> float:
        return self.x[0]

    @property
    def velocity(self) -> float:
        return self.x[1]

    @property
    def position_std_dev(self) -> float:
        return math.sqrt(max(self.P[0][0], 0.0))


def smooth_altitude_series(
    readings: List[Tuple[datetime, float]],
    process_noise_accel_std: float = 0.5,
    measurement_noise_std: float = 0.3,
) -> List[dict]:
    """
    Runs the Kalman filter over a chronologically-ordered list of
    (timestamp, altitude) readings, returning the smoothed altitude and
    estimated vertical velocity at each point. `readings` must already be
    sorted oldest-first (same convention as predictive_analytics.py).
    """
    if not readings:
        return []

    kf = KalmanFilter1D(
        initial_position=readings[0][1],
        process_noise_accel_std=process_noise_accel_std,
        measurement_noise_std=measurement_noise_std,
    )

    # First point has no prediction to compare against yet (nothing to
    # form an innovation from), so innovation/nis are reported as None --
    # an honest "not applicable yet," not a fabricated zero.
    results = [{
        "timestamp": readings[0][0],
        "raw_altitude": readings[0][1],
        "filtered_altitude": round(kf.position, 3),
        "vertical_velocity_mps": round(kf.velocity, 3),
        "altitude_std_dev": round(kf.position_std_dev, 3),
        "innovation": None,
        "nis": None,
    }]

    prev_ts = readings[0][0]
    for ts, altitude in readings[1:]:
        dt = (ts - prev_ts).total_seconds()
        prev_ts = ts
        if dt <= 0:
            # Duplicate/out-of-order timestamp (see the known "duplicate
            # telemetry" issue in CLAUDE.md) -- skip the predict step
            # rather than dividing by a zero/negative dt.
            continue
        kf.predict(dt)
        kf.update(altitude)
        results.append({
            "timestamp": ts,
            "raw_altitude": altitude,
            "filtered_altitude": round(kf.position, 3),
            "vertical_velocity_mps": round(kf.velocity, 3),
            "altitude_std_dev": round(kf.position_std_dev, 3),
            "innovation": round(kf.last_innovation, 4) if kf.last_innovation is not None else None,
            "nis": round(kf.last_nis, 3) if kf.last_nis is not None else None,
        })

    return results


# ============================================================================
# SENSOR/ACTUATOR FAULT DETECTION FROM THE INNOVATION SEQUENCE
#
# WHY THIS EXISTS:
# The predictive_analytics.py anomaly detector already flags a CURRENT
# reading as unusual by comparing it to this drone's own historical mean
# (a z-score test). That's a good, real technique, but it only looks at
# the raw value -- it can't distinguish "this altitude reading is unusual
# because the drone is actually climbing fast" from "this altitude
# reading is unusual because the altimeter itself is glitching," since it
# has no model of expected DYNAMICS, just a historical distribution of
# values. The Kalman filter already built for altitude smoothing DOES
# have a dynamics model (the constant-velocity prediction) -- and the gap
# between what that model predicted and what the sensor actually reported
# (the innovation, already computed and saved in KalmanFilter1D.update()
# above) is a much more targeted fault signal: it stays small for a
# healthy sensor tracking real drone motion, and grows for a
# malfunctioning/disconnected/spoofed sensor, REGARDLESS of whether the
# drone's true altitude happens to be "normal" for it historically.
#
# THE STATISTICAL TEST -- NORMALIZED INNOVATION SQUARED (NIS):
# This is real, standard practice, not invented for this app. For a
# correctly-tuned linear Kalman filter operating on a fault-free sensor,
# the normalized innovation squared NIS = innovation^2 / S (S = innovation
# covariance, already computed as `s` in update() above) is chi-square
# distributed with 1 degree of freedom (1 DOF because this is a scalar,
# single-measurement filter) -- this is the standard "innovation
# consistency" / "innovation magnitude" test described in Bar-Shalom, Li
# & Kirubarajan, "Estimation with Applications to Tracking and
# Navigation" (Wiley, 2001), Ch. 5.4 (the same reference already cited in
# this file's main docstring for the process noise model). Using the
# filter's own innovation sequence specifically to detect and diagnose
# sensor/system faults -- rather than just as a smoothing byproduct -- is
# a distinct, established sub-field, originating with Mehra & Peschon,
# "An innovations approach to fault detection and diagnosis in dynamic
# systems" (Automatica, 1971), the founding paper for this exact
# technique.
#
# CHI-SQUARE THRESHOLDS USED (standard tabulated critical values for 1
# degree of freedom, found in any statistics chi-square table):
#   - 3.841 = 95th percentile (5% false-alarm rate per single sample)
#   - 6.635 = 99th percentile (1% false-alarm rate per single sample)
#
# WHY A SINGLE NIS SPIKE ISN'T ENOUGH TO CALL A "FAULT":
# A healthy, correctly-tuned filter will still exceed the 95% threshold
# on about 1 in 20 samples PURELY BY CHANCE (that's what "95th
# percentile" means) -- flagging every such blip as a sensor fault would
# be mostly false alarms. This is why the check below requires several
# CONSECUTIVE samples over threshold: for MIN_CONSECUTIVE_EXCEEDANCES=3
# independent 5%-probability events in a row, the chance of that
# happening by coincidence in a healthy sensor is about
# 0.05^3 ~= 0.000125 (roughly 1 in 8000) -- a much more defensible bar
# for actually flagging a fault rather than normal filter noise.
#
# WHAT WAS VERIFIED BEFORE SHIPPING:
# See test_kalman_filter.py -- confirmed a sensor with injected persistent
# bias/dropout (simulating a stuck or degraded altimeter) triggers a
# flagged fault, while a normal noisy-but-healthy sensor stream (200+
# samples) produces zero false-positive fault flags despite individual
# NIS values occasionally crossing the 95% threshold, exactly as the
# statistics above predict.
# ============================================================================

NIS_THRESHOLD_95 = 3.841   # chi-square critical value, 1 DOF, alpha=0.05
NIS_THRESHOLD_99 = 6.635   # chi-square critical value, 1 DOF, alpha=0.01
MIN_CONSECUTIVE_EXCEEDANCES = 3


def detect_sensor_faults(
    smoothed_points: List[dict],
    nis_threshold: float = NIS_THRESHOLD_95,
    min_consecutive: int = MIN_CONSECUTIVE_EXCEEDANCES,
) -> List[dict]:
    """
    Scans the output of smooth_altitude_series() for runs of consecutive
    NIS values above `nis_threshold` -- see the module docstring above for
    why a single exceedance isn't treated as a fault, but several in a row
    are. Returns one entry per detected fault run (not per point), with
    the run's start/end timestamp and its peak NIS (how far outside the
    expected range the worst point in the run was).
    """
    faults = []
    run_start_idx = None
    run_points = []

    def _close_run(end_idx):
        if run_start_idx is None:
            return
        peak = max(p["nis"] for p in run_points)
        faults.append({
            "start_timestamp": smoothed_points[run_start_idx]["timestamp"],
            "end_timestamp": smoothed_points[end_idx]["timestamp"],
            "num_consecutive_points": len(run_points),
            "peak_nis": round(peak, 2),
            "severity": "CRITICAL" if peak > NIS_THRESHOLD_99 else "WARNING",
        })

    for i, point in enumerate(smoothed_points):
        nis = point.get("nis")
        if nis is not None and nis > nis_threshold:
            if run_start_idx is None:
                run_start_idx = i
            run_points.append(point)
        else:
            if len(run_points) >= min_consecutive:
                _close_run(i - 1)
            run_start_idx = None
            run_points = []

    if len(run_points) >= min_consecutive:
        _close_run(len(smoothed_points) - 1)

    return faults
