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
        s = p00 + self.r  # S = H@P@H^T + R
        if s == 0:
            return

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

    results = [{
        "timestamp": readings[0][0],
        "raw_altitude": readings[0][1],
        "filtered_altitude": round(kf.position, 3),
        "vertical_velocity_mps": round(kf.velocity, 3),
        "altitude_std_dev": round(kf.position_std_dev, 3),
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
        })

    return results
