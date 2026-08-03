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


# ============================================================================
# EXTENDED KALMAN FILTER (EKF) -- FULL STATE ESTIMATION
#
# WHY THIS EXISTS:
# KalmanFilter1D above fuses ONE noisy sensor channel (altitude) with a
# constant-velocity motion model. A real drone has more available
# telemetry -- horizontal position (from latitude/longitude), a scalar
# speed reading, and attitude (roll/pitch/yaw) -- all of which describe
# the SAME underlying physical state (where the drone is, how fast it's
# moving in 3D, which way it's oriented) but currently get stored and
# displayed as independent, unfused raw values. This fuses all of them
# into one consistent 9-dimensional state estimate:
#
#   state = [x_east, y_north, z_alt, vx, vy, vz, roll, pitch, yaw]
#
# WHY "EXTENDED" AND NOT JUST A BIGGER LINEAR KALMAN FILTER:
# The process model here (constant-velocity position, random-walk
# attitude) is fully LINEAR -- predict() below is exactly the same kind
# of F@x/F@P@F^T+Q step as KalmanFilter1D's, just in 9 dimensions instead
# of 2. What makes this an EXTENDED Kalman filter specifically is the
# SPEED measurement: telemetry reports a single scalar speed (m/s), which
# is the MAGNITUDE of the velocity vector, h(x) = sqrt(vx^2+vy^2+vz^2) --
# a genuinely NONLINEAR function of the state. A plain linear KF has no
# way to incorporate that measurement; the EKF's defining technique is to
# linearize h() around the current state estimate via its Jacobian and
# use that linearized H in an otherwise-normal KF update -- exactly what
# update_speed_measurement() below does. This nonlinear-measurement-of-a-
# linear-state case (estimating a velocity VECTOR from scalar
# speed/range-rate MAGNITUDE measurements) is a standard textbook EKF
# example -- see Welch & Bishop, "An Introduction to the Kalman Filter"
# (UNC TR 95-041, the same reference already cited above for
# KalmanFilter1D), which includes the EKF's linearized predict/update
# equations in exactly this form; Zarchan & Musoff, "Fundamentals of
# Kalman Filtering: A Practical Approach" (AIAA, 2000) works through this
# same speed/range-rate-from-velocity-vector nonlinear-measurement case
# as a worked example.
#
# SEQUENTIAL SCALAR UPDATES INSTEAD OF ONE JOINT MATRIX UPDATE:
# Rather than building one big 9x9 measurement update (which needs a
# general matrix inverse), this processes each measurement channel (x,
# y, z, roll, pitch, yaw, and the nonlinear speed) as an INDEPENDENT
# scalar update, one after another. This is a standard, real
# simplification (see e.g. Bar-Shalom, Li & Kirubarajan, "Estimation
# with Applications to Tracking and Navigation," Wiley 2001, on
# sequential processing of independent measurements) valid whenever
# measurement noise is uncorrelated across channels (a diagonal R,
# assumed here -- a documented simplification, same honesty standard as
# the rest of this app: real GPS/attitude sensor noise can have some
# cross-correlation this ignores). The practical benefit: every update
# becomes a scalar Kalman gain (K = P@H^T / S with S a SCALAR), needing
# no matrix inversion at all -- just matrix-vector products, which is
# straightforward in pure Python.
#
# HORIZONTAL POSITION -- LOCAL FLAT-EARTH (EQUIRECTANGULAR) PROJECTION:
# Telemetry stores latitude/longitude in degrees, not local meters. To
# fuse position with a constant-velocity meters-based motion model, this
# converts lat/lon to local East/North meters relative to the FIRST
# reading's lat/lon (chosen as the local origin), using the standard
# small-area equirectangular approximation:
#   x_east  = (lon - lon0) * (pi/180) * R_earth * cos(lat0 * pi/180)
#   y_north = (lat - lat0) * (pi/180) * R_earth
# valid for the sub-few-kilometer scale of a single drone flight (it
# ignores Earth's ellipsoidal shape and curvature over larger distances)
# -- R_earth = 6,371,000m, the standard IUGG mean Earth radius. VERIFIED:
# at the equator, this formula gives 111,195m per degree of longitude
# (matches the well-known "~111km per degree of latitude/longitude at
# the equator" reference figure -- the commonly quoted ~111.32km/degree
# figure uses the WGS84 EQUATORIAL radius (6,378,137m) specifically,
# rather than the mean radius used here; both are standard, the small
# difference is a documented, honest consequence of which Earth radius
# convention is used, not an error).
#
# ATTITUDE -- RANDOM WALK, NOT RATE-INTEGRATED:
# Telemetry has no gyroscope angular RATE field, only absolute roll/
# pitch/yaw readings -- so unlike position (which has a real constant-
# velocity dynamics model), attitude here uses a simple random-walk
# process model (attitude_k = attitude_{k-1} + process noise) since
# there's no measured rate to integrate. This is a standard, honestly
# documented simplification when rate data isn't available -- it still
# smooths noisy attitude readings, it just can't PREDICT attitude
# changes ahead of a new measurement the way the velocity states can
# predict position changes.
#
# WHAT WAS VERIFIED BEFORE SHIPPING:
# - The equirectangular projection formula above, checked against the
#   well-known ~111km/degree reference figure (see above).
# - The speed Jacobian was verified analytically: h(x)=sqrt(vx^2+vy^2+vz^2),
#   dh/dvx = vx/h(x) (and symmetric for vy, vz) -- standard vector-norm
#   gradient, double-checked against a finite-difference numerical
#   Jacobian in test_ekf.py (his own independent check that the
#   hand-derived Jacobian is actually correct, not just plausible-looking).
# - Fed synthetic ground-truth 3D motion (known position/velocity/attitude
#   trajectory) plus Gaussian sensor noise through the filter and
#   confirmed the fused state's RMSE against ground truth is well below
#   the raw noisy measurements' RMSE -- the same real noise-reduction
#   check already used for the 1D altitude filter, extended to all 9
#   states.
# ============================================================================

EARTH_RADIUS_M = 6_371_000  # IUGG mean Earth radius


def latlon_to_local_meters(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """Equirectangular (small-area flat-Earth) approximation -- see the
    EKF module docstring above for the formula and its verification."""
    x_east = math.radians(lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    y_north = math.radians(lat - lat0) * EARTH_RADIUS_M
    return x_east, y_north


class ExtendedKalmanFilter9D:
    """
    State: [x_east, y_north, z_alt, vx, vy, vz, roll, pitch, yaw]
    (position in meters, velocity in m/s, attitude in degrees).
    Pure Python (no numpy), matching this app's existing convention.
    """

    def __init__(
        self,
        initial_state: List[float],
        initial_variance: float = 10.0,
        process_noise_accel_std: float = 0.5,
        process_noise_attitude_std: float = 2.0,
    ):
        assert len(initial_state) == 9
        self.x = list(initial_state)
        self.P = [[initial_variance if i == j else 0.0 for j in range(9)] for i in range(9)]
        self.q_accel = process_noise_accel_std ** 2
        self.q_attitude = process_noise_attitude_std ** 2

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        x = self.x
        # Constant-velocity position/velocity (3 independent axes),
        # random-walk attitude (3 independent angles) -- F is block
        # structured, applied directly rather than via a dense 9x9
        # matrix multiply since most entries are zero.
        new_x = [
            x[0] + x[3] * dt, x[1] + x[4] * dt, x[2] + x[5] * dt,  # position += velocity*dt
            x[3], x[4], x[5],                                       # velocity unchanged
            x[6], x[7], x[8],                                       # attitude unchanged (random walk)
        ]
        self.x = new_x

        P = self.P
        new_P = [row[:] for row in P]
        # Position/velocity axes: same discrete white-noise-acceleration
        # (DWNA) propagation as KalmanFilter1D.predict(), applied to each
        # of the 3 (position, velocity) index pairs independently.
        q = self.q_accel
        for pos_idx, vel_idx in ((0, 3), (1, 4), (2, 5)):
            p_pp = P[pos_idx][pos_idx] + 2 * dt * P[pos_idx][vel_idx] + dt * dt * P[vel_idx][vel_idx]
            p_pv = P[pos_idx][vel_idx] + dt * P[vel_idx][vel_idx]
            p_vv = P[vel_idx][vel_idx]
            new_P[pos_idx][pos_idx] = p_pp + q * (dt ** 4) / 4
            new_P[pos_idx][vel_idx] = p_pv + q * (dt ** 3) / 2
            new_P[vel_idx][pos_idx] = p_pv + q * (dt ** 3) / 2
            new_P[vel_idx][vel_idx] = p_vv + q * (dt ** 2)
        # Attitude axes: pure random walk -- variance just grows by
        # q_attitude*dt each step (standard discretized random-walk
        # process noise), no position/velocity-style coupling terms.
        for att_idx in (6, 7, 8):
            new_P[att_idx][att_idx] = P[att_idx][att_idx] + self.q_attitude * dt

        self.P = new_P

    def _scalar_update(self, innovation: float, h_row: List[float], r: float) -> None:
        """Shared scalar Kalman update -- see module docstring for why
        every measurement here (including the nonlinear speed one, via
        its linearized Jacobian passed in as h_row) reduces to this."""
        P = self.P
        # Ph = P @ h_row^T  (9-vector)
        ph = [sum(P[i][k] * h_row[k] for k in range(9)) for i in range(9)]
        s = sum(h_row[k] * ph[k] for k in range(9)) + r  # S = h_row @ P @ h_row^T + R
        if s == 0:
            return
        k_gain = [ph[i] / s for i in range(9)]  # K = P@h_row^T / S

        self.x = [self.x[i] + k_gain[i] * innovation for i in range(9)]

        # P = P - K @ (h_row @ P)   (equivalent to (I - K@H)@P for a
        # scalar/rank-1 update)
        h_p = [sum(h_row[k] * P[k][j] for k in range(9)) for j in range(9)]
        self.P = [[P[i][j] - k_gain[i] * h_p[j] for j in range(9)] for i in range(9)]

    def update_position(self, x_east: float, y_north: float, z_alt: float, r: float = 0.09) -> None:
        """Linear measurement -- H picks out [x,y,z] directly. r=0.09
        (0.3m std dev) matches KalmanFilter1D's default barometric
        altitude noise assumption, reused here for horizontal position
        too as a representative default (see kalman_filter.py's module
        docstring for that number's origin)."""
        for idx, z in ((0, x_east), (1, y_north), (2, z_alt)):
            h_row = [1.0 if i == idx else 0.0 for i in range(9)]
            innovation = z - self.x[idx]
            self._scalar_update(innovation, h_row, r)

    def update_attitude(self, roll: float, pitch: float, yaw: float, r: float = 4.0) -> None:
        """Linear measurement -- H picks out [roll,pitch,yaw] directly.
        r=4.0 (2 degree std dev) is a representative typical noise level
        for a consumer/hobby-grade AHRS attitude estimate."""
        for idx, z in ((6, roll), (7, pitch), (8, yaw)):
            h_row = [1.0 if i == idx else 0.0 for i in range(9)]
            innovation = z - self.x[idx]
            self._scalar_update(innovation, h_row, r)

    def update_speed(self, speed_measured: float, r: float = 0.25) -> None:
        """
        Nonlinear measurement -- h(x) = sqrt(vx^2+vy^2+vz^2). Linearized
        via its Jacobian (analytically verified against a finite-
        difference numerical Jacobian in test_ekf.py) and processed as a
        scalar update exactly like the linear channels above -- this is
        the one place this filter is genuinely "extended," not just "a
        bigger linear Kalman filter." r=0.25 (0.5 m/s std dev) is a
        representative typical GPS-derived groundspeed noise level.
        """
        vx, vy, vz = self.x[3], self.x[4], self.x[5]
        speed_pred = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        if speed_pred < 1e-6:
            # Jacobian is undefined at exactly zero velocity (division by
            # zero) -- skip this update rather than fabricate a direction.
            return
        h_row = [0.0, 0.0, 0.0, vx / speed_pred, vy / speed_pred, vz / speed_pred, 0.0, 0.0, 0.0]
        innovation = speed_measured - speed_pred
        self._scalar_update(innovation, h_row, r)

    @property
    def state_dict(self) -> dict:
        # position_uncertainty_m: combined position std dev, sqrt(trace of
        # the 3x3 position sub-block of P) -- the multi-axis generalization
        # of KalmanFilter1D.position_std_dev, so a caller (e.g. a Flutter
        # "confidence" display) has a real, filter-derived uncertainty
        # number instead of nothing to show.
        position_variance = self.P[0][0] + self.P[1][1] + self.P[2][2]
        return {
            "x_east": round(self.x[0], 3), "y_north": round(self.x[1], 3), "z_alt": round(self.x[2], 3),
            "vx": round(self.x[3], 3), "vy": round(self.x[4], 3), "vz": round(self.x[5], 3),
            "roll": round(self.x[6], 3), "pitch": round(self.x[7], 3), "yaw": round(self.x[8], 3),
            "speed_estimate": round(math.sqrt(self.x[3] ** 2 + self.x[4] ** 2 + self.x[5] ** 2), 3),
            "position_uncertainty_m": round(math.sqrt(max(position_variance, 0.0)), 3),
        }


def fuse_full_state(readings: List[dict]) -> List[dict]:
    """
    Runs the 9-state EKF over a chronologically-ordered list of telemetry
    dicts (each with a "timestamp" and optionally "latitude", "longitude",
    "altitude", "speed", "roll", "pitch", "yaw" -- any subset may be None,
    handled gracefully by simply skipping that channel's update for that
    reading, the same graceful-degradation approach used elsewhere in
    this app). `readings` must already be sorted oldest-first.
    """
    if not readings:
        return []

    lat0 = next((r["latitude"] for r in readings if r.get("latitude") is not None), None)
    lon0 = next((r["longitude"] for r in readings if r.get("longitude") is not None), None)

    first = readings[0]
    initial_alt = first.get("altitude") or 0.0
    initial_x, initial_y = 0.0, 0.0
    if lat0 is not None and lon0 is not None and first.get("latitude") is not None:
        initial_x, initial_y = latlon_to_local_meters(first["latitude"], first["longitude"], lat0, lon0)

    ekf = ExtendedKalmanFilter9D(initial_state=[
        initial_x, initial_y, initial_alt, 0.0, 0.0, 0.0,
        first.get("roll") or 0.0, first.get("pitch") or 0.0, first.get("yaw") or 0.0,
    ])

    def _apply_measurements(r):
        if lat0 is not None and r.get("latitude") is not None and r.get("longitude") is not None:
            x_east, y_north = latlon_to_local_meters(r["latitude"], r["longitude"], lat0, lon0)
            z_alt = r.get("altitude")
            if z_alt is not None:
                ekf.update_position(x_east, y_north, z_alt)
        if r.get("roll") is not None and r.get("pitch") is not None and r.get("yaw") is not None:
            ekf.update_attitude(r["roll"], r["pitch"], r["yaw"])
        if r.get("speed") is not None:
            ekf.update_speed(r["speed"])

    _apply_measurements(first)
    results = [{"timestamp": first["timestamp"], **ekf.state_dict}]

    prev_ts = first["timestamp"]
    for r in readings[1:]:
        dt = (r["timestamp"] - prev_ts).total_seconds()
        prev_ts = r["timestamp"]
        if dt <= 0:
            continue  # duplicate/out-of-order timestamp, same handling as smooth_altitude_series
        ekf.predict(dt)
        _apply_measurements(r)
        results.append({"timestamp": r["timestamp"], **ekf.state_dict})

    return results
