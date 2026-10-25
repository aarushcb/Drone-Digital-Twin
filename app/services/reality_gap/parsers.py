"""
D1 spec 5.2 -- format detection and parsing of PX4 ULog / ArduPilot
DataFlash logs into ONE canonical signal schema.

WHY A CANONICAL SCHEMA IS THE POINT OF THIS FILE (spec 5.2, quoting Part E
of the R&D report): "the single most important structural decision is a
rigorous telemetry ingestion and feature store." Every later D-module
(D2 battery prognostics, D3 fault injection) is supposed to reuse this
rather than re-parsing logs itself. So the contract here is deliberately
narrow and explicit: a parser emits {signal_key: Signal} using the exact
keys and SI units in the table below, and nothing else. Missing keys are
absent -- never zero-filled, never imputed (spec 9 case 2).

UNITS ARE SI AND NORMALISED AT THE PARSER BOUNDARY, NOT LATER:
ArduPilot logs attitude in degrees and PX4 in radians; ArduPilot logs
rate setpoints in deg/s and PX4 in rad/s; the two stacks disagree on
actuator output scaling. Converting at the parser means every downstream
consumer sees one convention and cannot silently compare a degree
against a radian. Each conversion is marked at its mapping-table entry.

WHAT IS DELIBERATELY NOT HERE:
- .tlog parsing. Spec 2.1 lists it as v1-required, but this pass was
  scoped to ULog + DataFlash only. detect_format() recognises .tlog and
  parse_log() raises a clear UnsupportedLogFormat naming it as a pass-1
  gap -- it does not silently return an empty result. No endpoint can
  reach this yet (spec 8 endpoints are not built), so nothing user-facing
  is broken by the gap; it is recorded in the pass-1 validation file.
- Parameter allow-list filtering (spec 6). Raw parameters ARE extracted
  here because both formats embed them and re-reading the file later
  would be wasteful, but the curated allow-list diff is a later pass.

VERIFIED, NOT ASSUMED (project rule, CLAUDE.md 7):
- The ULog magic bytes below were confirmed against a real file written
  by pyulog's own writer: the first 8 bytes are
  55 4c 6f 67 01 12 35 01 -- i.e. the 7-byte magic from spec 2.1
  followed by the version byte. Checked, not taken from the spec on
  faith.
- The DataFlash 0xA3 0x95 header pair was confirmed against
  pymavlink.DFReader's own HEAD1/HEAD2 constants rather than from
  memory.
"""

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

ULOG_MAGIC = b"\x55\x4c\x6f\x67\x01\x12\x35"  # "ULog\x01\x12\x35", spec 2.1
DATAFLASH_HEAD = b"\xa3\x95"                   # DFReader.HEAD1/HEAD2

FORMAT_ULOG = "ulog"
FORMAT_DATAFLASH = "dataflash"
FORMAT_TLOG = "tlog"

ACCEPTED_EXTENSIONS = (".ulg", ".ulog", ".bin", ".tlog")


class LogParseError(Exception):
    """Raised when a file cannot be parsed as its detected format.

    Callers (the spec 8 endpoints, in a later pass) are expected to turn
    this into an HTTP 422 carrying the message -- never a 500. See spec 9
    case 7.
    """


class UnsupportedLogFormat(LogParseError):
    """Detected a format this build cannot parse. Spec 9 case 17 -> HTTP 415."""


class EmptyLogError(LogParseError):
    """Parsed successfully but produced no usable telemetry. Spec 9 case 6 -> HTTP 422."""


def detect_format(path: str) -> str:
    """
    Magic bytes FIRST, extension only as a fallback (spec 2.1 [DECISION]:
    "never extension alone"). A .bin that is really a ULog, or a log
    saved with no extension at all, must still be identified correctly --
    users rename these files constantly.

    The extension fallback exists only for .tlog, which is a raw MAVLink
    byte stream with no file-level magic number to key on: its first byte
    is whatever MAVLink start-of-frame marker the stream happens to begin
    with (0xFE for MAVLink 1, 0xFD for MAVLink 2), which is far too weak
    a signal to classify on by itself.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(16)
    except OSError as exc:
        raise LogParseError(f"could not read {os.path.basename(path)}: {exc}") from exc

    if head.startswith(ULOG_MAGIC):
        return FORMAT_ULOG
    if head.startswith(DATAFLASH_HEAD):
        return FORMAT_DATAFLASH

    ext = os.path.splitext(path)[1].lower()
    if ext == ".tlog":
        return FORMAT_TLOG
    if ext in (".ulg", ".ulog"):
        # Extension claims ULog but the magic says otherwise -- that is a
        # corrupt/truncated header, not a different format. Say so
        # precisely rather than falling through to "unsupported".
        raise LogParseError(
            f"{os.path.basename(path)} has a ULog extension but its header is not a ULog "
            f"magic number (got {head[:7].hex()}); the file is probably truncated or corrupt"
        )
    raise UnsupportedLogFormat(
        f"{os.path.basename(path)} is not a recognised flight log. "
        f"Accepted formats: {', '.join(ACCEPTED_EXTENSIONS)}"
    )


# ---------------------------------------------------------------------------
# Canonical schema containers
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    """One canonical signal. `t_us` is monotonically increasing microseconds
    on the log's own boot clock (NOT wall clock -- see LogMeta.start_time_utc
    for that), `values` is always float32 (spec 5.7 memory discipline)."""

    key: str
    t_us: np.ndarray
    values: np.ndarray
    dropped_samples: int = 0  # NaN/Inf removed here -- spec 9 case 12

    @property
    def native_rate_hz(self) -> Optional[float]:
        """Median-interval sample rate. Median rather than mean because a
        single clock jump or logging dropout would drag a mean rate badly
        off, and this number gates real decisions downstream (spec 9 case
        5 excludes <5 Hz signals from spectral metrics)."""
        if self.t_us.size < 2:
            return None
        deltas = np.diff(self.t_us.astype(np.float64))
        deltas = deltas[deltas > 0]
        if deltas.size == 0:
            return None
        median_dt_us = float(np.median(deltas))
        if median_dt_us <= 0:
            return None
        return 1e6 / median_dt_us

    @property
    def duration_s(self) -> float:
        if self.t_us.size < 2:
            return 0.0
        return float(self.t_us[-1] - self.t_us[0]) / 1e6


@dataclass
class LogMeta:
    """Spec 2.2 -- metadata the app extracts rather than asking the user for."""

    source_format: str
    filename: str
    sha256: str
    start_time_utc: Optional[str] = None
    duration_s: float = 0.0
    firmware_version: Optional[str] = None
    airframe: Optional[str] = None
    sample_counts: Dict[str, int] = field(default_factory=dict)
    looks_simulated: Optional[bool] = None  # None = inconclusive, spec 2.2
    sim_evidence: List[str] = field(default_factory=list)
    actuator_pwm_range: Optional[Tuple[float, float]] = None
    # A5: spec 3.7's rotor/noise band gate needs these. sensor_combined accel
    # is low-passed at IMU_ACCEL_CUTOFF (30 Hz on this airframe), so a 20-100 Hz
    # rotor-band ratio read off accel.* is measuring the autopilot's filter, not
    # the airframe. Whoever computes 3.7 has to know where the wall is.
    imu_accel_cutoff: Optional[float] = None
    imu_gyro_cutoff: Optional[float] = None


@dataclass
class ParsedLog:
    signals: Dict[str, Signal]
    params: Dict[str, float]
    meta: LogMeta
    warnings: List[str] = field(default_factory=list)
    # Health spec 1.1: per requested topic, whether it was logged, how many
    # instances, which instance was read and at what rate. A feature that
    # can't be computed returns null with a reason drawn from here, rather
    # than a number from data that was never logged.
    topic_info: Dict[str, dict] = field(default_factory=dict)
    # Health spec A-1: motor count, its source, and how output channels were
    # mapped to motor numbers.
    motor_info: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared cleaning helpers
# ---------------------------------------------------------------------------

CLOCK_JUMP_BACKWARD_US = 1_000_000  # spec 9 case 13: backwards jump > 1 s


def _clean_series(
    t_us: np.ndarray, values: np.ndarray, warnings: List[str], key: str
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Applies spec 9 cases 12 and 13 to one raw series, in that order:

    12. NaN/Inf samples are dropped from THIS signal only (not the whole
        row -- other signals at the same instant are still perfectly
        good), and the count is returned so the caller can warn above the
        5% threshold.
    13. Timestamps are sorted, exact duplicates dropped keeping the first,
        and if the clock jumps BACKWARD by more than a second the log is
        cut there and the longer of the two fragments is kept.

    Order matters: dropping non-finite samples first means a NaN cannot
    masquerade as a clock anomaly during the monotonicity pass.
    """
    if t_us.size == 0:
        return t_us, values, 0

    # t_us is int64 -- integers are always finite, so the old
    # `np.isfinite(t_us.astype(np.float64))` allocated a full-length float64
    # copy of every timestamp to prove something the dtype already guarantees.
    finite = np.isfinite(values)
    dropped = int(finite.size - int(finite.sum()))
    if dropped:
        if dropped > 0.05 * finite.size:
            warnings.append(f"high_dropped_sample_rate:{key}")
        t_us = t_us[finite]
        values = values[finite]
    if t_us.size < 2:
        return t_us, values, dropped

    # Log timestamps are already strictly increasing in the overwhelming
    # majority of cases. Checking first costs one np.diff; sorting
    # unconditionally cost an argsort plus four full-length gather copies
    # every time, for nothing.
    if not bool((np.diff(t_us) > 0).all()):
        order = np.argsort(t_us, kind="stable")
        t_us = t_us[order]
        values = values[order]
        keep = np.ones(t_us.size, dtype=bool)
        keep[1:] = np.diff(t_us) != 0
        t_us = t_us[keep]
        values = values[keep]

    # Post-sort, post-dedupe the series is strictly increasing by
    # construction. The backwards-jump check of case 13 therefore has to
    # happen on the RAW ordering before this function is called -- see
    # _detect_clock_jump, invoked by _store().
    return t_us, values, dropped


def _detect_clock_jump(t_us_raw: np.ndarray) -> Optional[int]:
    """Index of the first backwards jump larger than CLOCK_JUMP_BACKWARD_US
    in the RAW (unsorted) series, or None. Spec 9 case 13."""
    if t_us_raw.size < 2:
        return None
    deltas = np.diff(t_us_raw.astype(np.int64))
    bad = np.nonzero(deltas < -CLOCK_JUMP_BACKWARD_US)[0]
    return int(bad[0]) + 1 if bad.size else None


def _quaternion_to_euler(
    qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Intrinsic Z-Y-X (yaw-pitch-roll) Euler angles in radians, per spec 5.2
    [DECISION] "attitude comes from quaternions, not from any logged Euler
    field".

    The pitch term is clipped to [-1, 1] before asin: with float32
    quaternions that are only normalised to within rounding error, the
    argument can land at 1.0000001 and produce a NaN for an aircraft that
    is merely pointing straight up. Clipping is the standard guard, not a
    fudge -- the true value at that point is exactly +/- pi/2.
    """
    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return roll, pitch, yaw


def unwrap_yaw(yaw_rad: np.ndarray) -> np.ndarray:
    """
    Spec 5.2 / spec 9 case 11. A yaw series crossing +/-pi wraps, and a wrap
    produces a 2*pi jump that shows up in RMSE as an enormous spike that is
    a pure artefact of the angle representation, not a real disagreement
    between two flights. Unwrapping is mandatory before ANY metric.
    """
    if yaw_rad.size < 2:
        return yaw_rad
    return np.unwrap(yaw_rad.astype(np.float64)).astype(np.float32)


def _sha256(path: str) -> str:
    """Spec 9 case 14 needs to detect the same file uploaded twice."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _store(
    signals: Dict[str, Signal],
    warnings: List[str],
    key: str,
    t_us: np.ndarray,
    values: np.ndarray,
) -> None:
    """Clean, cast to float32 (spec 5.7) and record a canonical signal.
    Silently skips a signal that cleans down to nothing -- an absent key
    is the schema's way of saying 'not available', which is exactly right
    for a series that was entirely NaN."""
    if t_us.size == 0 or values.size == 0:
        return
    size = min(t_us.size, values.size)
    t_us = np.asarray(t_us[:size], dtype=np.int64)
    # float32 AT EXTRACTION, not after cleaning. Signal has always stored
    # float32 (spec 5.7); casting up to float64 here and back down at the end
    # doubled every intermediate for no precision gain -- PX4 logs float32.
    values = np.asarray(values[:size], dtype=np.float32)

    jump_idx = _detect_clock_jump(t_us)
    if jump_idx is not None:
        head_len, tail_len = jump_idx, t_us.size - jump_idx
        if head_len >= tail_len:
            t_us, values = t_us[:jump_idx], values[:jump_idx]
        else:
            t_us, values = t_us[jump_idx:], values[jump_idx:]
        if "clock_jump_truncated" not in warnings:
            warnings.append("clock_jump_truncated")

    t_us, values, dropped = _clean_series(t_us, values, warnings, key)
    if t_us.size == 0:
        return
    signals[key] = Signal(key=key, t_us=t_us, values=values, dropped_samples=dropped)


# ---------------------------------------------------------------------------
# PX4 ULog
# ---------------------------------------------------------------------------

# Only these topics are read off disk. Spec 5.7: "parse topic by topic,
# keeping only the signals in 5.2; discard the rest". pyulog accepts this
# list directly and never materialises the other topics at all, which is
# the single biggest memory saving available on a 512 MB instance.
ULOG_TOPICS = [
    "vehicle_attitude",
    "vehicle_angular_velocity",
    "sensor_combined",
    "vehicle_local_position",
    "vehicle_gps_position",
    "actuator_outputs",
    "vehicle_rates_setpoint",
    "vehicle_thrust_setpoint",
    "battery_status",
    "vehicle_imu_status",
    "cpuload",
    "vehicle_status",
    # Health spec 1.2 additions.
    "vehicle_attitude_setpoint",
    "failure_detector_status",
    "estimator_status",
    "estimator_selector_status",
    "actuator_motors",
    "vehicle_land_detected",
]

# (canonical_key, ulog_topic, ulog_field, scale) -- scale applied as
# value * scale. All PX4 core fields are already SI, so scale is 1.0
# except where the spec calls for a sign flip.
ULOG_DIRECT_MAP: List[Tuple[str, str, str, float]] = [
    ("gyro.x", "vehicle_angular_velocity", "xyz[0]", 1.0),
    ("gyro.y", "vehicle_angular_velocity", "xyz[1]", 1.0),
    ("gyro.z", "vehicle_angular_velocity", "xyz[2]", 1.0),
    ("accel.x", "sensor_combined", "accelerometer_m_s2[0]", 1.0),
    ("accel.y", "sensor_combined", "accelerometer_m_s2[1]", 1.0),
    ("accel.z", "sensor_combined", "accelerometer_m_s2[2]", 1.0),
    ("pos.x", "vehicle_local_position", "x", 1.0),
    ("pos.y", "vehicle_local_position", "y", 1.0),
    ("pos.z", "vehicle_local_position", "z", 1.0),
    ("vel.x", "vehicle_local_position", "vx", 1.0),
    ("vel.y", "vehicle_local_position", "vy", 1.0),
    ("vel.z", "vehicle_local_position", "vz", 1.0),
    # NED z is positive-down, so relative altitude is its negation (spec 5.2).
    ("alt.rel", "vehicle_local_position", "z", -1.0),
    ("gps.fix_type", "vehicle_gps_position", "fix_type", 1.0),
    ("gps.nsats", "vehicle_gps_position", "satellites_used", 1.0),
    ("gps.eph", "vehicle_gps_position", "eph", 1.0),
    ("rate_sp.roll", "vehicle_rates_setpoint", "roll", 1.0),
    ("rate_sp.pitch", "vehicle_rates_setpoint", "pitch", 1.0),
    ("rate_sp.yaw", "vehicle_rates_setpoint", "yaw", 1.0),
    # PX4 body-Z thrust setpoint is negative-up; spec 5.2 wants 0..1 positive.
    ("thrust.sp", "vehicle_thrust_setpoint", "xyz[2]", -1.0),
    ("batt.voltage", "battery_status", "voltage_filtered_v", 1.0),
    ("batt.current", "battery_status", "current_filtered_a", 1.0),
    ("cpu.load", "cpuload", "load", 1.0),
    ("mem.free", "cpuload", "ram_usage", 1.0),
    # Health spec 1.2. Raw (unfiltered) battery fields: the resistance fit
    # needs them, because the _filtered fields lag and bias the dV/dI slope.
    ("batt.voltage_raw", "battery_status", "voltage_v", 1.0),
    ("batt.current_raw", "battery_status", "current_a", 1.0),
    ("batt.discharged_mah", "battery_status", "discharged_mah", 1.0),
    ("batt.cell_delta", "battery_status", "max_cell_voltage_delta", 1.0),
    ("batt.cell_count", "battery_status", "cell_count", 1.0),
    ("gps.sacc", "vehicle_gps_position", "s_variance_m_s", 1.0),
    ("gps.noise", "vehicle_gps_position", "noise_per_ms", 1.0),
    ("gps.jamming", "vehicle_gps_position", "jamming_indicator", 1.0),
    ("fd.imbalance", "failure_detector_status", "imbalanced_prop_metric", 1.0),
    ("est.vel_tr", "estimator_status", "vel_test_ratio", 1.0),
    ("est.pos_tr", "estimator_status", "pos_test_ratio", 1.0),
    ("est.hgt_tr", "estimator_status", "hgt_test_ratio", 1.0),
    ("est.mag_tr", "estimator_status", "mag_test_ratio", 1.0),
    ("imu.clip0", "vehicle_imu_status", "accel_clipping[0]", 1.0),
    ("imu.clip1", "vehicle_imu_status", "accel_clipping[1]", 1.0),
    ("imu.clip2", "vehicle_imu_status", "accel_clipping[2]", 1.0),
    ("land.landed", "vehicle_land_detected", "landed", 1.0),
]

# State flags, not telemetry: a log carrying only these still has nothing to
# analyse, so it is still an empty log (spec 9 case 6).
BOOKKEEPING_SIGNALS = {"status.armed", "land.landed"}

# PX4 output functions 101..112 are Motor 1..12 (control allocation, v1.13+).
MOTOR_FUNCTION_BASE = 101
# Output banks whose channel -> function mapping lives in <PREFIX>_FUNC<n>.
OUTPUT_BANK_PREFIXES = ("PWM_MAIN", "PWM_AUX", "PWM_EXTRA", "UAVCAN_EC")

DEFAULT_PWM_MIN = 1000.0
DEFAULT_PWM_MAX = 2000.0

# A5: raw FIFO IMU. Parsed in a SEPARATE ULog() pass, not added to
# ULOG_TOPICS -- a 60 MB profile-529 log carries ~105k FIFO messages per
# sensor (~23 MB of buffer each), and folding them into the main pass adds
# that to its peak instead of overlapping with it.
FIFO_TOPIC_KEYS = {"sensor_accel_fifo": "accel_fifo", "sensor_gyro_fifo": "gyro_fifo"}


def parse_ulog(path: str) -> ParsedLog:
    """
    Parse a PX4 ULog into the canonical schema.

    TRUNCATED FILES ARE THE NORMAL CASE, NOT THE EXCEPTION (spec 9 case 7):
    a log cut when the battery is pulled mid-flight is extremely common.
    pyulog recovers whatever it can and sets `file_corruption`; this
    function therefore attempts the parse, keeps the partial data, records
    a `partial_parse_recovered` warning, and only fails if the recovered
    data is empty.
    """
    from pyulog import ULog

    warnings: List[str] = []
    try:
        ulog = ULog(path, message_name_filter_list=ULOG_TOPICS, disable_str_exceptions=True)
    except Exception as exc:  # noqa: BLE001 -- pyulog raises bare Exception subclasses
        raise LogParseError(
            f"could not parse {os.path.basename(path)} as ULog: {str(exc)[:200]}"
        ) from exc

    if getattr(ulog, "file_corruption", False):
        warnings.append("partial_parse_recovered")

    signals: Dict[str, Signal] = {}

    def topic_time(ds) -> np.ndarray:
        return np.asarray(ds.data["timestamp"], dtype=np.int64)

    # Multi-instance topics (several IMUs, several EKFs, several output banks)
    # appear once per instance in data_list. A plain {name: ds} dict keeps
    # whichever instance comes LAST, which on the reference hex log was the
    # AUX output bank (not the motors), a secondary IMU and a secondary EKF.
    # Pick each one deliberately instead.
    instances: Dict[str, list] = {}
    for d in ulog.data_list:
        instances.setdefault(d.name, []).append(d)
    for lst in instances.values():
        lst.sort(key=lambda d: d.multi_id)
    datasets = {name: lst[0] for name, lst in instances.items()}
    params = {k: float(v) for k, v in ulog.initial_parameters.items() if isinstance(v, (int, float))}

    # EKF: the instance the selector says is primary (the most common value
    # over the log, since it can switch mid-flight).
    sel = datasets.get("estimator_selector_status")
    if sel is not None and "primary_instance" in sel.data and "estimator_status" in instances:
        prim = np.asarray(sel.data["primary_instance"], dtype=np.int64)
        if prim.size:
            vals, counts = np.unique(prim, return_counts=True)
            primary = int(vals[np.argmax(counts)])
            match = [d for d in instances["estimator_status"] if d.multi_id == primary]
            if match:
                datasets["estimator_status"] = match[0]
    # IMU: the vehicle_imu_status instance whose accelerometer the primary
    # EKF is actually using.
    est = datasets.get("estimator_status")
    if est is not None and "accel_device_id" in est.data and "vehicle_imu_status" in instances:
        dev = int(np.asarray(est.data["accel_device_id"])[0])
        match = [d for d in instances["vehicle_imu_status"]
                 if "accel_device_id" in d.data and int(np.asarray(d.data["accel_device_id"])[0]) == dev]
        if match:
            datasets["vehicle_imu_status"] = match[0]

    topic_info: Dict[str, dict] = {}

    # --- attitude: quaternion -> Euler, per spec 5.2 [DECISION] ---
    att = datasets.get("vehicle_attitude")
    if att is not None and all(f"q[{i}]" in att.data for i in range(4)):
        t = topic_time(att)
        qw, qx, qy, qz = (np.asarray(att.data[f"q[{i}]"], dtype=np.float64) for i in range(4))
        roll, pitch, yaw = _quaternion_to_euler(qw, qx, qy, qz)
        _store(signals, warnings, "att.roll", t, roll)
        _store(signals, warnings, "att.pitch", t, pitch)
        _store(signals, warnings, "att.yaw", t, unwrap_yaw(yaw.astype(np.float32)))
        for name, q in zip(("qw", "qx", "qy", "qz"), (qw, qx, qy, qz)):
            _store(signals, warnings, f"att.{name}", t, q)
    # Health spec 1.2: attitude setpoint quaternion, for tracking error.
    att_sp = datasets.get("vehicle_attitude_setpoint")
    if att_sp is not None and all(f"q_d[{i}]" in att_sp.data for i in range(4)):
        t = topic_time(att_sp)
        for i, name in enumerate(("qw", "qx", "qy", "qz")):
            _store(signals, warnings, f"att_sp.{name}", t, np.asarray(att_sp.data[f"q_d[{i}]"], dtype=np.float64))

    # --- everything with a straight field mapping ---
    for key, topic, field_name, scale in ULOG_DIRECT_MAP:
        ds = datasets.get(topic)
        if ds is None or field_name not in ds.data:
            continue
        values = np.asarray(ds.data[field_name], dtype=np.float64) * scale
        _store(signals, warnings, key, topic_time(ds), values)

    # --- vibration: PX4 reports one scalar metric, not per-axis ---
    # PX4's accel_vibration_metric is ONE combined magnitude, unlike
    # ArduPilot's per-axis VIBE.VibeX/Y/Z. It is stored under vibe.x only.
    # Copying the same array into vibe.y and vibe.z would fabricate two
    # signals carrying no independent information and would triple-count
    # this single measurement in the IMU subsystem average (spec 4.2).
    imu_status = datasets.get("vehicle_imu_status")
    if imu_status is not None and "accel_vibration_metric" in imu_status.data:
        _store(
            signals,
            warnings,
            "vibe.x",
            topic_time(imu_status),
            np.asarray(imu_status.data["accel_vibration_metric"], dtype=np.float64),
        )

    # --- actuators (health spec amendment A-1: every motor, not the first four) ---
    act_info = _parse_ulog_actuators(instances, datasets, params, signals, warnings, topic_time)

    # Arming state -> status.armed, so align.find_arm_anchor's arm_event
    # branch is reachable on a real log. PX4 vehicle_status.arming_state:
    # 1 = DISARMED, 2 = ARMED. Emitted as 0.0/1.0 because the canonical
    # schema carries float signals, and find_arm_anchor tests > 0.5.
    status = datasets.get("vehicle_status")
    if status is not None and "arming_state" in status.data:
        armed = (np.asarray(status.data["arming_state"], dtype=np.int32) == 2).astype(np.float64)
        _store(signals, warnings, "status.armed", topic_time(status), armed)

    # Free the main pass's buffers before the FIFO pass allocates its own,
    # so the two never coexist at peak. Only the small info dict survives.
    for topic in ULOG_TOPICS:
        lst = instances.get(topic)
        if not lst:
            topic_info[topic] = {"present": False}
            continue
        chosen = datasets[topic]
        t = np.asarray(chosen.data["timestamp"], dtype=np.int64)
        rate = None
        if t.size >= 2:
            dts = np.diff(t)[np.diff(t) > 0]
            if dts.size:
                rate = round(1e6 / float(np.median(dts)), 2)
        topic_info[topic] = {
            "present": True,
            "instances": len(lst),
            "selected_instance": int(chosen.multi_id),
            "rate_hz": rate,
            "samples": int(t.size),
        }
    info = dict(getattr(ulog, "msg_info_dict", {}) or {})
    del datasets, instances, ulog
    _parse_fifo_topics(path, signals, warnings)

    for fifo_topic, prefix in FIFO_TOPIC_KEYS.items():
        sig = signals.get(f"{prefix}.x")
        topic_info[fifo_topic] = (
            {"present": True, "rate_hz": round(sig.native_rate_hz, 2) if sig.native_rate_hz else None}
            if sig is not None else {"present": False}
        )

    pwm_range = act_info.pop("pwm_range", None)
    meta = _ulog_meta(path, info, signals, params, pwm_range)
    if not (set(signals) - BOOKKEEPING_SIGNALS):
        raise EmptyLogError(f"{os.path.basename(path)} contains no usable telemetry")
    return ParsedLog(signals=signals, params=params, meta=meta, warnings=warnings,
                     topic_info=topic_info, motor_info=act_info)


def _parse_fifo_topics(path: str, signals: Dict[str, Signal], warnings: List[str]) -> None:
    """
    A5: expand sensor_accel_fifo / sensor_gyro_fifo into canonical
    accel_fifo.* / gyro_fifo.* signals.

    WHY THESE TOPICS ARE NOT ONE-SAMPLE-PER-MESSAGE (the thing that makes
    this different from every other topic here): a FIFO record carries a
    BATCH -- fixed-width x[0..31]/y[..]/z[..] arrays, a `samples` count
    saying how many of those 32 slots are actually valid, and a `dt`
    giving the microseconds between consecutive samples WITHIN the batch.
    On the reference log that is 104,830 messages x 10 valid samples at
    dt=250 us, i.e. ~1.05 M samples at 4 kHz -- not the 210 Hz you would
    get by treating one message as one sample.

    The raw x/y/z are int16 ADC counts; `scale` converts them to SI
    (m/s^2 for accel, rad/s for gyro) and is per-message, so it is applied
    per-row rather than as a constant.

    Sample timestamps are reconstructed as timestamp_sample + i*dt. PX4
    documents timestamp_sample as the time of the FIRST sample in the
    batch; if that convention is ever the last sample instead, every
    reconstructed time shifts by at most samples*dt = 2.5 ms, which is an
    eighth of one 50 Hz grid cell and cannot move an alignment decision.
    Noted rather than guessed at silently.

    This runs its own ULog() pass and drops it before returning, so the
    FIFO buffers never coexist with the main parse's.
    """
    from pyulog import ULog

    # Deferred: align imports this module, so a top-level import here
    # would be circular. The 500 Hz cap lives in align because that is
    # where spec 5.4 puts it; duplicating the float would risk drift.
    from .align import HIGH_RATE_CAP_HZ

    try:
        ulog = ULog(path, message_name_filter_list=list(FIFO_TOPIC_KEYS),
                    disable_str_exceptions=True)
    except Exception:  # noqa: BLE001 -- absent/!unreadable FIFO is not fatal
        return

    for ds in ulog.data_list:
        prefix = FIFO_TOPIC_KEYS.get(ds.name)
        if prefix is None or "samples" not in ds.data:
            continue
        d = ds.data
        counts = np.asarray(d["samples"], dtype=np.int32)
        width = max((int(k[2:-1]) for k in d if k.startswith("x[")), default=-1) + 1
        if width <= 0 or counts.size == 0:
            continue
        keep = int(counts.max())
        if keep <= 0:
            continue
        keep = min(keep, width)

        # (n_messages, keep) validity mask -- honours a per-message `samples`
        # that is smaller than the array width, so padding slots never become
        # fabricated readings.
        valid = np.arange(keep, dtype=np.int32)[None, :] < counts[:, None]

        # One timestamp grid shared by all three axes: the same ndarray object
        # is handed to each Signal rather than copied three times.
        base = np.asarray(d["timestamp_sample"] if "timestamp_sample" in d else d["timestamp"],
                          dtype=np.int64)
        dt_us = np.asarray(d["dt"], dtype=np.float64)
        # int64 throughout: the float64 route allocated a (n, keep) float64
        # AND its int64 cast AND the masked copy -- three full-size arrays for
        # a quantity that is integer microseconds by construction.
        dt_int = np.rint(dt_us).astype(np.int64)
        grid = base[:, None] + np.arange(keep, dtype=np.int64)[None, :] * dt_int[:, None]
        t_us = grid[valid]
        del grid

        # Decimate to the spec 5.4 high-rate cap. The FIFO runs at ~4 kHz;
        # 5.4's high-rate spectral grid is min(native_real, native_sim)
        # CAPPED AT 500 Hz, so everything above 500 Hz is discarded
        # downstream regardless -- keeping it costs 8x the memory for data
        # nothing reads. 500 Hz still leaves a 250 Hz Nyquist, covering
        # 3.7's 20-100 Hz rotor band and most of the 100 Hz+ noise floor,
        # and this is still PRE-IMU_ACCEL_CUTOFF data, which is the whole
        # reason A5 wants the topic.
        # ponytail: fixed decimation to 500 Hz; if a later pass needs the
        # full 4 kHz band, take the factor from the caller instead.
        native_hz = 1e6 / float(np.median(dt_us)) if np.median(dt_us) > 0 else 0.0
        factor = int(native_hz // HIGH_RATE_CAP_HZ) if native_hz > HIGH_RATE_CAP_HZ else 1

        scale = np.asarray(d["scale"], dtype=np.float32)
        for axis in ("x", "y", "z"):
            cols = [d.get(f"{axis}[{i}]") for i in range(keep)]
            if any(c is None for c in cols):
                continue
            block = np.stack(cols, axis=1).astype(np.float32)
            block *= scale[:, None]
            key = f"{prefix}.{axis}"
            values = block[valid]
            del block
            if factor > 1 and values.size > 27 * factor:
                # zero_phase keeps the decimated series time-aligned with
                # t_us[::factor]; a causal filter would introduce a group
                # delay that alignment would later read as a real lag.
                values = sp_signal.decimate(
                    values, factor, ftype="fir", zero_phase=True).astype(np.float32)
            axis_t = t_us[::factor][: values.size] if factor > 1 else t_us
            if values.size:
                # Deliberately bypasses _store: it upcasts to float64 and holds
                # several full-length intermediates, which on a 1.05 M-sample
                # signal is tens of MB of transient per axis. FIFO timestamps
                # are monotonic by construction (base + i*dt), so the sort and
                # dedupe _store exists for have nothing to do here.
                finite = np.isfinite(values)
                signals[key] = Signal(
                    key=key,
                    t_us=axis_t if finite.all() else axis_t[finite],
                    values=values if finite.all() else values[finite],
                    dropped_samples=int(values.size - int(finite.sum())),
                )
        if prefix not in "".join(warnings):
            warnings.append(f"fifo_expanded:{ds.name}")


def _parse_ulog_actuators(instances, datasets, params, signals, warnings, topic_time) -> Dict[str, object]:
    """
    Every motor, by MOTOR number rather than output channel (health spec A-1).

    Two sources, both recorded:

    * act.motor{i} from actuator_motors.control[i]: the control allocator's
      per-motor command, already in motor order and normalised 0..1 (PX4
      v1.13+). This is what the health features use.
    * act.out{i} from actuator_outputs, mapped channel -> motor through the
      <BANK>_FUNC<n> parameters (101 = Motor 1) and normalised with that
      channel's own <BANK>_MIN<n>/<BANK>_MAX<n>. On the reference hex the
      limits are 1155-1900 us, so the old fixed 1000-2000 normalisation was
      wrong, and the motors live on instance 0 while instance 1 is the AUX
      bank, so the old last-instance read was wrong too.

    Logs from firmware without output functions fall back to the old
    behaviour (instance 0, channels in order, bank-level or default limits),
    flagged as an assumption.
    """
    info: Dict[str, object] = {}
    act_lst = instances.get("actuator_outputs") or []
    act0 = act_lst[0] if act_lst else None
    n_motors, count_source = (_actual_actuator_count_ulog(act0.data, params) if act0 is not None else (0, "none"))
    info["motor_count"] = n_motors
    info["motor_count_source"] = count_source

    motors = datasets.get("actuator_motors")
    if motors is not None and n_motors:
        t = topic_time(motors)
        for i in range(n_motors):
            fname = f"control[{i}]"
            if fname in motors.data:
                _store(signals, warnings, f"act.motor{i}", t, np.asarray(motors.data[fname], dtype=np.float64))

    # Channel -> motor mapping from output functions.
    mapping = {}  # motor index -> (bank prefix, 1-based channel)
    for prefix in OUTPUT_BANK_PREFIXES:
        for ch in range(1, 17):
            fn = params.get(f"{prefix}_FUNC{ch}")
            if fn is not None and MOTOR_FUNCTION_BASE <= int(fn) < MOTOR_FUNCTION_BASE + 12:
                mapping.setdefault(int(fn) - MOTOR_FUNCTION_BASE, (prefix, ch))

    def channel_limits(prefix: str, ch: int) -> Tuple[float, float]:
        lo = params.get(f"{prefix}_MIN{ch}", params.get(f"{prefix}_MIN", params.get("PWM_MIN", DEFAULT_PWM_MIN)))
        hi = params.get(f"{prefix}_MAX{ch}", params.get(f"{prefix}_MAX", params.get("PWM_MAX", DEFAULT_PWM_MAX)))
        return float(lo), float(hi)

    if act_lst and mapping and n_motors:
        # Which logged instance carries the motors: the one where the most
        # motor-mapped channels actually move.
        def moving(ds) -> int:
            hits = 0
            for _, ch in mapping.values():
                f = f"output[{ch - 1}]"
                if f in ds.data and np.ptp(np.asarray(ds.data[f], dtype=np.float64)) > 0:
                    hits += 1
            return hits
        chosen = max(act_lst, key=lambda d: (moving(d), -d.multi_id))
        t = topic_time(chosen)
        ranges = []
        for i in range(n_motors):
            if i not in mapping:
                continue
            prefix, ch = mapping[i]
            f = f"output[{ch - 1}]"
            if f not in chosen.data:
                continue
            lo, hi = channel_limits(prefix, ch)
            ranges.append((lo, hi))
            _store(signals, warnings, f"act.out{i}", t, _normalise_actuator(np.asarray(chosen.data[f], dtype=np.float64), lo, hi))
        info["mapping_source"] = "output_functions"
        info["actuator_outputs_instance"] = int(chosen.multi_id)
        info["channel_of_motor"] = {i: f"{p}{c}" for i, (p, c) in sorted(mapping.items()) if i < n_motors}
        if ranges:
            info["pwm_range"] = (min(r[0] for r in ranges), max(r[1] for r in ranges))
    elif act0 is not None and n_motors:
        lo, hi = channel_limits("PWM_MAIN", 0)
        t = topic_time(act0)
        for i in range(n_motors):
            f = f"output[{i}]"
            if f in act0.data:
                _store(signals, warnings, f"act.out{i}", t, _normalise_actuator(np.asarray(act0.data[f], dtype=np.float64), lo, hi))
        info["mapping_source"] = "channel_order_assumed"
        info["actuator_outputs_instance"] = int(act0.multi_id)
        info["pwm_range"] = (lo, hi)
        warnings.append("actuator_mapping_assumed:no output function parameters; channel i taken as motor i+1")
    return info


def _normalise_actuator(raw: np.ndarray, pwm_min: float, pwm_max: float) -> np.ndarray:
    """
    Spec 5.2 [DECISION]: normalise every actuator signal to [0,1] before
    comparison, using the log's own PWM_MIN/PWM_MAX when present.

    Values already inside [0,1] are passed through untouched -- PX4's newer
    normalised outputs are already in that form, and rescaling them by
    1000/2000 would map a perfectly good 0.5 to a nonsensical -0.5. The
    check is on the observed data range rather than on a firmware version
    string because the latter is not reliably present.
    """
    finite = raw[np.isfinite(raw)]
    if finite.size and finite.min() >= -0.01 and finite.max() <= 1.01:
        return raw
    span = pwm_max - pwm_min
    if span <= 0:
        return raw
    return (raw - pwm_min) / span


def _actual_actuator_count_ulog(act_data: dict, params: dict) -> Tuple[int, str]:
    """
    Best-available REAL actuator/motor count for this log, read from the
    vehicle's own config rather than assumed. Returns (count, source).

    PX4's actuator_outputs message is a FIXED-WIDTH output[0..15] struct
    regardless of airframe -- verified against the real reference fixture
    (px4_641d242a.ulg): all 16 output[] fields are present even though
    only 4 motors are driven, with outputs 4-15 sitting at exactly
    min==max==0.0 (struct padding, not a logged value -- an idle-but-
    connected motor reports a real PWM, never exactly 0.0). Field
    PRESENCE is therefore useless as a motor-count signal for ULog.

    CA_ROTOR_COUNT -- PX4's control-allocation rotor count parameter,
    present on firmware built with the new control allocation module
    (confirmed present and correct, value 4, on the same reference log)
    -- is the authoritative source when available.

    Fallback for older firmware without CA_ROTOR_COUNT: count outputs
    that ever take a nonzero value, using the padding-vs-real distinction
    above. This is a heuristic (an output idling exactly at 0 forever
    would misclassify as unused), so it is reported as such via the
    `source` string rather than silently presented with the same
    confidence as a parameter read.
    """
    rotor_count = params.get("CA_ROTOR_COUNT")
    if rotor_count is not None and rotor_count > 0:
        return int(rotor_count), "CA_ROTOR_COUNT"

    indices = sorted(
        int(k[len("output["):-1]) for k in act_data
        if k.startswith("output[") and k.endswith("]")
    )
    active = sum(1 for i in indices if np.any(np.asarray(act_data[f"output[{i}]"]) != 0))
    return active, "nonzero_output_heuristic"


def _ulog_meta(path, info, signals, params, pwm_range) -> LogMeta:
    start_utc = None
    # ULog's own start timestamp is a boot-relative microsecond counter, so
    # a UTC wall-clock time is only available if the log carries one -- do
    # not invent one from the file mtime, which is when it was COPIED, not
    # when it was flown.
    if "time_ref_utc" in info:
        try:
            from datetime import datetime, timezone

            start_utc = datetime.fromtimestamp(
                int(info["time_ref_utc"]), tz=timezone.utc
            ).isoformat()
        except (ValueError, OSError, OverflowError):
            start_utc = None

    sim_evidence: List[str] = []
    # PX4 SITL evidence, spec 2.2. SYS_AUTOSTART in the 10000-range is the
    # conventional simulation airframe block; HIL_ topics and a set
    # SYS_HITL are the other two hints the spec names.
    if params.get("SYS_HITL", 0):
        sim_evidence.append("SYS_HITL set")
    autostart = params.get("SYS_AUTOSTART")
    if autostart is not None and 10000 <= autostart < 20000:
        sim_evidence.append(f"SYS_AUTOSTART={int(autostart)} (simulation airframe range)")
    if any(name.startswith("SIM_") for name in params):
        sim_evidence.append("SIM_* parameters present")
    sys_name = str(info.get("sys_name", ""))
    if "SITL" in sys_name.upper():
        sim_evidence.append(f"sys_name={sys_name}")

    duration = max((s.duration_s for s in signals.values()), default=0.0)
    return LogMeta(
        source_format=FORMAT_ULOG,
        filename=os.path.basename(path),
        sha256=_sha256(path),
        start_time_utc=start_utc,
        duration_s=duration,
        firmware_version=str(info.get("ver_sw")) if "ver_sw" in info else None,
        airframe=str(int(autostart)) if autostart is not None else None,
        sample_counts={k: int(v.t_us.size) for k, v in signals.items()},
        looks_simulated=True if sim_evidence else None,
        sim_evidence=sim_evidence,
        actuator_pwm_range=pwm_range,
        imu_accel_cutoff=params.get("IMU_ACCEL_CUTOFF"),
        imu_gyro_cutoff=params.get("IMU_GYRO_CUTOFF"),
    )


# ---------------------------------------------------------------------------
# ArduPilot DataFlash
# ---------------------------------------------------------------------------

_DEG2RAD = math.pi / 180.0

# (canonical_key, DF message type, DF field, scale)
DATAFLASH_MAP: List[Tuple[str, str, str, float]] = [
    # ArduPilot logs attitude in DEGREES; the canonical schema is radians.
    ("att.roll", "ATT", "Roll", _DEG2RAD),
    ("att.pitch", "ATT", "Pitch", _DEG2RAD),
    ("att.yaw", "ATT", "Yaw", _DEG2RAD),
    ("gyro.x", "IMU", "GyrX", 1.0),  # already rad/s
    ("gyro.y", "IMU", "GyrY", 1.0),
    ("gyro.z", "IMU", "GyrZ", 1.0),
    ("accel.x", "IMU", "AccX", 1.0),  # already m/s^2
    ("accel.y", "IMU", "AccY", 1.0),
    ("accel.z", "IMU", "AccZ", 1.0),
    ("alt.rel", "CTUN", "Alt", 1.0),
    ("gps.fix_type", "GPS", "Status", 1.0),
    ("gps.nsats", "GPS", "NSats", 1.0),
    # NOTE: GPS.HDop is deliberately NOT mapped to gps.eph. Spec 5.2
    # [DECISION]: eph is a metre estimate and HDop is dimensionless, so
    # they are not the same quantity and multiplying HDop by an assumed
    # UERE to fake metres is exactly the fabricated precision this project
    # forbids. The comparison layer excludes the pair as incommensurable;
    # storing HDop under its own key keeps it available without pretending.
    ("gps.hdop", "GPS", "HDop", 1.0),
    # ArduPilot RATE desired rates are in deg/s; canonical schema is rad/s.
    ("rate_sp.roll", "RATE", "RDes", _DEG2RAD),
    ("rate_sp.pitch", "RATE", "PDes", _DEG2RAD),
    ("rate_sp.yaw", "RATE", "YDes", _DEG2RAD),
    ("thrust.sp", "RATE", "AOut", 1.0),
    ("batt.voltage", "BAT", "Volt", 1.0),
    ("batt.current", "BAT", "Curr", 1.0),
    ("vibe.x", "VIBE", "VibeX", 1.0),
    ("vibe.y", "VIBE", "VibeY", 1.0),
    ("vibe.z", "VIBE", "VibeZ", 1.0),
    ("cpu.load", "PM", "Load", 1.0 / 1000.0),  # PM.Load is per-mille
    ("mem.free", "PM", "MemFree", 1.0),
    # Health spec 1.2. Field names are from ArduPilot's log-message docs and
    # are NOT yet verified against a real .bin (none is available; see
    # BACKLOG.md). BAT.Volt is ArduPilot's unfiltered pack voltage; CurrTot
    # is mAh drawn. VIBE.Clip0-2 are cumulative clipping counters.
    # ArduPilot has no cell-count parameter, so per-cell power features need
    # the drone's stored battery_cells later; until then they return
    # cell_count_unknown rather than guess.
    ("batt.voltage_raw", "BAT", "Volt", 1.0),
    ("batt.current_raw", "BAT", "Curr", 1.0),
    ("batt.discharged_mah", "BAT", "CurrTot", 1.0),
    ("imu.clip0", "VIBE", "Clip0", 1.0),
    ("imu.clip1", "VIBE", "Clip1", 1.0),
    ("imu.clip2", "VIBE", "Clip2", 1.0),
    ("gps.sacc", "GPA", "SAcc", 1.0),
]

# Position/velocity come from the EKF messages, which differ by EKF
# version; try them in order of preference and take the first present.
DATAFLASH_NED_CANDIDATES = [
    ("XKF1", {"pos.x": "PN", "pos.y": "PE", "pos.z": "PD", "vel.x": "VN", "vel.y": "VE", "vel.z": "VD"}),
    ("NKF1", {"pos.x": "PN", "pos.y": "PE", "pos.z": "PD", "vel.x": "VN", "vel.y": "VE", "vel.z": "VD"}),
]

DATAFLASH_TYPES = sorted(
    {m[1] for m in DATAFLASH_MAP} | {c[0] for c in DATAFLASH_NED_CANDIDATES} | {"RCOU", "PARM", "MSG"}
)


def parse_dataflash(path: str) -> ParsedLog:
    """
    Parse an ArduPilot DataFlash (.bin) log into the canonical schema.

    TIMEBASE: `TimeUS` (microseconds since boot) is used wherever the
    message carries it, deliberately in preference to DFReader's derived
    `_timestamp` wall-clock. TimeUS is the same kind of quantity as ULog's
    `timestamp` -- a monotonic boot-relative counter -- so the two formats
    end up on directly comparable clocks, which is what makes the
    cross-format comparison in spec 10.2 item 12 meaningful. A wall-clock
    timebase would also silently depend on whether the vehicle ever got a
    GPS time fix.
    """
    from pymavlink import DFReader

    warnings: List[str] = []
    try:
        reader = DFReader.DFReader_binary(path)
    except Exception as exc:  # noqa: BLE001 -- pymavlink raises bare Exception
        raise LogParseError(
            f"could not parse {os.path.basename(path)} as DataFlash: {str(exc)[:200]}"
        ) from exc

    collected: Dict[str, Dict[str, List[float]]] = {}
    params: Dict[str, float] = {}
    messages: List[str] = []

    while True:
        try:
            msg = reader.recv_match(type=DATAFLASH_TYPES)
        except Exception:  # noqa: BLE001 -- a truncated tail is normal (spec 9 case 7)
            warnings.append("partial_parse_recovered")
            break
        if msg is None:
            break
        mtype = msg.get_type()
        if mtype == "PARM":
            try:
                params[str(msg.Name)] = float(msg.Value)
            except (AttributeError, TypeError, ValueError):
                pass
            continue
        if mtype == "MSG":
            messages.append(str(getattr(msg, "Message", "")))
            continue

        bucket = collected.setdefault(mtype, {})
        time_us = getattr(msg, "TimeUS", None)
        if time_us is None:
            ts = getattr(msg, "_timestamp", None)
            if ts is None:
                continue
            time_us = ts * 1e6
        bucket.setdefault("_t", []).append(float(time_us))
        for fname in msg.get_fieldnames():
            if fname == "TimeUS":
                continue
            value = getattr(msg, fname, None)
            if isinstance(value, (int, float)):
                bucket.setdefault(fname, []).append(float(value))

    signals: Dict[str, Signal] = {}

    def times_for(mtype: str) -> Optional[np.ndarray]:
        bucket = collected.get(mtype)
        if not bucket or "_t" not in bucket:
            return None
        return np.asarray(bucket["_t"], dtype=np.int64)

    for key, mtype, fname, scale in DATAFLASH_MAP:
        bucket = collected.get(mtype)
        if not bucket or fname not in bucket:
            continue
        t = times_for(mtype)
        values = np.asarray(bucket[fname], dtype=np.float64) * scale
        if key == "att.yaw":
            values = unwrap_yaw(values.astype(np.float32)).astype(np.float64)
        _store(signals, warnings, key, t, values)

    for mtype, mapping in DATAFLASH_NED_CANDIDATES:
        bucket = collected.get(mtype)
        if not bucket:
            continue
        t = times_for(mtype)
        for key, fname in mapping.items():
            if key in signals or fname not in bucket:
                continue
            _store(signals, warnings, key, t, np.asarray(bucket[fname], dtype=np.float64))
        break

    # Health spec A-1: every motor, by motor number. ArduPilot maps channels to
    # functions with SERVO<n>_FUNCTION (33..40 = Motor 1..8, 82..85 = Motor
    # 9..12) and limits each channel with SERVO<n>_MIN/MAX.
    pwm_min = float(params.get("RC1_MIN", DEFAULT_PWM_MIN))
    pwm_max = float(params.get("RC1_MAX", DEFAULT_PWM_MAX))
    motor_info: Dict[str, object] = {}
    rcou = collected.get("RCOU")
    if rcou:
        t = times_for("RCOU")
        mapping = {}
        for ch in range(1, 17):
            fn = params.get(f"SERVO{ch}_FUNCTION")
            if fn is None:
                continue
            fn = int(fn)
            if 33 <= fn <= 40:
                mapping[fn - 33] = ch
            elif 82 <= fn <= 85:
                mapping[fn - 82 + 8] = ch
        ranges = []
        if mapping:
            for motor, ch in sorted(mapping.items()):
                fname = f"C{ch}"
                if fname not in rcou:
                    continue
                lo = float(params.get(f"SERVO{ch}_MIN", pwm_min))
                hi = float(params.get(f"SERVO{ch}_MAX", pwm_max))
                ranges.append((lo, hi))
                _store(signals, warnings, f"act.out{motor}", t,
                       _normalise_actuator(np.asarray(rcou[fname], dtype=np.float64), lo, hi))
            motor_info = {"motor_count": len(mapping), "motor_count_source": "SERVOn_FUNCTION",
                          "mapping_source": "output_functions",
                          "channel_of_motor": {m: f"SERVO{c}" for m, c in sorted(mapping.items())}}
        else:
            # Unverified layout: count channels that ever move, take them in order.
            rcou_indices = sorted(int(k[1:]) for k in rcou if k != "_t" and k.startswith("C") and k[1:].isdigit())
            active = [i for i in rcou_indices if np.any(np.asarray(rcou[f"C{i}"]) != 0)]
            for motor, ch in enumerate(active):
                _store(signals, warnings, f"act.out{motor}", t,
                       _normalise_actuator(np.asarray(rcou[f"C{ch}"], dtype=np.float64), pwm_min, pwm_max))
            ranges.append((pwm_min, pwm_max))
            motor_info = {"motor_count": len(active), "motor_count_source": "nonzero_output_heuristic",
                          "mapping_source": "channel_order_assumed"}
            warnings.append("actuator_mapping_assumed:no SERVOn_FUNCTION parameters; active channels taken in order")
        if ranges:
            pwm_min, pwm_max = min(r[0] for r in ranges), max(r[1] for r in ranges)

    if not signals:
        raise EmptyLogError(f"{os.path.basename(path)} contains no usable telemetry")

    sim_evidence: List[str] = []
    if any(name.startswith("SIM_") for name in params):
        sim_evidence.append("SIM_* parameters present")
    if any("SITL" in m.upper() for m in messages):
        sim_evidence.append("SITL banner in MSG stream")

    duration = max((s.duration_s for s in signals.values()), default=0.0)
    meta = LogMeta(
        source_format=FORMAT_DATAFLASH,
        filename=os.path.basename(path),
        sha256=_sha256(path),
        duration_s=duration,
        firmware_version=next((m for m in messages if "ArduPilot" in m or "APM" in m), None),
        airframe=str(int(params["FRAME_CLASS"])) if "FRAME_CLASS" in params else None,
        sample_counts={k: int(v.t_us.size) for k, v in signals.items()},
        looks_simulated=True if sim_evidence else None,
        sim_evidence=sim_evidence,
        actuator_pwm_range=(pwm_min, pwm_max),
    )
    return ParsedLog(signals=signals, params=params, meta=meta, warnings=warnings, motor_info=motor_info)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def parse_log(path: str) -> ParsedLog:
    """Detect the format and parse. Raises LogParseError subclasses only --
    never an unhandled exception, per spec 9 case 7."""
    fmt = detect_format(path)
    if fmt == FORMAT_ULOG:
        return parse_ulog(path)
    if fmt == FORMAT_DATAFLASH:
        return parse_dataflash(path)
    if fmt == FORMAT_TLOG:
        raise UnsupportedLogFormat(
            ".tlog parsing is not implemented in this build (D1 pass 1 covers ULog and "
            "DataFlash only). Spec 2.1 lists .tlog as v1-required; it is a tracked gap, "
            "not a silent omission."
        )
    raise UnsupportedLogFormat(f"unhandled format: {fmt}")
