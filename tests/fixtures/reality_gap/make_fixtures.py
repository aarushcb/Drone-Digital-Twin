"""
Deterministic synthetic fixture generator for D1 (spec 10.1).

Run:  python3 tests/fixtures/reality_gap/make_fixtures.py

WHY GENERATE RATHER THAN COMMIT OPAQUE BINARIES:
The fixtures are real, parseable ULog and DataFlash files, but every
value in them is a closed-form function of sample index with KNOWN
ground truth -- a known time shift, a known variance ratio, a known yaw
sweep through +/-pi. A committed .ulg blob would be unreviewable and its
ground truth would live only in a comment. This script IS the ground
truth, and the generated files are committed alongside it so tests do
not depend on regeneration.

DETERMINISM (spec 10.3 item 16): the only randomness is numpy's
default_rng(FIXTURE_SEED) with a fixed seed, so regenerating produces
byte-identical files.

WHY THE ULOG WRITER USES pyulog's OWN write_ulog:
Hand-rolling a ULog binary encoder would mean writing a second ULog
implementation whose bugs could cancel out the parser's -- a test that
proves nothing. Round-tripping through pyulog's writer means the parser
under test is reading a file produced by the reference library, which is
the closest thing to ground truth available without shipping a
third-party log (which spec 5.7/Part G warns about on licence grounds).
"""

import os
import struct
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURE_SEED = 20260905

# --- Ground truth constants the tests assert against -----------------------
DURATION_S = 60.0
ATT_RATE_HZ = 50.0
IMU_RATE_HZ = 100.0
GPS_RATE_HZ = 2.0            # deliberately < LOW_RATE_HZ(5) to exercise spec 9 case 5
SIM_TIME_SHIFT_S = 1.40      # known offset the fine aligner must recover
SIM_GYRO_VARIANCE_SCALE = 0.5  # sim is deliberately "too smooth" (spec 3.2)
BOOT_OFFSET_REAL_US = 12_000_000   # the two logs start at different boot times
BOOT_OFFSET_SIM_US = 3_500_000
TAKEOFF_S = 5.0              # accel.z departs 1 g here -> takeoff anchor
LANDING_S = 55.0


def _t(rate_hz, duration_s=DURATION_S):
    return np.arange(0.0, duration_s, 1.0 / rate_hz)


def _mission(t):
    """A deterministic, closed-form 'square circuit' attitude/position
    profile. Multi-frequency so the 0.2-5 Hz band-pass in the fine
    aligner has real structure to lock onto -- a pure sine would
    correlate at many lags and make the alignment test meaningless."""
    roll = 0.30 * np.sin(2 * np.pi * 0.11 * t) + 0.10 * np.sin(2 * np.pi * 0.43 * t)
    pitch = 0.22 * np.sin(2 * np.pi * 0.17 * t + 0.6) + 0.07 * np.sin(2 * np.pi * 0.71 * t)
    # Yaw deliberately sweeps past +/-pi to exercise unwrap (spec 9 case 11).
    yaw = np.pi * np.sin(2 * np.pi * 0.05 * t) * 1.4
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))  # wrap into [-pi, pi] like a real log
    return roll, pitch, yaw


def _altitude(t):
    """Climb to 30 m, hold, descend -- gives classify_phases() all three
    phases and airborne_window() a real >0.5 m region."""
    alt = np.zeros_like(t)
    climb = (t >= TAKEOFF_S) & (t < 15.0)
    hold = (t >= 15.0) & (t < 45.0)
    descend = (t >= 45.0) & (t < LANDING_S)
    alt[climb] = 30.0 * (t[climb] - TAKEOFF_S) / 10.0
    alt[hold] = 30.0
    alt[descend] = 30.0 * (1 - (t[descend] - 45.0) / 10.0)
    return alt


def _euler_to_quat(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


# ---------------------------------------------------------------------------
# ULog writer
# ---------------------------------------------------------------------------


def _message_format(name, fields):
    from pyulog import ULog

    spec = name + ":" + "".join(f"{ftype} {fname};" for ftype, fname in fields)
    return ULog.MessageFormat(bytes(spec, "utf-8"), None)


def _dataset(msg_id, name, fields, data):
    from pyulog import ULog

    ds = ULog.Data.__new__(ULog.Data)
    ds.multi_id = 0
    ds.msg_id = msg_id
    ds.name = name
    ds.timestamp_idx = 0
    ds.field_data = [
        types.SimpleNamespace(field_name=fname, type_str=ftype) for ftype, fname in fields
    ]
    ds.data = data
    return ds


def write_ulog(path, *, shift_s=0.0, boot_offset_us=0, gyro_scale=1.0, seed=FIXTURE_SEED,
               include_topics=None, params=None, info=None):
    """
    Build a valid PX4 ULog. `shift_s` delays the mission content within
    the recording (the thing the fine aligner must recover); `boot_offset_us`
    changes the absolute boot-clock start (the thing the coarse anchor
    must absorb).
    """
    from pyulog import ULog

    rng = np.random.default_rng(seed)
    include_topics = include_topics or {
        "vehicle_attitude", "vehicle_angular_velocity", "sensor_combined",
        "vehicle_local_position", "vehicle_gps_position", "actuator_outputs",
        "battery_status", "cpuload",
    }

    ulog = ULog(None)
    ulog._file_version = 1
    ulog._start_timestamp = boot_offset_us
    ulog._msg_info_dict = dict(info or {"ver_sw": "v1.14.0-synthetic", "sys_name": "PX4"})
    ulog._msg_info_dict_types = {k: "char[%d]" % len(str(v)) for k, v in ulog._msg_info_dict.items()}
    ulog._initial_parameters = dict(params or {"SYS_AUTOSTART": 4001.0, "PWM_MAIN_MIN": 1000.0,
                                               "PWM_MAIN_MAX": 2000.0})

    datasets = []
    msg_id = 0

    def add(name, fields, data):
        nonlocal msg_id
        if name not in include_topics:
            return
        ulog._message_formats[name] = _message_format(name, fields)
        datasets.append(_dataset(msg_id, name, fields, data))
        msg_id += 1

    def stamps(t):
        return ((t + shift_s) * 1e6 + boot_offset_us).astype(np.uint64)

    # --- attitude (quaternion, per spec 5.2) ---
    t_att = _t(ATT_RATE_HZ)
    roll, pitch, yaw = _mission(t_att)
    qw, qx, qy, qz = _euler_to_quat(roll, pitch, yaw)
    add("vehicle_attitude",
        [("uint64_t", "timestamp"), ("float", "q[0]"), ("float", "q[1]"),
         ("float", "q[2]"), ("float", "q[3]")],
        {"timestamp": stamps(t_att), "q[0]": qw.astype(np.float32), "q[1]": qx.astype(np.float32),
         "q[2]": qy.astype(np.float32), "q[3]": qz.astype(np.float32)})

    # --- gyro: mission-correlated term + noise whose amplitude is the
    #     known variance-ratio knob ---
    t_imu = _t(IMU_RATE_HZ)
    r2, p2, y2 = _mission(t_imu)
    noise = rng.standard_normal((3, t_imu.size)) * 0.02 * gyro_scale
    add("vehicle_angular_velocity",
        [("uint64_t", "timestamp"), ("float", "xyz[0]"), ("float", "xyz[1]"), ("float", "xyz[2]")],
        {"timestamp": stamps(t_imu),
         "xyz[0]": (np.gradient(r2, 1 / IMU_RATE_HZ) + noise[0]).astype(np.float32),
         "xyz[1]": (np.gradient(p2, 1 / IMU_RATE_HZ) + noise[1]).astype(np.float32),
         "xyz[2]": (np.gradient(np.unwrap(y2), 1 / IMU_RATE_HZ) + noise[2]).astype(np.float32)})

    # --- accel: 1 g on z plus a takeoff/landing excursion so the takeoff
    #     detector has something real to find ---
    az = np.full(t_imu.size, -9.81) + rng.standard_normal(t_imu.size) * 0.05 * gyro_scale
    az[(t_imu >= TAKEOFF_S) & (t_imu < TAKEOFF_S + 2.0)] -= 2.5
    add("sensor_combined",
        [("uint64_t", "timestamp"), ("float", "accelerometer_m_s2[0]"),
         ("float", "accelerometer_m_s2[1]"), ("float", "accelerometer_m_s2[2]")],
        {"timestamp": stamps(t_imu),
         "accelerometer_m_s2[0]": (rng.standard_normal(t_imu.size) * 0.1 * gyro_scale).astype(np.float32),
         "accelerometer_m_s2[1]": (rng.standard_normal(t_imu.size) * 0.1 * gyro_scale).astype(np.float32),
         "accelerometer_m_s2[2]": az.astype(np.float32)})

    # --- local position, NED (z negative up) ---
    t_pos = _t(ATT_RATE_HZ)
    alt = _altitude(t_pos)
    add("vehicle_local_position",
        [("uint64_t", "timestamp"), ("float", "x"), ("float", "y"), ("float", "z"),
         ("float", "vx"), ("float", "vy"), ("float", "vz")],
        {"timestamp": stamps(t_pos),
         "x": (20 * np.sin(2 * np.pi * 0.02 * t_pos)).astype(np.float32),
         "y": (20 * np.cos(2 * np.pi * 0.02 * t_pos)).astype(np.float32),
         "z": (-alt).astype(np.float32),
         "vx": np.gradient(20 * np.sin(2 * np.pi * 0.02 * t_pos), 1 / ATT_RATE_HZ).astype(np.float32),
         "vy": np.gradient(20 * np.cos(2 * np.pi * 0.02 * t_pos), 1 / ATT_RATE_HZ).astype(np.float32),
         "vz": np.gradient(-alt, 1 / ATT_RATE_HZ).astype(np.float32)})

    # --- GPS at 2 Hz: below the 5 Hz low-rate gate on purpose ---
    t_gps = _t(GPS_RATE_HZ)
    add("vehicle_gps_position",
        [("uint64_t", "timestamp"), ("uint8_t", "fix_type"),
         ("uint8_t", "satellites_used"), ("float", "eph")],
        {"timestamp": stamps(t_gps),
         "fix_type": np.full(t_gps.size, 3, dtype=np.uint8),
         "satellites_used": np.full(t_gps.size, 14, dtype=np.uint8),
         "eph": (0.8 + 0.1 * np.sin(t_gps)).astype(np.float32)})

    # --- actuator outputs in raw PWM microseconds (normalisation target) ---
    t_act = _t(ATT_RATE_HZ)
    base = 1500 + 200 * np.sin(2 * np.pi * 0.11 * t_act)
    add("actuator_outputs",
        [("uint64_t", "timestamp"), ("float", "output[0]"), ("float", "output[1]"),
         ("float", "output[2]"), ("float", "output[3]")],
        {"timestamp": stamps(t_act),
         **{f"output[{i}]": (base + 10 * i).astype(np.float32) for i in range(4)}})

    add("battery_status",
        [("uint64_t", "timestamp"), ("float", "voltage_filtered_v"), ("float", "current_filtered_a")],
        {"timestamp": stamps(_t(5.0)),
         "voltage_filtered_v": np.linspace(16.8, 14.9, _t(5.0).size).astype(np.float32),
         "current_filtered_a": (18 + 2 * np.sin(_t(5.0))).astype(np.float32)})

    add("cpuload",
        [("uint64_t", "timestamp"), ("float", "load"), ("float", "ram_usage")],
        {"timestamp": stamps(_t(2.0)),
         "load": (0.42 + 0.05 * np.sin(_t(2.0))).astype(np.float32),
         "ram_usage": np.full(_t(2.0).size, 0.61, dtype=np.float32)})

    ulog._data_list = datasets
    ulog.write_ulog(path)
    return path


def write_empty_ulog(path, *, boot_offset_us=0):
    """A valid ULog whose only topic is outside the canonical schema.

    Exercises spec 9 case 6 ("parses, but zero samples in every mapped
    topic"), which the spec is careful to distinguish from case 7
    (corrupt/unparseable). The file must therefore be structurally
    perfect -- the emptiness has to come from the logging configuration,
    not from damage.
    """
    from pyulog import ULog

    ulog = ULog(None)
    ulog._file_version = 1
    ulog._start_timestamp = boot_offset_us
    ulog._msg_info_dict = {"ver_sw": "v1.14.0-synthetic"}
    ulog._msg_info_dict_types = {"ver_sw": "char[20]"}
    ulog._initial_parameters = {"SYS_AUTOSTART": 4001.0}

    name = "vehicle_land_detected"  # real PX4 topic, deliberately not in ULOG_TOPICS
    fields = [("uint64_t", "timestamp"), ("bool", "landed")]
    ulog._message_formats[name] = _message_format(name, fields)
    t = _t(10.0)
    ulog._data_list = [
        _dataset(0, name, fields, {
            "timestamp": (t * 1e6 + boot_offset_us).astype(np.uint64),
            "landed": np.ones(t.size, dtype=np.int8),
        })
    ]
    ulog.write_ulog(path)
    return path


# ---------------------------------------------------------------------------
# ArduPilot DataFlash writer
# ---------------------------------------------------------------------------

HEAD1, HEAD2 = 0xA3, 0x95
FMT_TYPE = 0x80
# DataFlash type char -> struct char, for the subset used here. Mirrors
# pymavlink.DFReader.FORMAT_TO_STRUCT (verified against it directly), kept
# local so the fixture writer does not depend on a private table.
DF_STRUCT = {"Q": "Q", "f": "f", "B": "B", "h": "h", "H": "H", "i": "i", "I": "I",
             "n": "4s", "N": "16s", "Z": "64s"}


def _df_record_length(fmt):
    """Total on-disk length of one record of a message type: the 3-byte
    (HEAD1, HEAD2, type) header plus the packed payload.

    THIS IS THE FIELD IT IS EASIEST TO GET WRONG. The `Length` in an FMT
    record is the length of the message type BEING DEFINED, not the
    length of the FMT record itself. Declaring a constant 89 (FMT's own
    size) for every type makes DFReader read 89 bytes for a 35-byte ATT
    record, walk off the end of it, and emit a cascade of
    "bad header 0x...." errors -- which is exactly what happened on the
    first run of this generator.
    """
    return 3 + struct.calcsize("<" + "".join(DF_STRUCT[c] for c in fmt))


def _df_fmt_message(mtype, name, fmt, columns):
    body = struct.pack(
        "<BB4s16s64s",
        mtype,
        _df_record_length(fmt),
        name.encode()[:4].ljust(4, b"\0"),
        fmt.encode()[:16].ljust(16, b"\0"),
        ",".join(columns).encode()[:64].ljust(64, b"\0"),
    )
    return struct.pack("<BBB", HEAD1, HEAD2, FMT_TYPE) + body


def _df_data_message(mtype, fmt, values):
    packed = b""
    for spec, value in zip(fmt, values):
        sc = DF_STRUCT[spec]
        if sc.endswith("s"):
            packed += struct.pack("<" + sc, str(value).encode()[: int(sc[:-1])])
        else:
            packed += struct.pack("<" + sc, value)
    return struct.pack("<BBB", HEAD1, HEAD2, mtype) + packed


def write_dataflash(path, *, shift_s=0.0, boot_offset_us=0, seed=FIXTURE_SEED + 1):
    """
    Build a minimal but genuinely valid ArduPilot DataFlash binary log.

    Only the message types the canonical schema actually maps are
    emitted (ATT, IMU, CTUN, GPS, RCOU, BAT, VIBE, PARM) -- the same
    "only what 5.2 needs" discipline the parser applies on read.
    """
    rng = np.random.default_rng(seed)
    out = bytearray()

    defs = [
        (0x81, "PARM", "QNf", ["TimeUS", "Name", "Value"]),
        (0x82, "ATT", "Qffffff", ["TimeUS", "DesRoll", "Roll", "DesPitch", "Pitch", "DesYaw", "Yaw"]),
        (0x83, "IMU", "Qffffff", ["TimeUS", "GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"]),
        (0x84, "CTUN", "Qf", ["TimeUS", "Alt"]),
        # GWk/GMS are REQUIRED even though the canonical schema never reads
        # them: pymavlink's DFReaderClock_usec.find_time_base() calls
        # _gpsTimeToTime(gps.GWk, gps.GMS) to establish the log's time
        # base, and a GPS record missing them makes DFReader raise
        # "unsupported operand type(s) for *: 'int' and 'NoneType'" on
        # every GPS record. Real ArduPilot GPS records carry both, so
        # including them makes the fixture more faithful, not less.
        (0x85, "GPS", "QBIHBf", ["TimeUS", "Status", "GMS", "GWk", "NSats", "HDop"]),
        (0x86, "RCOU", "Qhhhh", ["TimeUS", "C1", "C2", "C3", "C4"]),
        (0x87, "BAT", "Qff", ["TimeUS", "Volt", "Curr"]),
        (0x88, "VIBE", "Qfff", ["TimeUS", "VibeX", "VibeY", "VibeZ"]),
    ]
    for mtype, name, fmt, cols in defs:
        out += _df_fmt_message(mtype, name, fmt, cols)

    for pname, pvalue in (("FRAME_CLASS", 1.0), ("RC1_MIN", 1000.0), ("RC1_MAX", 2000.0)):
        out += _df_data_message(0x81, "QNf", [0, pname, pvalue])

    def us(t):
        return (t * 1e6 + boot_offset_us).astype(np.int64)

    t_att = _t(ATT_RATE_HZ)
    roll, pitch, yaw = _mission(t_att + shift_s)
    stamps = us(t_att)
    for i in range(t_att.size):
        # ArduPilot logs attitude in DEGREES -- the parser converts.
        out += _df_data_message(0x82, "Qffffff", [
            int(stamps[i]), float(np.degrees(roll[i])), float(np.degrees(roll[i])),
            float(np.degrees(pitch[i])), float(np.degrees(pitch[i])),
            float(np.degrees(yaw[i])), float(np.degrees(yaw[i]))])

    t_imu = _t(IMU_RATE_HZ)
    r2, p2, y2 = _mission(t_imu + shift_s)
    gx = np.gradient(r2, 1 / IMU_RATE_HZ) + rng.standard_normal(t_imu.size) * 0.02
    gy = np.gradient(p2, 1 / IMU_RATE_HZ) + rng.standard_normal(t_imu.size) * 0.02
    gz = np.gradient(np.unwrap(y2), 1 / IMU_RATE_HZ) + rng.standard_normal(t_imu.size) * 0.02
    az = np.full(t_imu.size, -9.81) + rng.standard_normal(t_imu.size) * 0.05
    az[(t_imu >= TAKEOFF_S) & (t_imu < TAKEOFF_S + 2.0)] -= 2.5
    stamps = us(t_imu)
    for i in range(t_imu.size):
        out += _df_data_message(0x83, "Qffffff", [
            int(stamps[i]), float(gx[i]), float(gy[i]), float(gz[i]),
            0.0, 0.0, float(az[i])])

    t_ctun = _t(ATT_RATE_HZ)
    alt = _altitude(t_ctun)
    stamps = us(t_ctun)
    for i in range(t_ctun.size):
        out += _df_data_message(0x84, "Qf", [int(stamps[i]), float(alt[i])])

    t_gps = _t(GPS_RATE_HZ)
    stamps = us(t_gps)
    gps_week = 2330  # arbitrary but plausible GPS week; only used for DFReader's clock
    for i in range(t_gps.size):
        gms = int((t_gps[i] * 1000) % 604_800_000)  # ms into the GPS week
        out += _df_data_message(0x85, "QBIHBf", [int(stamps[i]), 3, gms, gps_week, 14, 0.9])

    t_rcou = _t(ATT_RATE_HZ)
    base = 1500 + 200 * np.sin(2 * np.pi * 0.11 * (t_rcou + shift_s))
    stamps = us(t_rcou)
    for i in range(t_rcou.size):
        out += _df_data_message(0x86, "Qhhhh", [int(stamps[i])] + [int(base[i] + 10 * k) for k in range(4)])

    t_bat = _t(5.0)
    volts = np.linspace(16.8, 14.9, t_bat.size)
    stamps = us(t_bat)
    for i in range(t_bat.size):
        out += _df_data_message(0x87, "Qff", [int(stamps[i]), float(volts[i]), 18.0])

    t_vibe = _t(10.0)
    stamps = us(t_vibe)
    for i in range(t_vibe.size):
        out += _df_data_message(0x88, "Qfff", [int(stamps[i]), 3.0, 3.1, 4.2])

    with open(path, "wb") as handle:
        handle.write(out)
    return path


# ---------------------------------------------------------------------------


def main():
    real_ulg = os.path.join(FIXTURE_DIR, "synthetic_real.ulg")
    sim_ulg = os.path.join(FIXTURE_DIR, "synthetic_sim.ulg")
    real_bin = os.path.join(FIXTURE_DIR, "synthetic_real.bin")
    truncated = os.path.join(FIXTURE_DIR, "truncated.ulg")
    empty_ulg = os.path.join(FIXTURE_DIR, "empty.ulg")
    garbage = os.path.join(FIXTURE_DIR, "garbage.bin")

    write_ulog(real_ulg, shift_s=0.0, boot_offset_us=BOOT_OFFSET_REAL_US, gyro_scale=1.0)
    # The sim log: same mission, delayed by a KNOWN SIM_TIME_SHIFT_S inside
    # its own recording, started at a different boot time, and deliberately
    # too smooth (spec 3.2's classic sim-to-real failure).
    write_ulog(
        sim_ulg,
        shift_s=SIM_TIME_SHIFT_S,
        boot_offset_us=BOOT_OFFSET_SIM_US,
        gyro_scale=np.sqrt(SIM_GYRO_VARIANCE_SCALE),
        seed=FIXTURE_SEED + 7,
        params={"SYS_AUTOSTART": 10016.0, "SIM_GPS_NOISE": 0.1,
                "PWM_MAIN_MIN": 1000.0, "PWM_MAIN_MAX": 2000.0},
        info={"ver_sw": "v1.14.0-synthetic", "sys_name": "PX4 SITL"},
    )
    write_dataflash(real_bin, shift_s=0.0, boot_offset_us=BOOT_OFFSET_REAL_US)

    # Spec 9 case 7: truncated mid-data-section. Cut at 60% -- past the
    # definition section (so the header and formats parse) but partway
    # through the data, which is exactly what a mid-flight power loss
    # produces.
    with open(real_ulg, "rb") as handle:
        blob = handle.read()
    with open(truncated, "wb") as handle:
        handle.write(blob[: int(len(blob) * 0.6)])

    # Spec 9 case 6: a structurally valid ULog with ZERO MAPPED topics.
    # It must contain a real topic (so it parses cleanly and is not
    # mistaken for the corrupt case 7) that carries none of the 25
    # canonical signals -- vehicle_land_detected is exactly that: a
    # genuine PX4 topic outside ULOG_TOPICS. An earlier version of this
    # fixture used `cpuload`, which is WRONG because cpuload maps to
    # cpu.load/mem.free and therefore parses to a non-empty result.
    write_empty_ulog(empty_ulg, boot_offset_us=BOOT_OFFSET_REAL_US)

    # Spec 9 case 17: not a flight log in any accepted format.
    with open(garbage, "wb") as handle:
        handle.write(b"this is not a flight log\n" * 40)

    for path in (real_ulg, sim_ulg, real_bin, truncated, empty_ulg, garbage):
        print(f"  {os.path.basename(path):24s} {os.path.getsize(path):>9,d} bytes")


# ---------------------------------------------------------------------------
# Health test logs: the real reference hex, perturbed per scenario
# ---------------------------------------------------------------------------
#
# Run:  python3 tests/fixtures/reality_gap/make_fixtures.py --health-logs ~/Desktop/drone-test-logs
#
# WHY THESE START FROM THE REAL LOG, NOT FROM _mission():
# The closed-form fixtures above are 60 s, 4-motor and carry none of the health
# topics (actuator_motors, estimator_status, vehicle_imu_status...), so they
# can't qualify for a health report at all. These start from real/px4_777ed799.ulg
# (12S hex, PX4 1.14, 643 s airborne, SDLOG_PROFILE 1), keep only the topics the
# parser reads (ULOG_TOPICS: a lossless round trip, every feature identical),
# and change one thing per scenario, so everything not under test is real
# flight data. Written with pyulog's write_ulog, like the fixtures above.
#
# Not committed: the real log is gitignored, and so are these.

HEALTH_SEED = 20261027
REAL_HEX = os.path.join(FIXTURE_DIR, "real", "px4_777ed799.ulg")
HEX_CELLS = 12
# Pack resistance the real log fits to (pwr.r_cell 3.58 mOhm/cell x 12 cells).
HEX_R_PACK_OHM = 3.58e-3 * HEX_CELLS
# 0-based motor indices by CA_ROTORn_PX in the real log: motors 3 and 5 at
# PX +0.57 (front), 4 and 6 at PX -0.57 (rear), 1 and 2 at PX 0.
HEX_FRONT, HEX_REAR = (2, 4), (3, 5)
SPINNING = 0.05          # actuator_motors below this is idle or disarmed: left alone
FIRST_FLIGHT_UTC = 1790834400   # 2026-10-01 06:00 UTC; each file is a later day


def _load_hex():
    from pyulog import ULog
    from app.services.reality_gap.parsers import ULOG_TOPICS
    return ULog(REAL_HEX, message_name_filter_list=ULOG_TOPICS, disable_str_exceptions=True)


def _instances(u, topic):
    """Every instance of a topic (vehicle_imu_status and estimator_status are
    logged per IMU / per EKF instance); scenarios change them all alike."""
    return [d.data for d in u.data_list if d.name == topic]


def _set(d, field, values):
    d[field] = np.asarray(values).astype(d[field].dtype)


def _takeoff_us(u):
    land = _instances(u, "vehicle_land_detected")[0]
    airborne = np.flatnonzero(land["landed"] == 0)
    return int(land["timestamp"][airborne[0]])


def _flown_at(u, utc_s):
    """PX4 carries wall-clock time in vehicle_gps_position.time_utc_usec, not
    in the ULog header (time_ref_utc is SDLOG_UTC_OFFSET in seconds). Shift
    it so each file has its own flight date; samples without a fix stay 0."""
    for d in _instances(u, "vehicle_gps_position"):
        t = d["time_utc_usec"].astype(np.int64)
        valid = t > 0
        shift = int(utc_s * 1_000_000) - int(t[valid][0])
        _set(d, "time_utc_usec", np.where(valid, t + shift, 0))


def _vibration(u, factor):
    """Scale vibration as a mechanical change would: PX4's
    accel_vibration_metric, and the high-frequency part of the accelerometer
    (above a 0.1 s moving average, roughly 5 Hz), which feeds the band features."""
    for d in _instances(u, "vehicle_imu_status"):
        _set(d, "accel_vibration_metric", d["accel_vibration_metric"] * factor)
    for d in _instances(u, "sensor_combined"):
        for ax in range(3):
            k = f"accelerometer_m_s2[{ax}]"
            x = d[k].astype(np.float64)
            slow = np.convolve(x, np.ones(21) / 21.0, mode="same")
            _set(d, k, slow + (x - slow) * factor)


def _motors(u, fn):
    """Apply fn(index, values) to each spinning motor's actuator_motors output."""
    for d in _instances(u, "actuator_motors"):
        for i in range(6):
            k = f"control[{i}]"
            x = d[k].astype(np.float64)
            spin = np.isfinite(x) & (x > SPINNING)
            _set(d, k, np.where(spin, np.clip(fn(i, x), 0.0, 1.0), x))


def _thrust(u, factor):
    for d in _instances(u, "vehicle_thrust_setpoint"):
        _set(d, "xyz[2]", d["xyz[2]"] * factor)


def _resistance(u, factor):
    """More internal resistance R drops terminal voltage by dR x I (V = OCV - R I),
    so the fit in features._resistance reads R x factor."""
    d_r = (factor - 1.0) * HEX_R_PACK_OHM
    for d in _instances(u, "battery_status"):
        for v, i in (("voltage_v", "current_a"), ("voltage_filtered_v", "current_filtered_a")):
            if v in d and i in d:
                _set(d, v, d[v].astype(np.float64) - d_r * d[i].astype(np.float64))


def _gps(u, rng, eph=None, sats=None, sacc=None):
    for d in _instances(u, "vehicle_gps_position"):
        n = d["timestamp"].size
        slow = np.sin(np.linspace(0, 6 * np.pi, n))   # slowly varying, as real GPS quality does
        if eph is not None:
            _set(d, "eph", np.clip(eph * (1 + 0.25 * slow) + rng.normal(0, 0.1 * eph, n), 0.01, None))
        if sats is not None:
            _set(d, "satellites_used", np.clip(np.round(sats + 1.2 * slow + rng.normal(0, 0.5, n)), 0, 40))
        if sacc is not None:
            _set(d, "s_variance_m_s", np.clip(sacc * (1 + 0.25 * slow) + rng.normal(0, 0.1 * sacc, n), 0.01, None))


def _test_ratios(u, factor=1.0, reject=None):
    """Scale every estimator test ratio; `reject` = (start_s, length_s) after
    takeoff during which the height test ratio sits at 1.05-1.6, as a baro
    disturbance the EKF rejects would."""
    t_off = _takeoff_us(u)
    for d in _instances(u, "estimator_status"):
        for k in ("vel_test_ratio", "pos_test_ratio", "hgt_test_ratio", "mag_test_ratio"):
            _set(d, k, d[k] * factor)
        if reject:
            t = (d["timestamp"].astype(np.int64) - t_off) / 1e6
            m = (t >= reject[0]) & (t < reject[0] + reject[1])
            bump = 1.05 + 0.55 * np.abs(np.sin(np.linspace(0, 5 * np.pi, int(m.sum()))))
            h = d["hgt_test_ratio"].astype(np.float64)
            h[m] = np.maximum(h[m], bump)
            _set(d, "hgt_test_ratio", h)


def _natural(u, rng):
    """Flight-to-flight variation of a healthy drone: small, and well inside
    every sigma floor in baseline.py, so a normal flight never reads as a change."""
    _vibration(u, 1 + float(np.clip(rng.normal(0, 0.025), -0.05, 0.05)))
    offsets = rng.normal(0, 0.002, 6)
    _motors(u, lambda i, x: x + offsets[i])
    _thrust(u, 1 + float(rng.normal(0, 0.006)))
    _resistance(u, 1 + float(np.clip(rng.normal(0, 0.02), -0.04, 0.04)))
    _test_ratios(u, 1 + float(np.clip(rng.normal(0, 0.04), -0.08, 0.08)))


def _crop(u, end_us):
    """Keep only samples up to end_us, as a log whose recording stopped there."""
    for ds in u.data_list:
        keep = ds.data["timestamp"].astype(np.int64) <= end_us
        ds.data = {k: v[keep] for k, v in ds.data.items()}


def write_health_ulog(path, day, perturb=None, natural=True):
    """One scenario: the real hex flight on `day` days after FIRST_FLIGHT_UTC,
    with natural variation and then `perturb(u, rng)` applied."""
    u = _load_hex()
    rng = np.random.default_rng([HEALTH_SEED, day])
    _flown_at(u, FIRST_FLIGHT_UTC + day * 86400 + int(rng.integers(0, 8 * 3600)))
    if natural:
        _natural(u, rng)
    if perturb:
        perturb(u, rng)
    u.write_ulog(path)
    return path


def write_corrupt_ulog(path, source, seed):
    """A valid ULog header (magic, version, timestamp) followed by noise where
    the definitions should be: identified as ULog, then unparseable."""
    with open(source, "rb") as fh:
        header = fh.read(16)
    noise = np.random.default_rng(seed).integers(0, 256, 2 * 1024 * 1024, dtype=np.uint8).tobytes()
    with open(path, "wb") as fh:
        fh.write(header + noise)
    return path


def write_dataflash_from_ulog(path, source):
    """
    An ArduPilot DataFlash log of the same real flight, converted from a
    health ULog so the ArduPilot path runs on 643 s of real dynamics rather
    than the 60 s closed-form fixture.

    APPROXIMATIONS (it is a conversion, not a recorded ArduPilot log):
    - VIBE is computed the way ArduPilot defines it (standard deviation of the
      accelerometer's high-passed output, per axis) from PX4's accelerometer,
      which PX4 has already low-passed at IMU_ACCEL_CUTOFF (30 Hz); a real
      ArduPilot IMU log at its own rate would read higher.
    - Motor outputs are actuator_motors mapped to 1000-2000 us PWM.
    - No ArduPilot EKF test ratios, attitude setpoint quaternions or GPS eph:
      the parser doesn't map them for DataFlash, so leaving them out matches
      what it would see in a real log.
    """
    from app.services.reality_gap.parsers import parse_log

    p = parse_log(source)
    s = p.signals
    out = bytearray()
    defs = [
        (0x81, "PARM", "QNf", ["TimeUS", "Name", "Value"]),
        (0x82, "ATT", "Qffffff", ["TimeUS", "DesRoll", "Roll", "DesPitch", "Pitch", "DesYaw", "Yaw"]),
        (0x83, "IMU", "Qffffff", ["TimeUS", "GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"]),
        (0x84, "CTUN", "Qf", ["TimeUS", "Alt"]),
        (0x85, "GPS", "QBIHBf", ["TimeUS", "Status", "GMS", "GWk", "NSats", "HDop"]),
        (0x86, "RCOU", "Qhhhhhh", ["TimeUS", "C1", "C2", "C3", "C4", "C5", "C6"]),
        (0x87, "BAT", "Qfff", ["TimeUS", "Volt", "Curr", "CurrTot"]),
        (0x88, "VIBE", "QfffIII", ["TimeUS", "VibeX", "VibeY", "VibeZ", "Clip0", "Clip1", "Clip2"]),
        (0x89, "RATE", "Qffff", ["TimeUS", "RDes", "PDes", "YDes", "AOut"]),
        (0x8A, "XKF1", "Qffffff", ["TimeUS", "VN", "VE", "VD", "PN", "PE", "PD"]),
        (0x8B, "GPA", "Qf", ["TimeUS", "SAcc"]),
        (0x8C, "MSG", "QZ", ["TimeUS", "Message"]),
    ]
    for mtype, name, fmt, cols in defs:
        out += _df_fmt_message(mtype, name, fmt, cols)
    params = [("FRAME_CLASS", 2.0), ("FRAME_TYPE", 1.0)]      # hexa, X
    for ch in range(1, 7):
        params += [(f"SERVO{ch}_FUNCTION", 32.0 + ch), (f"SERVO{ch}_MIN", 1000.0), (f"SERVO{ch}_MAX", 2000.0)]
    for name, value in params:
        out += _df_data_message(0x81, "QNf", [0, name, value])
    t_start = int(s["accel.z"].t_us[0])
    out += _df_data_message(0x8C, "QZ", [t_start, "ArduCopter V4.5.7 (synthetic conversion)"])
    out += _df_data_message(0x8C, "QZ", [t_start, "Frame: HEXA/X"])

    def grid(rate):
        t0, t1 = int(s["accel.z"].t_us[0]), int(s["accel.z"].t_us[-1])
        return np.arange(t0, t1, int(1e6 / rate), dtype=np.int64)

    def at(key, t):
        sig = s[key]
        return np.interp(t, sig.t_us.astype(np.float64), sig.values.astype(np.float64))

    def deg(key, t):
        return np.degrees(at(key, t))

    t = grid(50.0)
    att = [deg("att.roll", t), deg("att.pitch", t), deg("att.yaw", t)]
    for k in range(t.size):
        out += _df_data_message(0x82, "Qffffff", [int(t[k]), float(att[0][k]), float(att[0][k]), float(att[1][k]),
                                                  float(att[1][k]), float(att[2][k]), float(att[2][k])])
    t = grid(200.0)
    imu = [at(k, t) for k in ("gyro.x", "gyro.y", "gyro.z", "accel.x", "accel.y", "accel.z")]
    for k in range(t.size):
        out += _df_data_message(0x83, "Qffffff", [int(t[k])] + [float(v[k]) for v in imu])
    # VIBE at 10 Hz: per-axis standard deviation of the accelerometer above
    # ~5 Hz over the last 0.5 s (ArduPilot's definition, approximately).
    acc = imu[3:]
    hp = [a - np.convolve(a, np.ones(40) / 40.0, mode="same") for a in acc]
    rms = [np.sqrt(np.convolve(h ** 2, np.ones(100) / 100.0, mode="same")) for h in hp]
    t_v = grid(10.0)
    idx = np.searchsorted(t, t_v).clip(0, t.size - 1)
    for k, i in enumerate(idx):
        out += _df_data_message(0x88, "QfffIII", [int(t_v[k])] + [float(r[i]) for r in rms] + [0, 0, 0])
    t = grid(50.0)
    rate = [deg("rate_sp.roll", t), deg("rate_sp.pitch", t), deg("rate_sp.yaw", t), at("thrust.sp", t)]
    xkf = [at(k, t) for k in ("vel.x", "vel.y", "vel.z", "pos.x", "pos.y", "pos.z")]
    alt = at("alt.rel", t)
    motors = [at(f"act.motor{i}", t) for i in range(6)]
    for k in range(t.size):
        out += _df_data_message(0x89, "Qffff", [int(t[k])] + [float(v[k]) for v in rate])
        out += _df_data_message(0x8A, "Qffffff", [int(t[k])] + [float(v[k]) for v in xkf])
        out += _df_data_message(0x84, "Qf", [int(t[k]), float(alt[k])])
        out += _df_data_message(0x86, "Qhhhhhh", [int(t[k])] + [int(1000 + 1000 * np.clip(m[k], 0, 1)) for m in motors])
    t = grid(5.0)
    bat = [at("batt.voltage_raw", t), at("batt.current_raw", t), at("batt.discharged_mah", t)]
    sats, sacc = at("gps.nsats", t), at("gps.sacc", t)
    # GPS week and ms-of-week for the source flight's own start time (GPS time
    # is UTC + 18 leap seconds), so the converted log carries the same date.
    from datetime import datetime
    from app.services.reality_gap.parsers import GPS_EPOCH_UNIX_S, GPS_UTC_LEAP_S
    gps_s0 = datetime.fromisoformat(p.meta.start_time_utc).timestamp() - GPS_EPOCH_UNIX_S + GPS_UTC_LEAP_S
    for k in range(t.size):
        gps_s = gps_s0 + (t[k] - t_start) / 1e6
        week, ms = int(gps_s // 604_800), int(round((gps_s % 604_800) * 1000))
        out += _df_data_message(0x87, "Qfff", [int(t[k])] + [float(v[k]) for v in bat])
        out += _df_data_message(0x85, "QBIHBf", [int(t[k]), 3, ms, week, int(round(sats[k])), 0.7])
        out += _df_data_message(0x8B, "Qf", [int(t[k]), float(sacc[k])])
    with open(path, "wb") as fh:
        fh.write(out)
    return path


def _raise_motor(index, amount):
    return lambda u, rng: _motors(u, lambda i, x: x + amount if i == index else x)


def _cg_rearward(amount):
    """CG moved rearward: to hold attitude the rear pair (motors 4 and 6, adjacent)
    works harder and the front pair less, total thrust unchanged (deviation
    linear in PX). Rearward because the reference hex is already slightly
    rear-heavy (motors 2/4/6 high): moving its CG forward evens it out, which is
    a repair, not a fault."""
    def go(u, rng):
        _motors(u, lambda i, x: x + amount if i in HEX_REAR else (x - amount if i in HEX_FRONT else x))
    return go


def _saturate(index, gain):
    """A weak motor: the controller asks it for more, and on climbs and hard
    manoeuvres it pins at full output."""
    return lambda u, rng: _motors(u, lambda i, x: x * gain if i == index else x)


def health_log_plan():
    """(filename, builder) in the order the README lists them. Each builder
    takes the output path."""
    plan = []
    day = iter(range(1, 400))

    def ulog(name, perturb=None, natural=True):
        d = next(day)
        plan.append((name, lambda p, d=d: write_health_ulog(p, d, perturb, natural)))

    for k in range(1, 6):
        ulog(f"A{k}_normal.ulg")
    ulog("B1_vibration-high.ulg", lambda u, rng: _vibration(u, 1.6))
    ulog("B2_motor4-high.ulg", _raise_motor(3, 0.05))
    ulog("B3_motor-pair-high.ulg", _cg_rearward(0.035))
    ulog("B4_pack-resistance-up.ulg", lambda u, rng: _resistance(u, 1.4))
    ulog("B5_gps-poor.ulg", lambda u, rng: _gps(u, rng, eph=3.6, sats=5.5, sacc=0.75))
    ulog("B6_motor-saturated.ulg", _saturate(1, 1.65))
    ulog("B7_estimator-rejecting.ulg", lambda u, rng: _test_ratios(u, reject=(200.0, 90.0)))
    # Drift: five flat flights, then vibration +3% of its usual per flight.
    for k in range(1, 15):
        f = 1.0 + 0.03 * max(0, k - 5)
        ulog(f"C{k:02d}_drift-{'normal' if k <= 5 else f'vib+{round((f - 1) * 100)}pct'}.ulg",
             (lambda f: (lambda u, rng: _vibration(u, f)))(f) if f > 1 else None)
    # Step: five normal, then three with vibration 1.5x (a damaged prop left on).
    for k in range(1, 9):
        ulog(f"D{k:02d}_step-{'normal' if k <= 5 else 'shifted'}.ulg",
             (lambda u, rng: _vibration(u, 1.5)) if k > 5 else None)

    def cropped(after_takeoff_s):
        return lambda u, rng: _crop(u, _takeoff_us(u) + int(after_takeoff_s * 1e6))

    ulog("E1_too-short.ulg", cropped(50.0))
    ulog("E2_no-steady.ulg", cropped(90.0))
    ulog("E3_always-bad.ulg", lambda u, rng: _vibration(u, 2.2))
    plan.append(("E4_corrupt.ulg", lambda p: write_corrupt_ulog(p, REAL_HEX, HEALTH_SEED)))
    d5 = next(day)
    plan.append(("E5_ardupilot.bin", lambda p, d=d5: write_dataflash_from_ulog(
        p, write_health_ulog(p + ".src.ulg", d))))
    return plan


def make_health_logs(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, build in health_log_plan():
        path = os.path.join(out_dir, name)
        build(path)
        if os.path.exists(path + ".src.ulg"):
            os.remove(path + ".src.ulg")
        print(f"  {name:36s} {os.path.getsize(path) / 1e6:6.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--health-logs":
        print("Generating health test logs from the real reference hex...")
        make_health_logs(os.path.expanduser(sys.argv[2]))
    else:
        print("Generating D1 reality-gap fixtures...")
        main()
    print("Done.")
