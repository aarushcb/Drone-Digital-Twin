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


if __name__ == "__main__":
    print("Generating D1 reality-gap fixtures...")
    main()
    print("Done.")
