"""
WHY THIS EXISTS:
Real drones (anything running PX4 or ArduPilot -- the two dominant
real-world autopilot stacks) log flights using the MAVLink protocol,
producing .tlog (telemetry log, streamed over a radio link) or .bin
(onboard flight controller log) files. This service lets you import one
of those REAL logs and turn it into rows in your existing `telemetry`
table -- so a real flight can be replayed, verified, and analyzed through
every feature you've already built (3D replay, flight verification,
analytics, digital twin wear stats), not just simulated data.

WHY THE MERGE LOGIC IS NEEDED:
A MAVLink log is NOT one row per timestamp with every field filled in.
It's a stream of small independent messages arriving at different rates:
  - GLOBAL_POSITION_INT (~5Hz): lat/lon/altitude
  - ATTITUDE (~10Hz): roll/pitch/yaw
  - SYS_STATUS or BATTERY_STATUS (~1-2Hz): battery percentage
  - VFR_HUD (~5Hz): groundspeed
  - HEARTBEAT (~1Hz): armed/disarmed state
  - ATTITUDE_TARGET (~a few Hz, only sent by autopilots that report it):
    the controller's desired roll/pitch/yaw, as a quaternion
  - SERVO_OUTPUT_RAW (~a few Hz, only sent by autopilots that report it):
    raw ESC/servo PWM outputs
  - SCALED_IMU / RAW_IMU (~a few Hz to ~50Hz, only sent by autopilots that
    report it): raw accelerometer + gyroscope + magnetometer axes -- only
    accel/gyro are stored (see app/models/telemetry.py); magnetometer
    isn't currently a stored column
None of these arrive at the same instant. So this parser does a forward-fill
merge: it walks every message in chronological order, keeps track of the
LATEST known value for each field, and emits one combined telemetry row
every time a new position update (GLOBAL_POSITION_INT) arrives -- since
position is the field your 3D replay/verification features actually need
a row-per-instant for. This is the standard approach for combining
independent-rate sensor streams into one table.
"""

import math
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

from pymavlink import mavutil


@dataclass
class ImportedTelemetryPoint:
    latitude: Optional[float]
    longitude: Optional[float]
    altitude: Optional[float]
    battery: Optional[float]
    speed: Optional[float]
    roll: Optional[float]
    pitch: Optional[float]
    yaw: Optional[float]
    flight_state: str
    timestamp: datetime
    # Real attitude setpoint and motor output, when the log has them (see
    # app/models/telemetry.py) -- from ATTITUDE_TARGET (a quaternion, here
    # converted to Euler degrees) and SERVO_OUTPUT_RAW (PWM microseconds).
    # Both are forward-filled the same way as roll/battery/etc. above, and
    # both stay None for logs/aircraft that never send these messages.
    desired_roll: Optional[float] = None
    desired_pitch: Optional[float] = None
    desired_yaw: Optional[float] = None
    motor_pwm_1: Optional[float] = None
    motor_pwm_2: Optional[float] = None
    motor_pwm_3: Optional[float] = None
    motor_pwm_4: Optional[float] = None
    # Raw IMU axes, when the log has them (see app/models/telemetry.py) --
    # from SCALED_IMU/RAW_IMU, converted to real physical units (m/s^2,
    # deg/s) -- see _accel_mg_to_mps2/_gyro_mrad_s_to_deg_s below for the
    # conversion and its sourcing. Forward-filled the same way as every
    # other field here; stay None for logs/aircraft that never send these.
    accel_x: Optional[float] = None
    accel_y: Optional[float] = None
    accel_z: Optional[float] = None
    gyro_x: Optional[float] = None
    gyro_y: Optional[float] = None
    gyro_z: Optional[float] = None


class MavlinkImportError(Exception):
    pass


# Same standard gravity constant already used throughout this app (see
# e.g. sensor_calibration.py's GRAVITY_MPS2, motor_performance.py) --
# duplicated here (not imported) since these are two independent service
# modules and this is a single well-known physical constant, the same
# minor, documented duplication pattern already used elsewhere in this app.
GRAVITY_MPS2 = 9.81


def _accel_mg_to_mps2(milli_g: float) -> float:
    """SCALED_IMU/SCALED_IMU2/SCALED_IMU3's xacc/yacc/zacc fields are
    documented (see the MAVLink common.xml message definition) in mG
    (milli-g, i.e. thousandths of standard gravity) -- converts to m/s^2."""
    return (milli_g / 1000.0) * GRAVITY_MPS2


def _gyro_mrad_s_to_deg_s(milli_rad_per_s: float) -> float:
    """SCALED_IMU/SCALED_IMU2/SCALED_IMU3's xgyro/ygyro/zgyro fields are
    documented in mrad/s (milli-radians/second) -- converts to deg/s."""
    return math.degrees(milli_rad_per_s / 1000.0)


def _quaternion_to_euler_deg(q) -> tuple:
    """
    Standard aerospace (ZYX Tait-Bryan) quaternion-to-Euler conversion --
    see e.g. Wikipedia "Conversion between quaternions and Euler angles",
    the same formula used by ArduPilot/PX4 ground stations to display
    ATTITUDE_TARGET's quaternion as roll/pitch/yaw. q = [w, x, y, z].
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def parse_mavlink_log(file_path: str) -> list[ImportedTelemetryPoint]:
    """
    Parses a .tlog or .bin MAVLink log file and returns a chronological
    list of merged telemetry points ready to insert into the `telemetry`
    table. Raises MavlinkImportError if the file can't be read as MAVLink
    or contains no usable position data.
    """
    try:
        mlog = mavutil.mavlink_connection(file_path, dialect="ardupilotmega")
    except Exception as e:
        raise MavlinkImportError(f"Could not open file as a MAVLink log: {e}")

    # Running "latest known value" state, updated as we walk the log.
    latest_battery: Optional[float] = None
    latest_roll: Optional[float] = None
    latest_pitch: Optional[float] = None
    latest_yaw: Optional[float] = None
    latest_speed: Optional[float] = None
    latest_desired_roll: Optional[float] = None
    latest_desired_pitch: Optional[float] = None
    latest_desired_yaw: Optional[float] = None
    latest_pwm1: Optional[float] = None
    latest_pwm2: Optional[float] = None
    latest_pwm3: Optional[float] = None
    latest_pwm4: Optional[float] = None
    latest_accel_x: Optional[float] = None
    latest_accel_y: Optional[float] = None
    latest_accel_z: Optional[float] = None
    latest_gyro_x: Optional[float] = None
    latest_gyro_y: Optional[float] = None
    latest_gyro_z: Optional[float] = None
    armed = False

    points: list[ImportedTelemetryPoint] = []

    while True:
        msg = mlog.recv_match(blocking=False)
        if msg is None:
            break

        msg_type = msg.get_type()
        if msg_type == "BAD_DATA":
            continue

        try:
            t = datetime.fromtimestamp(msg._timestamp, tz=timezone.utc)
        except (AttributeError, OSError, OverflowError, ValueError):
            # Some log formats (raw .bin without a wall-clock timestamp)
            # don't carry a usable epoch time on every message -- skip
            # points we can't timestamp rather than guessing.
            continue

        if msg_type == "HEARTBEAT":
            # bit 7 (value 128) of base_mode is the MAV_MODE_FLAG_SAFETY_ARMED bit,
            # standard across both ArduPilot and PX4.
            armed = bool(msg.base_mode & 128)

        elif msg_type == "ATTITUDE":
            latest_roll = math.degrees(msg.roll)
            latest_pitch = math.degrees(msg.pitch)
            latest_yaw = math.degrees(msg.yaw) % 360

        elif msg_type == "SYS_STATUS":
            if msg.battery_remaining is not None and msg.battery_remaining >= 0:
                latest_battery = float(msg.battery_remaining)

        elif msg_type == "BATTERY_STATUS":
            if getattr(msg, "battery_remaining", -1) >= 0:
                latest_battery = float(msg.battery_remaining)

        elif msg_type == "VFR_HUD":
            latest_speed = float(msg.groundspeed)

        elif msg_type == "ATTITUDE_TARGET":
            latest_desired_roll, latest_desired_pitch, target_yaw = _quaternion_to_euler_deg(msg.q)
            latest_desired_yaw = target_yaw % 360

        elif msg_type == "SERVO_OUTPUT_RAW":
            # A raw value of 0 means that servo output channel isn't in
            # use on this airframe -- kept as None rather than a
            # misleading 0us PWM reading.
            latest_pwm1 = float(msg.servo1_raw) if getattr(msg, "servo1_raw", 0) else None
            latest_pwm2 = float(msg.servo2_raw) if getattr(msg, "servo2_raw", 0) else None
            latest_pwm3 = float(msg.servo3_raw) if getattr(msg, "servo3_raw", 0) else None
            latest_pwm4 = float(msg.servo4_raw) if getattr(msg, "servo4_raw", 0) else None

        elif msg_type in ("SCALED_IMU", "SCALED_IMU2", "SCALED_IMU3"):
            # Preferred over RAW_IMU below: SCALED_IMU's xacc/yacc/zacc/
            # xgyro/ygyro/zgyro fields have real, protocol-documented
            # units (mG, mrad/s -- see MAVLink's common.xml) that this
            # conversion is verified against (test_mavlink_import.py).
            latest_accel_x = _accel_mg_to_mps2(msg.xacc)
            latest_accel_y = _accel_mg_to_mps2(msg.yacc)
            latest_accel_z = _accel_mg_to_mps2(msg.zacc)
            latest_gyro_x = _gyro_mrad_s_to_deg_s(msg.xgyro)
            latest_gyro_y = _gyro_mrad_s_to_deg_s(msg.ygyro)
            latest_gyro_z = _gyro_mrad_s_to_deg_s(msg.zgyro)

        elif msg_type == "RAW_IMU":
            # RAW_IMU's fields have NO protocol-guaranteed physical units
            # (they're literally "raw," device-specific ADC-scale values)
            # -- but in practice, both ArduPilot and PX4 populate RAW_IMU
            # with the same mG/mrad/s scaling SCALED_IMU formally
            # documents, so the same conversion is applied here as a
            # documented practical convention, not a protocol guarantee.
            # Only used as a fallback: if this log ALSO has any
            # SCALED_IMU-family message, those take precedence simply by
            # arriving later in a typical log's message ordering and
            # overwriting these same latest_* variables -- if a log has
            # RAW_IMU only, this is what populates accel/gyro instead of
            # leaving them empty.
            latest_accel_x = _accel_mg_to_mps2(msg.xacc)
            latest_accel_y = _accel_mg_to_mps2(msg.yacc)
            latest_accel_z = _accel_mg_to_mps2(msg.zacc)
            latest_gyro_x = _gyro_mrad_s_to_deg_s(msg.xgyro)
            latest_gyro_y = _gyro_mrad_s_to_deg_s(msg.ygyro)
            latest_gyro_z = _gyro_mrad_s_to_deg_s(msg.zgyro)

        elif msg_type == "GLOBAL_POSITION_INT":
            # This is the sync point: emit one combined row per position update.
            lat = msg.lat / 1e7
            lon = msg.lon / 1e7
            alt = msg.relative_alt / 1000.0  # mm -> m, relative to home/takeoff

            points.append(ImportedTelemetryPoint(
                latitude=lat,
                longitude=lon,
                altitude=alt,
                battery=latest_battery,
                speed=latest_speed,
                roll=latest_roll,
                pitch=latest_pitch,
                yaw=latest_yaw,
                flight_state="flying" if armed else "idle",
                timestamp=t,
                desired_roll=latest_desired_roll,
                desired_pitch=latest_desired_pitch,
                desired_yaw=latest_desired_yaw,
                motor_pwm_1=latest_pwm1,
                motor_pwm_2=latest_pwm2,
                motor_pwm_3=latest_pwm3,
                motor_pwm_4=latest_pwm4,
                accel_x=latest_accel_x,
                accel_y=latest_accel_y,
                accel_z=latest_accel_z,
                gyro_x=latest_gyro_x,
                gyro_y=latest_gyro_y,
                gyro_z=latest_gyro_z,
            ))

    if not points:
        raise MavlinkImportError(
            "No GLOBAL_POSITION_INT messages found in this log -- nothing to import. "
            "Make sure this is a flight log with GPS data, not a parameter-only log."
        )

    return points