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


class MavlinkImportError(Exception):
    pass


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
            ))

    if not points:
        raise MavlinkImportError(
            "No GLOBAL_POSITION_INT messages found in this log -- nothing to import. "
            "Make sure this is a flight log with GPS data, not a parameter-only log."
        )

    return points