"""
WHY THIS EXISTS:
Everything so far analyzes ONE flight session at a time. This looks at a
drone's ENTIRE lifetime of logged telemetry to answer: how much has this
specific physical drone actually been used, and is it due for maintenance?

HONEST LIMITATION, stated plainly rather than glossed over: motor wear
here is an ESTIMATE based on total flight hours against an assumed
typical hobbyist brushless motor lifespan (300 hours) -- NOT calculated
from real manufacturer thrust-curve/wear data, which we don't have access
to (this was flagged as a genuine constraint earlier: real motor wear
modeling needs actual hardware datasheets). This is a reasonable,
clearly-labeled heuristic, not a claim of precision.
"""

import math
from datetime import datetime
from typing import List, Optional

SESSION_GAP_SECONDS = 45  # matches the same threshold used in Flight Verification
ASSUMED_MOTOR_LIFESPAN_HOURS = 300  # typical hobbyist brushless motor estimate
CHARGE_DETECTION_THRESHOLD_PERCENT = 5  # battery jump this big = "it got charged," not sensor noise


def _distance_meters(lat1, lon1, lat2, lon2) -> float:
    meters_per_deg_lat = 110540
    meters_per_deg_lon = 111320 * math.cos(math.radians(lat1))
    dz = (lat2 - lat1) * meters_per_deg_lat
    dx = (lon2 - lon1) * meters_per_deg_lon
    return math.sqrt(dx * dx + dz * dz)


def _split_into_sessions(readings: List[dict]) -> List[List[dict]]:
    """
    Splits a chronologically-sorted list of readings into separate
    flight sessions, wherever the time gap between consecutive readings
    exceeds SESSION_GAP_SECONDS -- the same logic used in Flight
    Verification, just applied across the WHOLE history instead of just
    the most recent session.
    """
    if not readings:
        return []

    sessions = [[readings[0]]]
    for i in range(1, len(readings)):
        gap = (readings[i]["timestamp"] - readings[i - 1]["timestamp"]).total_seconds()
        if gap > SESSION_GAP_SECONDS:
            sessions.append([])
        sessions[-1].append(readings[i])
    return sessions


def compute_digital_twin_stats(readings: List[dict]) -> dict:
    """
    `readings` should be a list of dicts with at least: timestamp,
    battery (optional), latitude (optional), longitude (optional) --
    sorted OLDEST FIRST.
    """
    if not readings:
        return {
            "total_flight_hours": 0.0,
            "total_distance_km": 0.0,
            "charge_cycle_count": 0,
            "motor_wear_percent": 0.0,
            "maintenance_recommended": False,
            "session_count": 0,
            "reading_count": 0,
        }

    sessions = _split_into_sessions(readings)

    total_flight_seconds = 0.0
    total_distance_meters = 0.0

    for session in sessions:
        if len(session) >= 2:
            total_flight_seconds += (session[-1]["timestamp"] - session[0]["timestamp"]).total_seconds()
            for i in range(1, len(session)):
                a, b = session[i - 1], session[i]
                if a.get("latitude") is not None and b.get("latitude") is not None:
                    total_distance_meters += _distance_meters(
                        a["latitude"], a["longitude"], b["latitude"], b["longitude"]
                    )

    # Charge cycles: count meaningful battery INCREASES between
    # consecutive readings, regardless of session boundaries (charging
    # typically happens between sessions, exactly where a gap would be).
    charge_cycles = 0
    for i in range(1, len(readings)):
        prev_battery = readings[i - 1].get("battery")
        curr_battery = readings[i].get("battery")
        if prev_battery is not None and curr_battery is not None:
            if curr_battery - prev_battery >= CHARGE_DETECTION_THRESHOLD_PERCENT:
                charge_cycles += 1

    total_flight_hours = total_flight_seconds / 3600
    motor_wear_percent = min(100.0, (total_flight_hours / ASSUMED_MOTOR_LIFESPAN_HOURS) * 100)

    return {
        "total_flight_hours": round(total_flight_hours, 2),
        "total_distance_km": round(total_distance_meters / 1000, 3),
        "charge_cycle_count": charge_cycles,
        "motor_wear_percent": round(motor_wear_percent, 1),
        "maintenance_recommended": motor_wear_percent >= 80,
        "session_count": len(sessions),
        "reading_count": len(readings),
    }
