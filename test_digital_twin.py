import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timedelta
from app.services.digital_twin import compute_digital_twin_stats


def reading(t, battery=None, lat=None, lon=None):
    return {"timestamp": t, "battery": battery, "latitude": lat, "longitude": lon}


def test_two_sessions_with_a_charge_between():
    base = datetime(2026, 1, 1, 12, 0, 0)
    readings = [
        # Session 1: readings 10 seconds apart (realistic telemetry
        # frequency), battery draining 100 -> 80 over a few minutes
        reading(base, battery=100, lat=12.97, lon=77.59),
        reading(base + timedelta(seconds=10), battery=95, lat=12.9702, lon=77.5902),
        reading(base + timedelta(seconds=20), battery=90, lat=12.9704, lon=77.5904),
        reading(base + timedelta(seconds=30), battery=80, lat=12.9706, lon=77.5906),
        # Gap of 2 hours (charging happened here) -- battery jumps back to 100
        reading(base + timedelta(hours=2), battery=100, lat=12.97, lon=77.59),
        reading(base + timedelta(hours=2, seconds=10), battery=95, lat=12.9705, lon=77.5905),
        reading(base + timedelta(hours=2, seconds=20), battery=85, lat=12.971, lon=77.591),
    ]
    stats = compute_digital_twin_stats(readings)
    assert stats["session_count"] == 2, f"Expected 2 sessions, got {stats['session_count']}"
    assert stats["charge_cycle_count"] == 1, f"Expected 1 charge cycle, got {stats['charge_cycle_count']}"
    assert stats["total_flight_hours"] > 0
    assert stats["total_distance_km"] > 0
    print(f"PASS: two sessions + one charge detected -> {stats}")


def test_empty_history():
    stats = compute_digital_twin_stats([])
    assert stats["reading_count"] == 0
    assert stats["motor_wear_percent"] == 0.0
    assert stats["maintenance_recommended"] is False
    print("PASS: empty history handled gracefully")


def test_high_flight_hours_triggers_maintenance_flag():
    # Build one GENUINELY continuous session: readings spaced 45 seconds
    # apart (right at, not over, the session-gap threshold) all the way
    # through 305 hours -- past the assumed 300-hour motor lifespan.
    # WHY THIS NEEDS A LOOP, NOT JUST TWO ENDPOINTS: _split_into_sessions
    # treats ANY gap over 45 seconds as a new session -- two readings 305
    # hours apart would just become two separate near-empty sessions
    # (caught by this exact mistake on the first attempt at this test),
    # not one long one. A real continuous session needs EVERY consecutive
    # pair within the threshold, which means actually generating the
    # readings in between, not just the start and end.
    base = datetime(2026, 1, 1, 0, 0, 0)
    total_seconds = 305 * 3600
    step_seconds = 45
    num_readings = total_seconds // step_seconds

    readings = [
        reading(base + timedelta(seconds=i * step_seconds), battery=100)
        for i in range(num_readings + 1)
    ]

    stats = compute_digital_twin_stats(readings)
    assert stats["session_count"] == 1, f"Expected exactly 1 continuous session, got {stats['session_count']}"
    assert stats["motor_wear_percent"] >= 80, f"Expected high wear, got {stats['motor_wear_percent']}"
    assert stats["maintenance_recommended"] is True
    print(f"PASS: long flight hours correctly triggers maintenance flag -> wear={stats['motor_wear_percent']}%")


if __name__ == "__main__":
    test_two_sessions_with_a_charge_between()
    test_empty_history()
    test_high_flight_hours_triggers_maintenance_flag()
    print("\nAll digital twin tests passed.")
