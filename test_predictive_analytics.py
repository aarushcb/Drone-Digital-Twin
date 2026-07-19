"""
Standalone test script -- run with `python3 test_predictive_analytics.py`,
no server or dependencies needed beyond the standard library.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timedelta
from app.services.predictive_analytics import (
    estimate_battery_remaining,
    detect_anomalies,
    calculate_predictive_risk_score,
)


class FakeTelemetry:
    """A minimal stand-in for the real Telemetry ORM object -- just needs
    the same attribute names the prediction functions read."""
    def __init__(self, timestamp, battery=None, temperature=None, speed=None):
        self.timestamp = timestamp
        self.battery = battery
        self.temperature = temperature
        self.speed = speed


def test_battery_prediction_on_known_linear_drain():
    """Battery draining EXACTLY 1% per minute, starting at 100% -- the
    correct answer is fully computable by hand, so this checks the
    regression math is actually right, not just plausible-looking."""
    base_time = datetime(2026, 1, 1, 12, 0, 0)
    history = [
        FakeTelemetry(base_time + timedelta(minutes=i), battery=100 - i)
        for i in range(10)
    ]
    # At minute 9, battery = 91%, draining 1%/min -> should hit 0% at minute 100,
    # i.e. 91 more minutes from now.
    result = estimate_battery_remaining(history)
    assert result is not None, "Expected a prediction with 10 clean linear readings"
    assert abs(result["drain_rate_percent_per_min"] - (-1.0)) < 0.01, (
        f"Expected drain rate ~-1.0%/min, got {result['drain_rate_percent_per_min']}"
    )
    assert abs(result["minutes_remaining"] - 91.0) < 0.5, (
        f"Expected ~91 minutes remaining, got {result['minutes_remaining']}"
    )
    assert result["confidence"] > 0.99, (
        f"Perfectly linear data should give near-1.0 confidence, got {result['confidence']}"
    )
    print(f"PASS: known linear drain -> {result}")


def test_battery_prediction_returns_none_with_insufficient_data():
    """Only 3 readings -- below MIN_READINGS_FOR_PREDICTION -- should
    honestly decline to predict rather than guessing from too little data."""
    base_time = datetime(2026, 1, 1, 12, 0, 0)
    history = [FakeTelemetry(base_time + timedelta(minutes=i), battery=100 - i) for i in range(3)]
    result = estimate_battery_remaining(history)
    assert result is None, "Expected no prediction with insufficient data"
    print("PASS: insufficient data correctly returns None")


def test_battery_prediction_handles_stable_battery():
    """Battery holding steady (e.g. charging or on the ground) shouldn't
    produce a scary 'minutes remaining' countdown."""
    base_time = datetime(2026, 1, 1, 12, 0, 0)
    history = [FakeTelemetry(base_time + timedelta(minutes=i), battery=100) for i in range(10)]
    result = estimate_battery_remaining(history)
    assert result is not None
    assert result["minutes_remaining"] is None, (
        "Stable battery should not produce a time-remaining estimate"
    )
    print("PASS: stable battery correctly reports no countdown")


def test_anomaly_detection_flags_a_clear_outlier():
    """9 normal readings around 40°C, then one at 95°C -- should be
    flagged as an anomaly against this drone's own baseline."""
    base_time = datetime(2026, 1, 1, 12, 0, 0)
    history = [
        FakeTelemetry(base_time + timedelta(minutes=i), temperature=40 + (i % 3))
        for i in range(9)
    ]
    current = FakeTelemetry(base_time + timedelta(minutes=9), temperature=95)
    anomalies = detect_anomalies(current, history)
    assert len(anomalies) == 1, f"Expected exactly one anomaly, got {anomalies}"
    assert "Temperature" in anomalies[0]
    print(f"PASS: clear outlier flagged -> {anomalies[0]}")


def test_anomaly_detection_does_not_flag_normal_variation():
    """Readings that vary normally within a tight, realistic range
    shouldn't trigger false alarms."""
    base_time = datetime(2026, 1, 1, 12, 0, 0)
    history = [
        FakeTelemetry(base_time + timedelta(minutes=i), temperature=40 + (i % 3))
        for i in range(9)
    ]
    current = FakeTelemetry(base_time + timedelta(minutes=9), temperature=41)
    anomalies = detect_anomalies(current, history)
    assert len(anomalies) == 0, f"Expected no anomalies for normal variation, got {anomalies}"
    print("PASS: normal variation correctly produces no false alarm")


def test_risk_score_increases_with_worse_conditions():
    """Sanity check on the overall scoring: a drone with low battery,
    imminent depletion, and high temperature should score meaningfully
    higher than a healthy drone -- confirms the weighting actually
    responds to worse conditions rather than being a fixed number."""
    base_time = datetime(2026, 1, 1, 12, 0, 0)

    healthy = FakeTelemetry(base_time, battery=90, temperature=35, speed=5)
    healthy_score = calculate_predictive_risk_score(healthy, [], None)

    critical = FakeTelemetry(base_time, battery=10, temperature=85, speed=5)
    critical_estimate = {"minutes_remaining": 1.0}
    critical_score = calculate_predictive_risk_score(critical, [], critical_estimate)

    assert healthy_score < 20, f"Expected a low score for healthy conditions, got {healthy_score}"
    assert critical_score > 70, f"Expected a high score for critical conditions, got {critical_score}"
    assert critical_score > healthy_score
    print(f"PASS: risk scoring responds correctly (healthy={healthy_score}, critical={critical_score})")


if __name__ == "__main__":
    test_battery_prediction_on_known_linear_drain()
    test_battery_prediction_returns_none_with_insufficient_data()
    test_battery_prediction_handles_stable_battery()
    test_anomaly_detection_flags_a_clear_outlier()
    test_anomaly_detection_does_not_flag_normal_variation()
    test_risk_score_increases_with_worse_conditions()
    print("\nAll predictive analytics tests passed.")
