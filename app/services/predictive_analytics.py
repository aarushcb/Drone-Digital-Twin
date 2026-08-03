"""
WHY THIS REPLACES THE OLD THRESHOLD LOGIC:
Since Phase 2, health_monitor.py and drone_analytics.py have used fixed
rules like "if battery < 20: CRITICAL" -- flagged as a deliberate
placeholder from the very first architecture review, precisely so it could
be swapped out once there was real historical telemetry to learn from.
That data now exists (thanks to testing across every phase). This module
is the upgrade: two genuinely statistical techniques instead of hardcoded
numbers.

WHY LINEAR REGRESSION + Z-SCORES, NOT A "BIGGER" MODEL:
A single drone realistically produces dozens to a few hundred telemetry
readings during testing/early use -- nowhere near enough data for a neural
network or even a moderately complex model to learn anything meaningful;
it would just memorize noise. Linear regression and z-score anomaly
detection are the CORRECT tools for this amount of data, not a
simplification made to avoid real work -- and critically, both are fully
explainable: you can look at the numbers and understand exactly why a
score came out the way it did, which matters for a safety-relevant metric
on a real drone. A deep learning model becomes worth it once there's
enough accumulated history across many flights to justify it.
"""

from typing import List, Optional, Dict
from datetime import datetime


def _linear_regression(x: List[float], y: List[float]) -> tuple[float, float]:
    """
    Ordinary least-squares regression, pure Python (no numpy dependency
    for something this small). Returns (slope, intercept) for the
    best-fit line y = slope * x + intercept.
    """
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator = sum((xi - mean_x) ** 2 for xi in x)
    if denominator == 0:
        return 0.0, mean_y
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _r_squared(x: List[float], y: List[float], slope: float, intercept: float) -> float:
    """
    How well the regression line actually fits the data, from 0 (no fit)
    to 1 (perfect fit) -- used as a confidence score so predictions come
    with an honest sense of how much to trust them, rather than presenting
    a single number as if it were certain.
    """
    y_pred = [slope * xi + intercept for xi in x]
    mean_y = sum(y) / len(y)
    ss_res = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred))
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return max(0.0, min(1.0, 1 - ss_res / ss_tot))


MIN_READINGS_FOR_PREDICTION = 5


def estimate_battery_remaining(telemetry_history: List) -> Optional[Dict]:
    """
    Fits a straight line through recent battery-percentage-over-time
    readings and extrapolates forward to estimate when battery will hit 0%.
    telemetry_history should be ordered oldest-first.

    Returns None if there isn't enough data yet, or if the battery isn't
    meaningfully draining (flat or charging) -- an honest "can't predict
    this right now" rather than forcing a number out of insufficient data.
    """
    readings = [
        (t.timestamp, t.battery) for t in telemetry_history if t.battery is not None
    ]
    if len(readings) < MIN_READINGS_FOR_PREDICTION:
        return None

    readings.sort(key=lambda r: r[0])
    t0 = readings[0][0]
    x = [(ts - t0).total_seconds() / 60 for ts, _ in readings]  # minutes elapsed
    y = [battery for _, battery in readings]

    slope, intercept = _linear_regression(x, y)  # slope = % battery change per minute

    if slope >= -0.01:
        # Battery is flat or charging -- there's no meaningful "time until
        # empty" to report right now.
        return {
            "drain_rate_percent_per_min": round(slope, 3),
            "minutes_remaining": None,
            "confidence": None,
            "sample_size": len(readings),
        }

    current_battery = y[-1]
    minutes_remaining = max(0.0, -current_battery / slope)
    confidence = _r_squared(x, y, slope, intercept)

    return {
        "drain_rate_percent_per_min": round(slope, 3),
        "minutes_remaining": round(minutes_remaining, 1),
        "confidence": round(confidence, 2),
        "sample_size": len(readings),
    }


MIN_READINGS_FOR_ANOMALY_DETECTION = 8
ANOMALY_Z_SCORE_THRESHOLD = 2.5


def detect_anomalies(current, telemetry_history: List) -> List[str]:
    """
    Flags when the CURRENT reading is a statistical outlier compared to
    THIS SPECIFIC DRONE's own recent history -- not a fixed global
    threshold. A temperature of 55°C might be completely normal for one
    drone's motors and genuinely alarming for another; comparing against
    each drone's own baseline is what makes this meaningfully different
    from (and better than) a hardcoded "temperature > 60" rule.
    """
    anomalies = []

    for field, label in [("temperature", "Temperature"), ("speed", "Speed")]:
        values = [
            getattr(t, field) for t in telemetry_history if getattr(t, field) is not None
        ]
        current_value = getattr(current, field, None)

        if current_value is None or len(values) < MIN_READINGS_FOR_ANOMALY_DETECTION:
            continue

        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std_dev = variance ** 0.5

        if std_dev == 0:
            continue  # every historical reading was identical -- no meaningful spread to compare against

        z_score = (current_value - mean) / std_dev
        if abs(z_score) > ANOMALY_Z_SCORE_THRESHOLD:
            direction = "higher" if z_score > 0 else "lower"
            anomalies.append(
                f"{label} ({current_value:.1f}) is unusually {direction} than this "
                f"drone's typical range (usual average: {mean:.1f})"
            )

    return anomalies


def calculate_predictive_risk_score(
    current, telemetry_history: List, battery_estimate: Optional[Dict],
    sensor_fault_penalty: int = 0,
) -> int:
    """
    A weighted 0-100 score combining several factors, replacing the old
    flat threshold-based score. Each factor's point contribution is
    intentionally simple and inspectable (not a trained/opaque weighting)
    so the score stays explainable -- you can look at any given score and
    trace exactly which factors drove it.

    sensor_fault_penalty: points contributed by Kalman-filter-based
    sensor/actuator fault detection (see
    app/services/kalman_filter.py's detect_sensor_faults) -- computed by
    the caller (routes.py), since it needs the full altitude time series,
    not just the single `current` reading this function otherwise looks
    at. Defaults to 0 so every existing caller/test that doesn't pass it
    behaves exactly as before.
    """
    score = 0

    # Current battery level: the most direct risk indicator.
    if current.battery is not None:
        if current.battery < 15:
            score += 40
        elif current.battery < 30:
            score += 25
        elif current.battery < 50:
            score += 10

    # Predicted time until battery depletion: a LOW battery that's stable
    # is less urgent than a HIGHER battery draining fast enough to run out
    # imminently -- this is exactly the kind of thing a flat threshold on
    # battery percentage alone can't capture.
    if battery_estimate and battery_estimate.get("minutes_remaining") is not None:
        minutes = battery_estimate["minutes_remaining"]
        if minutes < 2:
            score += 25
        elif minutes < 5:
            score += 15
        elif minutes < 10:
            score += 5

    # Temperature.
    if current.temperature is not None:
        if current.temperature > 80:
            score += 20
        elif current.temperature > 60:
            score += 10

    # Statistical anomalies (see detect_anomalies above) -- capped so a
    # noisy sensor spamming anomalies can't single-handedly max the score.
    anomalies = detect_anomalies(current, telemetry_history)
    score += min(15, len(anomalies) * 8)

    # Kalman-filter-detected sensor/actuator faults -- a genuinely
    # different signal from the z-score anomalies above (see the
    # FAULT DETECTION section of kalman_filter.py's docstring): those
    # compare a value to this drone's own history, this compares it to
    # what the drone's own DYNAMICS predicted a moment ago. Capped for
    # the same reason anomalies are capped -- one severely faulty sensor
    # shouldn't alone be indistinguishable from "everything is on fire."
    score += min(20, sensor_fault_penalty)

    return min(100, score)
