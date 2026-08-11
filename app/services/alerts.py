def generate_alerts(battery, temperature):
    alerts = []

    # Both fields are optional on a telemetry reading (e.g. a MAVLink log
    # import never populates temperature -- that field doesn't exist
    # anywhere in the MAVLink messages app/services/mavlink_import.py
    # parses), so a reading missing one is simply not evaluated for that
    # alert, rather than crashing the whole status endpoint.
    if battery is not None and battery < 20:
        alerts.append("LOW BATTERY")

    if temperature is not None and temperature > 80:
        alerts.append("OVERHEATING")

    return alerts
