def generate_alerts(battery, temperature):
    alerts = []

    if battery < 20:
        alerts.append("LOW BATTERY")

    if temperature > 80:
        alerts.append("OVERHEATING")

    return alerts
