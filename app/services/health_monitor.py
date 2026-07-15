def get_health_status(battery, temperature):
    if battery < 20:
        return "CRITICAL"

    if temperature > 80:
        return "OVERHEATING"

    if battery < 40:
        return "WARNING"

    return "GOOD"
