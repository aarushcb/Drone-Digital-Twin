def calculate_risk_score(battery, temperature, speed):
    score = 0

    if battery < 40:
        score += 30

    if battery < 20:
        score += 30

    if temperature > 70:
        score += 20

    if temperature > 80:
        score += 20

    if speed > 25:
        score += 20

    return min(score, 100)


def calculate_analytics(records):
    if not records:
        return None

    avg_speed = sum(r.speed for r in records) / len(records)

    avg_temperature = (
        sum(r.temperature for r in records)
        / len(records)
    )

    max_altitude = max(
        r.altitude for r in records
    )

    avg_battery = (
        sum(r.battery for r in records)
        / len(records)
    )

    return {
        "avg_speed": round(avg_speed, 2),
        "avg_temperature": round(avg_temperature, 2),
        "max_altitude": max_altitude,
        "avg_battery": round(avg_battery, 2),
        "telemetry_records": len(records)
    }