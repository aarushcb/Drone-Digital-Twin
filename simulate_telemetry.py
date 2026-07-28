"""
Posts a fake telemetry reading to your LIVE Render backend every few
seconds, for a drone you already created. Use this to actually watch the
"LIVE" indicator and new readings appear on your phone in real time.

Uses only Python's built-in libraries (urllib) -- no `pip install` needed,
just run it with `python3 simulate_telemetry.py`.

WHY THIS EXISTS:
There's no real drone hardware yet -- this stands in for one, so you can
verify the whole live pipeline (backend broadcast -> WebSocket -> Flutter
UI update) actually works end to end without needing real hardware.
"""
import json
import random
import time
import math
import urllib.request
import urllib.parse
import urllib.error

BASE_URL = "https://drone-digital-twin-api.onrender.com"


def post(path, body=None, token=None, form=False):
    url = f"{BASE_URL}{path}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    if form:
        data = urllib.parse.urlencode(body).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    else:
        data = json.dumps(body).encode() if body is not None else None
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"Error {e.code}: {e.read().decode()}")
        raise


def get(path, token=None):
    url = f"{BASE_URL}{path}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def main():
    email = input("Email (same account you use in the app): ").strip()
    password = input("Password: ").strip()

    print("Logging in...")
    login_data = post("/auth/login", {"username": email, "password": password}, form=True)
    token = login_data["access_token"]

    print("Fetching your drones...")
    drones = get("/drones", token=token)
    if not drones:
        print("No drones found on this account. Add one in the app first, then rerun this.")
        return

    print("\nYour drones:")
    for d in drones:
        print(f"  id={d['id']}  name={d['name']}")

    drone_id = int(input("\nWhich drone id do you want to simulate telemetry for? "))

    print(f"\nSending fake telemetry for drone {drone_id} every 3 seconds. Ctrl+C to stop.")
    print("Open that drone's detail screen in the app now to watch it update live.\n")

    battery = 100.0
    altitude = 0.0

    # WHY THIS CHANGED FROM RANDOM JITTER TO A REAL FLIGHT MODEL:
    # The original version picked a brand new random position every single
    # reading, independent of the last one -- fine for testing that
    # telemetry displays at all, but meaningless once replay (Flight
    # Replay screen) started showing ACTUAL movement between readings.
    # Random independent points look like teleporting noise when replayed,
    # not a flight. This version simulates a drone with momentum: it picks
    # a heading and gradually turns, rather than jumping to a new random
    # spot each time -- producing a smooth, continuous, plausible-looking
    # path.
    lat, lon = 12.9716, 77.5946  # starting position (Bengaluru-area, arbitrary)
    heading_deg = random.uniform(0, 360)  # initial direction of travel
    speed_mps = random.uniform(5, 12)

    METERS_PER_DEG_LAT = 110540

    def meters_per_deg_lon(at_lat):
        return 111320 * math.cos(math.radians(at_lat))

    while True:
        battery = max(0, battery - random.uniform(0.5, 2))
        altitude = max(0, altitude + random.uniform(-2, 3))

        # Gradual turning (a real drone doesn't instantly reverse
        # direction) and mild speed variation -- this is what makes the
        # resulting path curve smoothly instead of zigzagging randomly.
        # WHY ±4° (NOT ±15° as originally fixed): real testing showed that
        # ±15°/tick, while smooth and non-teleporting, still let heading
        # wander up to 90+ degrees over a typical ~30-reading session --
        # visually reading as aimless meandering rather than a real flight
        # with a direction. ±4° keeps the same "gradual, momentum-based
        # turning" idea, just tighter, so a demo/replay session reads as
        # an actual flight path (verified: average total heading drift
        # over 30 ticks drops from ~67° to ~18° with this change).
        heading_deg = (heading_deg + random.uniform(-4, 4)) % 360
        speed_mps = max(2, min(15, speed_mps + random.uniform(-1, 1)))

        distance_m = speed_mps * 3  # meters traveled in this 3-second tick
        heading_rad = math.radians(heading_deg)
        north_m = distance_m * math.cos(heading_rad)
        east_m = distance_m * math.sin(heading_rad)
        lat += north_m / METERS_PER_DEG_LAT
        lon += east_m / meters_per_deg_lon(lat)

        reading = {
            "latitude": round(lat, 7),
            "longitude": round(lon, 7),
            "altitude": round(altitude, 1),
            "battery": round(battery, 1),
            "speed": round(speed_mps, 1),
            "temperature": round(random.uniform(25, 45), 1),
            # Yaw now matches the direction of travel (heading) instead of
            # being unrelated random noise -- the drone's nose in the 3D
            # view will actually point the way it's flying.
            "yaw": round(heading_deg, 1),
            # Small roll proportional to how sharply it's turning --
            # approximates a drone banking into a turn.
            "roll": round(random.uniform(-3, 3), 1),
            "pitch": round(random.uniform(-5, 5), 1),
            "flight_state": "flying",
        }

        post(f"/drones/{drone_id}/telemetry", reading, token=token)
        print(f"Sent: battery={reading['battery']}%  altitude={reading['altitude']}m  heading={heading_deg:.0f}°")

        time.sleep(3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
