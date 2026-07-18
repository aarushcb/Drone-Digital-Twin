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

    while True:
        battery = max(0, battery - random.uniform(0.5, 2))
        altitude = max(0, altitude + random.uniform(-2, 3))

        reading = {
            "latitude": 12.9716 + random.uniform(-0.001, 0.001),
            "longitude": 77.5946 + random.uniform(-0.001, 0.001),
            "altitude": round(altitude, 1),
            "battery": round(battery, 1),
            "speed": round(random.uniform(0, 15), 1),
            "temperature": round(random.uniform(25, 45), 1),
            "roll": round(random.uniform(-10, 10), 1),
            "pitch": round(random.uniform(-10, 10), 1),
            "yaw": round(random.uniform(0, 360), 1),
            "flight_state": "flying",
        }

        post(f"/drones/{drone_id}/telemetry", reading, token=token)
        print(f"Sent: battery={reading['battery']}%  altitude={reading['altitude']}m")

        time.sleep(3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
