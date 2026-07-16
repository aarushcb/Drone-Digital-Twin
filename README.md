# Drone Digital Twin — Backend (v2)

This replaces the `app/` folder and `main.py` in your existing repo. It's
the same foundation, expanded with: user accounts + login, drone physical
specs, telemetry orientation/flight-state fields, and owner-scoped data
(each user only sees their own drones). I ran this exact code end-to-end
before sending it to you — register → login → create drone → post telemetry
→ check status → confirm a second user is blocked from seeing the first
user's drone. All of it passed.

## 1. Where these files go

In your existing `Drone-Digital-Twin` repo, **replace**:
- `app/` (the whole folder)
- `main.py`
- `requirements.txt` (if you had one)

**Add** (new files, didn't exist before):
- `.env.example`
- `.gitignore`
- `README.md` (this file, optional to keep)

The empty `backend/`, `frontend/`, `mobile/`, `docs/` placeholder folders
in your repo are untouched — leave them for now, we'll fill `mobile/` when
we start the Flutter app.

## 2. First-time setup (run once)

```bash
cd Drone-Digital-Twin
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env
```

*Why a virtual env:* keeps this project's Python packages separate from
anything else on your Mac — standard practice, avoids version conflicts
between different projects.

The default `.env` uses a local SQLite file, so there's nothing else to
install or configure to run this locally — no Postgres needed on your Mac.

## 3. Run the server

```bash
venv/bin/uvicorn main:app --reload
```

`--reload` restarts the server automatically whenever you save a code
change — useful while developing. Visit `http://127.0.0.1:8000/docs` in
your browser — FastAPI auto-generates an interactive UI for every endpoint,
including a working "Authorize" button once you've logged in.

## 4. Quick manual test (mirrors what I already verified)

```bash
# Register
curl -X POST http://127.0.0.1:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"yourpassword"}'

# Login (note: form data, not JSON — this is intentional, see auth.py comments)
curl -X POST http://127.0.0.1:8000/auth/login \
  -d "username=you@example.com&password=yourpassword"
# copy the access_token from the response for the next steps

# Create a drone (replace YOUR_TOKEN)
curl -X POST http://127.0.0.1:8000/drones \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"Mark I","model":"Quad-250","mass_kg":1.2,"frame_type":"quadcopter","motor_count":4}'
```

## 5. What's intentionally NOT done yet (next steps, in order)

1. **Real cloud database** — currently SQLite (local file). Before the
   Flutter app can talk to this from a phone, it needs to be deployed
   somewhere reachable over the internet (Supabase/Neon Postgres +
   Render/Railway hosting). That's the next piece of work.
2. **Alembic migrations** — right now tables auto-create from the models
   on startup, which is fine for active local development. Once there's
   real data in a production database, further schema changes should go
   through proper migrations instead of `create_all`, to avoid data loss.
3. **WebSocket endpoint** — for live telemetry push to the dashboard/3D
   view instead of polling.
4. **Replace threshold-based analytics** (`services/drone_analytics.py`,
   `services/health_monitor.py`) with real predictive modeling once there's
   enough historical telemetry to train on.

## 6. Endpoint summary

| Method | Path | Auth required | Notes |
|---|---|---|---|
| POST | /auth/register | No | Create account |
| POST | /auth/login | No | Returns access token |
| POST | /drones | Yes | Create a drone |
| GET | /drones | Yes | List your drones |
| GET | /drones/{id} | Yes | Get one drone |
| PATCH | /drones/{id} | Yes | Update specs/status |
| DELETE | /drones/{id} | Yes | Delete a drone |
| POST | /drones/{id}/telemetry | Yes | Log a telemetry reading |
| GET | /drones/{id}/telemetry | Yes | History (supports `limit`, `offset`, `start`, `end`) |
| GET | /drones/{id}/status | Yes | Current health/alerts/risk score |
| GET | /drones/{id}/analytics | Yes | Aggregate stats over recent history |
