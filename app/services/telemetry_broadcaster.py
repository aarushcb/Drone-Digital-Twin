"""
WHY THIS FILE EXISTS:
A normal REST request (like GET /drones/{id}/telemetry) is "ask once, get
one answer" -- the server has no way to push new data to you after that.
For a live dashboard, we need the opposite: the server should tell the app
immediately when new telemetry arrives, without the app having to keep
asking.

This is what a WebSocket is for -- a connection that stays open, so the
server can push messages whenever it wants. This class just keeps track of
"which open connections are watching which drone," so that when new
telemetry is saved for drone #5, we know exactly which open connections to
push it to.

This is in-memory (a plain Python dict), which is fine for a single-server
setup like your current Render deployment. If this app ever ran on multiple
server instances at once, this would need to move to something shared
between them (e.g. Redis) -- not a concern at your current scale.

DUPLICATE TELEMETRY -- ROOT CAUSE FOUND AND FIXED:
This was the real cause of the "telemetry occasionally arrives twice, same
timestamp" issue: it was never a server-side double-broadcast (routes.py
calls manager.broadcast() exactly once per POST -- confirmed by reading
every call site) and it was never duplicate event triggers. It was
Flutter's TelemetrySocketService.connect() being called a SECOND time for
the same drone (e.g. every pull-to-refresh on DroneDetailScreen re-runs
_loadInitialData(), which unconditionally calls _startLiveUpdates() ->
connect() again) while a live connection already existed -- the old
WebSocketChannel was silently overwritten client-side, never closed, so
it stayed registered here too. From that point on, every broadcast went
out over BOTH connections: the literal same message, sent twice, with an
identical timestamp -- exactly the reported symptom. Fixed at the source
in TelemetrySocketService.connect() (now closes any existing connection
before opening a new one).

connect() below is the SERVER-SIDE hardening on top of that real fix:
defense in depth against this exact failure mode recurring for any
reason (a future client bug, an app killed before it can cleanly
disconnect, a client library that reconnects internally without our
Flutter code knowing) -- if a NEW connection arrives for a (drone_id,
user_id) that already has one registered, the OLD one is closed and
replaced rather than accumulated alongside it. TRADEOFF, documented
honestly: this caps each account to ONE live connection per drone -- if
the same account legitimately watched the same drone's live view from
two devices at once, opening the second would close the first. Not a
concern at this app's current single-tester-single-device scale; flagged
here in case that changes.
"""

from fastapi import WebSocket
from typing import Dict, List, Tuple


class ConnectionManager:
    def __init__(self):
        # Keyed by drone_id -> list of (user_id, websocket) pairs, so a
        # reconnect from the SAME user for the SAME drone can be detected
        # and the stale entry replaced (see connect() below and the
        # module docstring's "DUPLICATE TELEMETRY" section for why this
        # matters).
        self.active_connections: Dict[int, List[Tuple[int, WebSocket]]] = {}

    async def connect(self, drone_id: int, user_id: int, websocket: WebSocket):
        existing = self.active_connections.setdefault(drone_id, [])
        for entry in list(existing):
            existing_user_id, existing_ws = entry
            if existing_user_id == user_id:
                try:
                    # 4409: custom app-defined close code, "replaced by a
                    # newer connection" -- mirrors HTTP 409 Conflict's
                    # meaning, not a standard WebSocket close code.
                    await existing_ws.close(code=4409)
                except Exception:
                    pass  # already gone -- fine, that's what we wanted anyway
                existing.remove(entry)

        await websocket.accept()
        existing.append((user_id, websocket))

    def disconnect(self, drone_id: int, websocket: WebSocket):
        connections = self.active_connections.get(drone_id, [])
        self.active_connections[drone_id] = [
            (uid, ws) for uid, ws in connections if ws is not websocket
        ]
        if not self.active_connections.get(drone_id) and drone_id in self.active_connections:
            del self.active_connections[drone_id]

    async def broadcast(self, drone_id: int, message: dict):
        connections = self.active_connections.get(drone_id, [])
        dead_connections = []
        for entry in connections:
            _user_id, connection = entry
            try:
                await connection.send_json(message)
            except Exception:
                # If sending fails, the connection is almost certainly
                # already gone (app closed, phone lost signal, etc.) --
                # mark it for cleanup rather than letting it crash the loop.
                dead_connections.append(entry)
        for entry in dead_connections:
            connections.remove(entry)


# One shared instance the whole app uses -- see app/api/routes.py (which
# broadcasts into it) and app/api/ws.py (which registers listeners into it).
manager = ConnectionManager()
