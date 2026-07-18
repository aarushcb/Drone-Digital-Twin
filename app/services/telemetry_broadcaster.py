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
"""

from fastapi import WebSocket
from typing import Dict, List


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, List[WebSocket]] = {}

    async def connect(self, drone_id: int, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.setdefault(drone_id, []).append(websocket)

    def disconnect(self, drone_id: int, websocket: WebSocket):
        connections = self.active_connections.get(drone_id, [])
        if websocket in connections:
            connections.remove(websocket)
        if not connections and drone_id in self.active_connections:
            del self.active_connections[drone_id]

    async def broadcast(self, drone_id: int, message: dict):
        connections = self.active_connections.get(drone_id, [])
        dead_connections = []
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception:
                # If sending fails, the connection is almost certainly
                # already gone (app closed, phone lost signal, etc.) --
                # mark it for cleanup rather than letting it crash the loop.
                dead_connections.append(connection)
        for connection in dead_connections:
            connections.remove(connection)


# One shared instance the whole app uses -- see app/api/routes.py (which
# broadcasts into it) and app/api/ws.py (which registers listeners into it).
manager = ConnectionManager()
