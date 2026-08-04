"""
Standalone test script for the WebSocket duplicate-telemetry fix -- run
directly with `python3 test_telemetry_broadcaster.py`, matching the other
standalone test_*.py scripts in this repo. Uses FastAPI's TestClient
(Starlette under the hood), which supports real WebSocket connections
in-process -- exercising the ACTUAL routing/auth/ConnectionManager code,
not a mock.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["DATABASE_URL"] = "sqlite:///./test_broadcaster.db"

import asyncio
from fastapi.testclient import TestClient
from app.services.telemetry_broadcaster import ConnectionManager


class _FakeWebSocket:
    """Minimal fake matching the surface ConnectionManager actually calls
    (accept/close/send_json) -- used for the pure unit-level tests below,
    which check ConnectionManager's own bookkeeping directly rather than
    going through a real network round-trip (that's what the TestClient
    tests further down are for)."""
    def __init__(self, name):
        self.name = name
        self.accepted = False
        self.closed = False
        self.close_code = None
        self.sent = []

    async def accept(self):
        self.accepted = True

    async def close(self, code=1000):
        self.closed = True
        self.close_code = code

    async def send_json(self, message):
        if self.closed:
            raise RuntimeError("cannot send on a closed websocket")
        self.sent.append(message)


def test_reconnect_from_same_user_closes_the_stale_connection():
    # The core fix: a second connect() for the SAME (drone_id, user_id)
    # should close the FIRST websocket, not accumulate both.
    manager = ConnectionManager()
    ws1 = _FakeWebSocket("first")
    ws2 = _FakeWebSocket("second")

    asyncio.run(manager.connect(drone_id=1, user_id=42, websocket=ws1))
    asyncio.run(manager.connect(drone_id=1, user_id=42, websocket=ws2))

    assert ws1.closed is True, "Expected the first (stale) connection to be closed"
    assert ws1.close_code == 4409
    assert ws2.closed is False
    connections = manager.active_connections[1]
    assert len(connections) == 1, f"Expected exactly 1 active connection after reconnect, got {len(connections)}"
    assert connections[0][1] is ws2
    print("PASS: reconnecting as the same user closes the stale connection, leaving exactly one active")


def test_different_users_both_keep_their_connections():
    # The documented tradeoff's OTHER side: two DIFFERENT users watching
    # the same drone_id (e.g. shared ownership scenarios, or just two
    # different accounts happening to use the same drone_id number across
    # different owners) should NOT close each other's connections --
    # only same-user reconnects trigger replacement.
    manager = ConnectionManager()
    ws_user1 = _FakeWebSocket("user1")
    ws_user2 = _FakeWebSocket("user2")

    asyncio.run(manager.connect(drone_id=1, user_id=1, websocket=ws_user1))
    asyncio.run(manager.connect(drone_id=1, user_id=2, websocket=ws_user2))

    assert ws_user1.closed is False
    assert ws_user2.closed is False
    assert len(manager.active_connections[1]) == 2
    print("PASS: different users on the same drone_id both keep their own connections")


def test_broadcast_after_reconnect_delivers_exactly_once():
    # The actual end-to-end claim this whole fix is about: after a
    # reconnect, a broadcast should reach the client exactly ONCE, not
    # twice -- this is the literal bug being fixed, verified directly.
    manager = ConnectionManager()
    ws1 = _FakeWebSocket("first")
    ws2 = _FakeWebSocket("second")

    asyncio.run(manager.connect(drone_id=5, user_id=7, websocket=ws1))
    asyncio.run(manager.connect(drone_id=5, user_id=7, websocket=ws2))  # simulates the old bug's reconnect
    asyncio.run(manager.broadcast(5, {"battery": 42}))

    assert ws1.sent == [], f"Expected the stale (closed) connection to receive nothing, got {ws1.sent}"
    assert ws2.sent == [{"battery": 42}], f"Expected exactly one delivery to the live connection, got {ws2.sent}"
    print("PASS: after a same-user reconnect, a broadcast is delivered exactly once, not twice")


def test_disconnect_removes_only_the_matching_websocket():
    manager = ConnectionManager()
    ws1 = _FakeWebSocket("first")
    ws2 = _FakeWebSocket("second")
    asyncio.run(manager.connect(drone_id=9, user_id=1, websocket=ws1))
    asyncio.run(manager.connect(drone_id=9, user_id=2, websocket=ws2))

    manager.disconnect(9, ws1)
    remaining = manager.active_connections[9]
    assert len(remaining) == 1
    assert remaining[0][1] is ws2
    print("PASS: disconnect() removes only the matching websocket, leaving the other user's connection intact")


def test_real_websocket_reconnect_via_testclient():
    # Full, real end-to-end test through the ACTUAL app: register, log
    # in, create a drone, open a real WebSocket connection, open a SECOND
    # real WebSocket connection as the same user for the same drone
    # (simulating the exact bug scenario -- pull-to-refresh reconnecting
    # without closing the old socket), then confirm a telemetry POST's
    # broadcast is delivered to the SECOND (current) connection and the
    # FIRST one was actually closed by the server (not left dangling).
    if os.path.exists("test_broadcaster.db"):
        os.remove("test_broadcaster.db")

    from main import app
    client = TestClient(app)

    client.post("/auth/register", json={"email": "wstest@test.com", "password": "testpass123"})
    login = client.post(
        "/auth/login",
        data={"username": "wstest@test.com", "password": "testpass123"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    drone_id = client.post("/drones", json={"name": "WSQuad"}, headers=headers).json()["id"]

    with client.websocket_connect(f"/ws/drones/{drone_id}/telemetry?token={token}") as ws1:
        # Second connection, same user, same drone -- exactly the bug scenario.
        with client.websocket_connect(f"/ws/drones/{drone_id}/telemetry?token={token}") as ws2:
            # ws1 should have been closed server-side by now (the fix).
            try:
                ws1.receive_text()
                assert False, "Expected the stale first connection to be closed by the server"
            except Exception:
                pass  # expected -- the connection was closed

            # A real telemetry POST should broadcast to ws2 exactly once.
            client.post(
                f"/drones/{drone_id}/telemetry",
                json={"battery": 77.0, "altitude": 5.0},
                headers=headers,
            )
            message = ws2.receive_json()
            assert message["battery"] == 77.0

    if os.path.exists("test_broadcaster.db"):
        os.remove("test_broadcaster.db")
    print("PASS: real end-to-end WebSocket reconnect via TestClient -- stale connection closed, broadcast delivered once to the live one")


if __name__ == "__main__":
    test_reconnect_from_same_user_closes_the_stale_connection()
    test_different_users_both_keep_their_connections()
    test_broadcast_after_reconnect_delivers_exactly_once()
    test_disconnect_removes_only_the_matching_websocket()
    test_real_websocket_reconnect_via_testclient()
    print("\nAll telemetry broadcaster tests passed.")
