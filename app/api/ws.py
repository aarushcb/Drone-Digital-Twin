"""
WHY AUTH WORKS DIFFERENTLY HERE THAN IN routes.py:
Every other endpoint uses get_current_user (app/api/deps.py), which reads
the "Authorization: Bearer <token>" HEADER. WebSocket connections from most
clients (including Flutter's web_socket_channel) don't let you attach custom
headers before the connection is even open -- so instead, the token is
passed as a URL query parameter instead: wss://.../ws/drones/5/telemetry?token=xyz.
This is a very common, accepted pattern for WebSocket auth specifically.

WHY WE MANUALLY CHECK OWNERSHIP HERE TOO:
Same reasoning as every REST endpoint -- a user should only be able to
listen to telemetry for a drone they actually own. We do this check once,
right when the connection opens, before accepting it.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from app.database.database import SessionLocal
from app.core.security import decode_access_token
from app.crud import user as user_crud
from app.crud import drone as drone_crud
from app.services.telemetry_broadcaster import manager

router = APIRouter()


@router.websocket("/ws/drones/{drone_id}/telemetry")
async def telemetry_websocket(websocket: WebSocket, drone_id: int, token: str = Query(...)):
    db = SessionLocal()
    try:
        email = decode_access_token(token)
        if email is None:
            await websocket.close(code=4401)  # custom code in the 4xxx app-defined range
            return

        user = user_crud.get_user_by_email(db, email)
        if user is None:
            await websocket.close(code=4401)
            return

        owned_drone = drone_crud.get_drone(db, drone_id, owner_id=user.id)
        if owned_drone is None:
            await websocket.close(code=4404)
            return
    finally:
        db.close()

    await manager.connect(drone_id, user.id, websocket)
    try:
        while True:
            # We don't expect the client to send anything -- this just
            # keeps the connection open and lets us detect a disconnect
            # (e.g. the app closing, phone locking, wifi dropping).
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(drone_id, websocket)
