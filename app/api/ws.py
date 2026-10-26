"""
WHY AUTH WORKS DIFFERENTLY HERE THAN IN routes.py:
Every other endpoint reads the "Authorization: Bearer <token>" header.
Browsers and Flutter's web_socket_channel can't set that header on a
WebSocket handshake, but they can set Sec-WebSocket-Protocol. So the client
offers two subprotocols, "bearer" and the token itself:

    Sec-WebSocket-Protocol: bearer, <token>

and the server accepts by echoing only "bearer". The token used to travel in
the URL (?token=...), which put a 7-day credential into the hosting
provider's request logs, since the request line is logged and headers
aren't. A short-lived ticket fetched over HTTPS first was the alternative,
but it needs server-side state and an extra round trip, and still puts a
(short-lived) credential in the logged URL.

WHY WE MANUALLY CHECK OWNERSHIP HERE TOO:
Same reasoning as every REST endpoint -- a user should only be able to
listen to telemetry for a drone they actually own. We do this check once,
right when the connection opens, before accepting it.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.database.database import SessionLocal
from app.core.security import decode_access_token
from app.crud import user as user_crud
from app.crud import drone as drone_crud
from app.services.telemetry_broadcaster import manager

router = APIRouter()

AUTH_SUBPROTOCOL = "bearer"


def token_from_subprotocols(websocket: WebSocket) -> str | None:
    """The token offered as the subprotocol after "bearer", or None."""
    offered = list(websocket.scope.get("subprotocols") or [])
    if AUTH_SUBPROTOCOL in offered:
        i = offered.index(AUTH_SUBPROTOCOL)
        if i + 1 < len(offered):
            return offered[i + 1]
    return None


@router.websocket("/ws/drones/{drone_id}/telemetry")
async def telemetry_websocket(websocket: WebSocket, drone_id: int):
    token = token_from_subprotocols(websocket)
    db = SessionLocal()
    try:
        email = decode_access_token(token) if token else None
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

    await manager.connect(drone_id, user.id, websocket, subprotocol=AUTH_SUBPROTOCOL)
    try:
        while True:
            # We don't expect the client to send anything -- this just
            # keeps the connection open and lets us detect a disconnect
            # (e.g. the app closing, phone locking, wifi dropping).
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(drone_id, websocket)
