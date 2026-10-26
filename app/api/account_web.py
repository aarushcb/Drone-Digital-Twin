"""
Public account-deletion page -- Google Play requires apps with account
creation to offer deletion at a web URL reachable without installing the app:

  GET  /account/delete   a plain HTML form (email + password)
  POST /account/delete   deletes the account, same as DELETE /auth/account

Asks for the password, exactly like signing in, and says plainly what is lost.
No JavaScript, no external assets: it has to work from any browser.
"""

from html import escape

from fastapi import APIRouter, Depends, Form
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.core.security import verify_password
from app.crud import user as user_crud
from app.database.database import get_db

router = APIRouter(prefix="/account", tags=["account"])

_STYLE = ("body{font-family:system-ui,sans-serif;max-width:34rem;margin:2rem auto;padding:0 1rem;"
          "background:#FBF8F3;color:#24242b;line-height:1.5}input,button{font:inherit;padding:.5rem;width:100%;"
          "box-sizing:border-box;margin:.25rem 0 .75rem}button{background:#24242b;color:#fff;border:0;border-radius:6px}"
          ".warn{background:#FFE3E5;padding:.75rem;border-radius:6px}")

_LOST = ("your account and sign-in, every drone, its telemetry history and 3D scene objects, every uploaded flight "
         "and its health report, the measurements taken from each log, your drones' learned baselines, the changes "
         "you logged and your battery packs")


def _page(body: str, status: int = 200) -> HTMLResponse:
    return HTMLResponse(f"<!doctype html><html lang=en><head><meta charset=utf-8>"
                        f"<meta name=viewport content='width=device-width,initial-scale=1'>"
                        f"<title>Delete your DroneTwinLabs account</title><style>{_STYLE}</style></head>"
                        f"<body><h1>Delete your DroneTwinLabs account</h1>{body}</body></html>", status_code=status)


@router.get("/delete", response_class=HTMLResponse)
def delete_form():
    return _page(f"""<p>This deletes {_LOST}. It happens immediately and cannot be undone.</p>
<p>Raw flight log files are never kept: each one is deleted from the server as soon as it has been analysed.</p>
<form method=post action="/account/delete">
<label>Email<input name=email type=email required autocomplete=username></label>
<label>Password<input name=password type=password required autocomplete=current-password></label>
<p class=warn>There is no way to recover a deleted account.</p>
<button type=submit>Delete my account and all its data</button>
</form>
<p>You can also delete your account inside the app: Settings, then Delete account.</p>""")


@router.post("/delete", response_class=HTMLResponse)
def delete_submit(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = user_crud.get_user_by_email(db, email)
    if user is None or not verify_password(password, user.hashed_password):
        return _page("<p class=warn>That email and password don't match an account. Nothing was deleted.</p>"
                     "<p><a href='/account/delete'>Try again</a></p>", status=401)
    deleted = user_crud.delete_account(db, user)
    rows = "".join(f"<li>{escape(k.replace('_', ' '))}: {v}</li>" for k, v in deleted.items() if k != "user")
    return _page(f"<p>The account <b>{escape(email)}</b> and all its data have been deleted.</p><ul>{rows}</ul>")
