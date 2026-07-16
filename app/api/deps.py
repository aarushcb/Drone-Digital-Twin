"""
WHY THIS FILE EXISTS:
This is the piece that turns "here's a token" into "here's the actual User
row in the database." Every protected endpoint in routes.py depends on
get_current_user — FastAPI runs it automatically before the endpoint code,
and if the token is missing/invalid/expired, it rejects the request with a
401 before your endpoint logic ever runs. This is the standard FastAPI auth
pattern (OAuth2PasswordBearer + a dependency), not something custom.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.core.security import decode_access_token
from app.crud import user as user_crud
from app.models.user import User

# tokenUrl tells FastAPI's auto-generated docs (/docs) where to send a
# login request to get a token — purely for the interactive docs UI.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    email = decode_access_token(token)
    if email is None:
        raise credentials_exception

    user = user_crud.get_user_by_email(db, email)
    if user is None:
        raise credentials_exception

    return user
