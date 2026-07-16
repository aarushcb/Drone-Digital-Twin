"""
WHY THIS FILE EXISTS:
Two separate jobs, both required for a real login system:
1. Passwords must NEVER be stored as plain text. We store a one-way hash
   (bcrypt) instead — even if the database is ever leaked, actual passwords
   aren't recoverable from it.
2. After login, the app needs a way to prove "I'm still the same logged-in
   user" on every future request without sending the password every time.
   That's what a JWT (JSON Web Token) is: a signed, tamper-proof string
   containing the user's identity and an expiry time. The app stores this
   token after login and attaches it to every request.
"""

from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
import jwt

from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> str | None:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        return payload.get("sub")
    except jwt.PyJWTError:
        return None
