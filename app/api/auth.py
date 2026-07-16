"""
Two endpoints:
- POST /auth/register — create an account
- POST /auth/login — exchange email+password for an access token

WHY LOGIN USES OAuth2PasswordRequestForm (form data) INSTEAD OF JSON:
This isn't a stylistic choice — it's what makes /docs' built-in "Authorize"
button work out of the box, and it's the standard FastAPI/OAuth2 convention
that most client libraries (including Flutter's http/dio packages) expect.
The form field is called "username" even though we're using email — that's
just the OAuth2 spec's naming, we map it to our email field internally.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.schemas.user import UserCreate, UserOut, Token
from app.crud import user as user_crud
from app.core.security import verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(user: UserCreate, db: Session = Depends(get_db)):
    if user_crud.get_user_by_email(db, user.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    return user_crud.create_user(db, user)


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = user_crud.get_user_by_email(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    token = create_access_token(subject=user.email)
    return Token(access_token=token)
