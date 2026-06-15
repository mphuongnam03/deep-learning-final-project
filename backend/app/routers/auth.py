from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.core.security import create_access_token, hash_password, verify_password
from backend.app.db.models import User
from backend.app.db.session import get_db
from backend.app.routers.deps import get_current_user
from backend.app.schemas import TokenResponse, UserCreate, UserLogin, UserRead


router = APIRouter(prefix="/auth", tags=["auth"])


def _user_read(user: User) -> UserRead:
    return UserRead(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
    )


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    existing = db.query(User).filter(User.email == payload.email.lower()).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    password_hash, password_salt = hash_password(payload.password)
    role = "admin" if db.query(User).count() == 0 else "student"
    user = User(
        email=payload.email.lower(),
        full_name=payload.full_name.strip(),
        password_hash=password_hash,
        password_salt=password_salt,
        role=role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(str(user.id), settings.jwt_secret_key, settings.jwt_expire_minutes)
    return TokenResponse(access_token=token, user=_user_read(user))


@router.post("/login", response_model=TokenResponse)
def login(payload: UserLogin, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    user = db.query(User).filter(User.email == payload.email.lower(), User.is_active.is_(True)).first()
    if not user or not verify_password(payload.password, user.password_hash, user.password_salt):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    token = create_access_token(str(user.id), settings.jwt_secret_key, settings.jwt_expire_minutes)
    return TokenResponse(access_token=token, user=_user_read(user))


@router.get("/me", response_model=UserRead)
def me(current_user: User = Depends(get_current_user)):
    return _user_read(current_user)


@router.post("/logout")
def logout():
    return {"message": "Client should discard the access token"}
