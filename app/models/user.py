"""
WHY THIS IS NEW:
Your original schema had no concept of a "user" at all — any drone/telemetry
was globally visible to anyone who called the API. That's fine for a solo
local project, but a Play Store app will have many different people, each
with their own drones, who should only see their own data. This is the
minimum needed for that: an account a person logs into, and drones get
tied to the account that registered them (see Drone.owner_id below).
"""

from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from app.database.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    drones = relationship("Drone", back_populates="owner", cascade="all, delete-orphan")
