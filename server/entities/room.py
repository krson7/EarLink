from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column

from server.database import Base


class Room(Base):
    """이어링크 대화방 정보."""

    __tablename__ = "rooms"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    room_code: Mapped[str] = mapped_column(
        String(6),
        unique=True,
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="OPEN",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(
            timezone.utc
        ),
    )

    closed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )