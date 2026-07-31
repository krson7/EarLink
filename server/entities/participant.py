from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import (
    DateTime,
    Enum as SqlEnum,
    ForeignKey,
    Integer,
)
from sqlalchemy.orm import Mapped, mapped_column

from server.database import Base


class AccessibilityMode(str, PyEnum):
    """참가자가 사용할 접근성 서비스 유형."""

    VISUAL = "VISUAL"
    HEARING = "HEARING"
    STANDARD = "STANDARD"


class Participant(Base):
    """대화방에 참여한 참가자 정보."""

    __tablename__ = "participants"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    room_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("rooms.id"),
        nullable=False,
        index=True,
    )

    accessibility_mode: Mapped[
        AccessibilityMode
    ] = mapped_column(
        SqlEnum(
            AccessibilityMode,
            name="accessibility_mode",
        ),
        nullable=False,
    )

    joined_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(
            timezone.utc
        ),
    )

    left_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )