from sqlalchemy import select
from sqlalchemy.exc import (
    IntegrityError,
    SQLAlchemyError,
)
from sqlalchemy.orm import Session

from server.entities.participant import (
    AccessibilityMode,
    Participant,
)
from server.entities.room import Room


# 이어링크에서 항상 사용하는 고정 대화방 코드
DEFAULT_ROOM_CODE = "EAR001"


class InvalidRoomCodeError(ValueError):
    """대화방 코드 형식이 올바르지 않을 때 발생한다."""


class RoomNotFoundError(ValueError):
    """대화방을 찾을 수 없을 때 발생한다."""


class RoomClosedError(ValueError):
    """종료된 대화방에 참가하려 할 때 발생한다."""


class ParticipantNotInRoomError(ValueError):
    """참가자가 해당 방에 속하지 않을 때 발생한다."""


class RoomCreationError(RuntimeError):
    """고정 대화방 생성에 실패했을 때 발생한다."""


class ParticipantCreationError(RuntimeError):
    """참가자 생성에 실패했을 때 발생한다."""


def normalize_room_code(
    room_code: str,
) -> str:
    """대화방 코드를 공백 제거 및 대문자로 정규화한다."""

    normalized_room_code = (
        room_code.strip().upper()
    )

    if not normalized_room_code:
        raise InvalidRoomCodeError(
            "대화방 코드를 입력해주세요."
        )

    return normalized_room_code


def get_room_by_code(
    db: Session,
    room_code: str,
) -> Room | None:
    """대화방 코드로 대화방을 조회한다."""

    normalized_room_code = normalize_room_code(
        room_code
    )

    return db.scalar(
        select(Room).where(
            Room.room_code
            == normalized_room_code
        )
    )


def ensure_default_room(
    db: Session,
) -> Room:
    """
    고정 대화방 EAR001이 존재하는지 확인한다.

    방이 없으면 새로 생성하고,
    이미 존재하면 기존 방을 그대로 반환한다.
    """

    existing_room = get_room_by_code(
        db=db,
        room_code=DEFAULT_ROOM_CODE,
    )

    if existing_room is not None:
        # 고정 방이 닫힌 상태라면 다시 열린 상태로 복구한다.
        if existing_room.status != "OPEN":
            existing_room.status = "OPEN"
            existing_room.closed_at = None

            try:
                db.commit()
                db.refresh(existing_room)

            except SQLAlchemyError as error:
                db.rollback()

                raise RoomCreationError(
                    "고정 대화방 상태를 변경하지 못했습니다."
                ) from error

        return existing_room

    default_room = Room(
        room_code=DEFAULT_ROOM_CODE,
        status="OPEN",
    )

    db.add(default_room)

    try:
        db.commit()
        db.refresh(default_room)

        return default_room

    except IntegrityError as error:
        # 여러 서버 프로세스가 동시에 실행되면서
        # 다른 프로세스가 먼저 방을 만든 경우를 처리한다.
        db.rollback()

        existing_room = get_room_by_code(
            db=db,
            room_code=DEFAULT_ROOM_CODE,
        )

        if existing_room is not None:
            return existing_room

        raise RoomCreationError(
            "고정 대화방 생성 중 중복 오류가 발생했습니다."
        ) from error

    except SQLAlchemyError as error:
        db.rollback()

        raise RoomCreationError(
            "고정 대화방 생성 중 데이터베이스 오류가 "
            "발생했습니다."
        ) from error


def join_room(
    db: Session,
    room_code: str,
    accessibility_mode: AccessibilityMode,
) -> Participant:
    """지정한 대화방에 참가자를 추가한다."""

    normalized_room_code = normalize_room_code(
        room_code
    )

    room = get_room_by_code(
        db=db,
        room_code=normalized_room_code,
    )

    if room is None:
        raise RoomNotFoundError(
            "존재하지 않는 대화방입니다."
        )

    if room.status != "OPEN":
        raise RoomClosedError(
            "현재 입장할 수 없는 대화방입니다."
        )

    participant = Participant(
        room_id=room.id,
        accessibility_mode=accessibility_mode,
    )

    db.add(participant)

    try:
        db.commit()
        db.refresh(participant)

        return participant

    except SQLAlchemyError as error:
        db.rollback()

        raise ParticipantCreationError(
            "대화방 참여 처리 중 데이터베이스 오류가 "
            "발생했습니다."
        ) from error


def join_default_room(
    db: Session,
    accessibility_mode: AccessibilityMode,
) -> Participant:
    """이어링크 고정 대화방 EAR001에 참가자를 추가한다."""

    return join_room(
        db=db,
        room_code=DEFAULT_ROOM_CODE,
        accessibility_mode=accessibility_mode,
    )


def get_participant_in_room(
    db: Session,
    room_code: str,
    participant_id: int,
) -> Participant:
    """
    참가자가 해당 대화방에 실제로 속해 있는지 확인한다.

    참가자가 존재하지 않거나 다른 방 소속이면
    ParticipantNotInRoomError를 발생시킨다.
    """

    normalized_room_code = normalize_room_code(
        room_code
    )

    participant = db.scalar(
        select(Participant)
        .join(
            Room,
            Participant.room_id == Room.id,
        )
        .where(
            Participant.id == participant_id,
            Room.room_code
            == normalized_room_code,
        )
    )

    if participant is None:
        raise ParticipantNotInRoomError(
            "해당 대화방에 참여한 사용자가 아닙니다."
        )

    return participant