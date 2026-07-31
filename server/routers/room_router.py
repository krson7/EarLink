from datetime import datetime

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.database import get_db
from server.entities.participant import (
    AccessibilityMode,
)
from server.services.room_service import (
    DEFAULT_ROOM_CODE,
    ParticipantCreationError,
    RoomClosedError,
    RoomNotFoundError,
    join_default_room,
)


router = APIRouter(
    prefix="/rooms",
    tags=["rooms"],
)


class JoinRoomRequest(BaseModel):
    """고정 대화방 참여 요청."""

    accessibility_mode: AccessibilityMode


class JoinRoomResponse(BaseModel):
    """고정 대화방 참여 응답."""

    participant_id: int
    room_id: int
    room_code: str
    accessibility_mode: AccessibilityMode
    joined_at: datetime


@router.post(
    "/join",
    response_model=JoinRoomResponse,
    status_code=status.HTTP_201_CREATED,
)
def join_room_api(
    request: JoinRoomRequest,
    db: Session = Depends(get_db),
) -> JoinRoomResponse:
    """이어링크 고정 대화방에 참가자를 추가한다."""

    try:
        participant = join_default_room(
            db=db,
            accessibility_mode=(
                request.accessibility_mode
            ),
        )

        return JoinRoomResponse(
            participant_id=participant.id,
            room_id=participant.room_id,
            room_code=DEFAULT_ROOM_CODE,
            accessibility_mode=(
                participant.accessibility_mode
            ),
            joined_at=participant.joined_at,
        )

    except RoomNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error

    except RoomClosedError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error

    except ParticipantCreationError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=str(error),
        ) from error