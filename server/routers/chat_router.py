import asyncio
import json

from fastapi import (
    APIRouter,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from sqlalchemy.orm import Session

from server.database import engine
from server.services.chat_service import chat_manager
from server.services.room_service import (
    get_participant_in_room,
    normalize_room_code,
)


router = APIRouter(
    tags=["chat"],
)


# 채팅 메시지 최대 길이
MAX_CHAT_MESSAGE_LENGTH = 1000


def validate_chat_participant(
    room_code: str,
    participant_id: int,
) -> None:
    """
    참가자가 실제로 해당 대화방에 속해 있는지 확인한다.

    WebSocket 연결 전체 시간 동안 DB 세션을 유지하지 않고,
    검증할 때만 짧게 세션을 열고 종료한다.
    """

    with Session(bind=engine) as db:
        get_participant_in_room(
            db=db,
            room_code=room_code,
            participant_id=participant_id,
        )


@router.websocket(
    "/ws/rooms/{room_code}/{participant_id}"
)
async def chat_websocket(
    websocket: WebSocket,
    room_code: str,
    participant_id: int,
) -> None:
    """대화방별 실시간 채팅 WebSocket을 처리한다."""

    # 검증 실패 시 브라우저에 실제 WebSocket 종료 코드인
    # 1008을 전달할 수 있도록 먼저 연결을 승인한다.
    await websocket.accept()

    try:
        normalized_room_code = normalize_room_code(
            room_code
        )

        # SQLAlchemy 동기 DB 조회가 이벤트 루프를 막지 않도록
        # 별도 스레드에서 참가자 검증을 실행한다.
        await asyncio.to_thread(
            validate_chat_participant,
            normalized_room_code,
            participant_id,
        )

    except (ValueError, HTTPException) as error:
        detail = (
            error.detail
            if isinstance(error, HTTPException)
            else str(error)
        )

        print(
            "[채팅 연결 거부] "
            f"room={room_code}, "
            f"participant={participant_id}, "
            f"reason={detail}"
        )

        # close reason은 길이 제한이 있으므로
        # 자세한 내용은 JSON 메시지로 먼저 전달한다.
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "code": (
                        "PARTICIPANT_VALIDATION_FAILED"
                    ),
                    "detail": str(detail),
                }
            )
        except Exception:
            pass

        try:
            await websocket.close(
                code=(
                    status.WS_1008_POLICY_VIOLATION
                ),
                reason=(
                    "대화방 또는 참가자 확인 실패"
                ),
            )
        except Exception:
            pass

        return

    except Exception as error:
        print(
            "[채팅 참가자 검증 오류] "
            f"room={room_code}, "
            f"participant={participant_id}, "
            f"error={error}"
        )

        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "code": (
                        "PARTICIPANT_VALIDATION_ERROR"
                    ),
                    "detail": (
                        "참가자 확인 중 서버 오류가 "
                        "발생했습니다."
                    ),
                }
            )
        except Exception:
            pass

        try:
            await websocket.close(
                code=(
                    status.WS_1011_INTERNAL_ERROR
                ),
                reason=(
                    "참가자 확인 중 서버 오류 발생"
                ),
            )
        except Exception:
            pass

        return

    # 아래부터는 검증이 성공한 연결만 실행된다.
    connected = False
    typing_active = False

    try:
        # WebSocket accept는 위에서 이미 처리했으며,
        # connect()는 연결 목록 등록만 담당한다.
        await chat_manager.connect(
            room_code=normalized_room_code,
            participant_id=participant_id,
            websocket=websocket,
        )

        connected = True

        # Clients wait for this before sending control frames. Older clients
        # still send plain text and ignore unknown server event types.
        await websocket.send_json({"type": "capabilities", "braille_events": True, "braille_drafts": True})

        while True:
            content = (
                await websocket.receive_text()
            )

            content = content.strip()
            source = None
            event = None
            if content.startswith("{") and len(content) <= MAX_CHAT_MESSAGE_LENGTH * 6 + 256:
                try:
                    event = json.loads(content)
                except (ValueError, RecursionError):
                    pass

            if isinstance(event, dict) and event.get("type") == "typing":
                if event.get("source") != "braille" or not isinstance(event.get("active"), bool):
                    await websocket.send_json({
                        "type": "error", "code": "INVALID_TYPING_EVENT",
                        "detail": "점자 입력 상태 형식이 올바르지 않습니다.",
                    })
                    continue
                draft = event.get("content")
                if draft is not None and (not isinstance(draft, str) or len(draft) > MAX_CHAT_MESSAGE_LENGTH):
                    await websocket.send_json({
                        "type": "error", "code": "INVALID_TYPING_EVENT",
                        "detail": "점자 입력 내용은 최대 1000자의 문자열이어야 합니다.",
                    })
                    continue
                typing_active = event["active"]
                await chat_manager.broadcast(
                    room_code=normalized_room_code,
                    message={
                        "type": "typing", "source": "braille",
                        "sender_id": participant_id, "active": typing_active,
                        **({"content": draft if typing_active else ""} if draft is not None else {}),
                    },
                )
                continue

            if (isinstance(event, dict) and event.get("type") == "message"
                    and event.get("source") == "braille"):
                if not isinstance(event.get("content"), str):
                    await websocket.send_json({
                        "type": "error", "code": "INVALID_MESSAGE",
                        "detail": "점자 메시지 내용은 문자열이어야 합니다.",
                    })
                    continue
                content = event["content"].strip()
                source = "braille"

            # 공백 또는 빈 메시지는 전송하지 않는다.
            if not content:
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": "EMPTY_MESSAGE",
                        "detail": (
                            "빈 메시지는 전송할 수 없습니다."
                        ),
                    }
                )

                continue

            # 지나치게 긴 메시지를 차단한다.
            if (
                len(content)
                > MAX_CHAT_MESSAGE_LENGTH
            ):
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": "MESSAGE_TOO_LONG",
                        "detail": (
                            "메시지는 최대 "
                            f"{MAX_CHAT_MESSAGE_LENGTH}자까지 "
                            "입력할 수 있습니다."
                        ),
                    }
                )

                continue

            message = {
                "type": "message",
                "sender_id": participant_id,
                "content": content,
            }
            if source == "braille":
                message["source"] = source
                if typing_active:
                    typing_active = False
                    await chat_manager.broadcast(
                        room_code=normalized_room_code,
                        message={
                            "type": "typing", "source": "braille",
                            "sender_id": participant_id, "active": False,
                        },
                    )

            # 같은 room_code에 연결된 참가자에게만
            # 메시지를 전송한다.
            await chat_manager.broadcast(
                room_code=normalized_room_code,
                message=message,
            )

    except WebSocketDisconnect:
        print(
            "[채팅 연결 해제 감지] "
            f"room={normalized_room_code}, "
            f"participant={participant_id}"
        )

    except Exception as error:
        print(
            "[채팅 WebSocket 오류] "
            f"room={normalized_room_code}, "
            f"participant={participant_id}, "
            f"error={error}"
        )

        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "INTERNAL_ERROR",
                    "detail": (
                        "채팅 처리 중 오류가 발생했습니다."
                    ),
                }
            )
        except Exception:
            pass

        try:
            await websocket.close(
                code=(
                    status.WS_1011_INTERNAL_ERROR
                ),
                reason=(
                    "채팅 처리 중 서버 오류 발생"
                ),
            )
        except Exception:
            pass

    finally:
        # 연결 관리자에 정상적으로 등록된 경우에만
        # 연결 정보를 제거한다.
        if connected:
            # An old socket must not clear a replacement connection's status.
            is_current = (chat_manager.active_connections.get(normalized_room_code, {})
                          .get(participant_id) is websocket)
            chat_manager.disconnect(
                room_code=normalized_room_code,
                participant_id=participant_id,
                websocket=websocket,
            )
            if typing_active and is_current:
                await chat_manager.broadcast(
                    room_code=normalized_room_code,
                    message={
                        "type": "typing", "source": "braille",
                        "sender_id": participant_id, "active": False,
                    },
                )
