"""이어링크(EarLink) FastAPI 서버.

주요 기능
- 서버 시작 시 고정 대화방 EAR001 자동 생성
- 일반 텍스트 실시간 채팅
- 지문자 손 키포인트 인식
- 대화방 참가자 검증

개인정보 보호 원칙
- 클라이언트는 손의 21개 키포인트 좌표만 서버로 전송한다.
- 영상 프레임은 서버에서 수신하거나 저장하지 않는다.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from sqlalchemy.orm import Session

from server.database import Base, engine

# create_all() 전에 엔티티가 SQLAlchemy에 등록되도록 import한다.
from server.entities.participant import Participant  # noqa: F401
from server.entities.room import Room  # noqa: F401

from server.fingerspelling import (
    FingerspellingSession,
    FingerspellPredictor,
)
from server.routers.chat_router import router as chat_router
from server.routers.room_router import router as room_router
from server.schemas import EchoMsg, HandMsg, JamoMsg
from server.services.room_service import (
    ensure_default_room,
    get_participant_in_room,
    normalize_room_code,
)


ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict[str, Any]:
    """프로젝트 루트의 config.yaml 파일을 읽는다."""

    config_path = ROOT / "config.yaml"

    if not config_path.exists():
        raise RuntimeError(
            f"config.yaml 파일을 찾을 수 없습니다: {config_path}"
        )

    loaded_config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    if not isinstance(loaded_config, dict):
        raise RuntimeError(
            "config.yaml의 최상위 설정은 객체 형식이어야 합니다."
        )

    return loaded_config


def prepare_default_room() -> None:
    """이어링크에서 사용할 고정 대화방을 준비한다."""

    with Session(bind=engine) as db:
        default_room = ensure_default_room(db)

        print(
            "[고정 대화방 준비 완료] "
            f"id={default_room.id}, "
            f"room_code={default_room.room_code}, "
            f"status={default_room.status}"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 서버 시작 및 종료 작업을 처리한다."""

    # 존재하지 않는 테이블만 생성한다.
    # 이미 존재하는 테이블과 데이터는 삭제하지 않는다.
    Base.metadata.create_all(bind=engine)

    # 서버 시작 시 EAR001 고정 방을 확인하고,
    # 존재하지 않으면 자동 생성한다.
    prepare_default_room()

    config = load_config()

    fingerspelling_config = (
        config.get("fingerspelling") or {}
    )

    if not isinstance(
        fingerspelling_config,
        dict,
    ):
        raise RuntimeError(
            "config.yaml의 fingerspelling 설정은 "
            "객체 형식이어야 합니다."
        )

    app.state.fs_cfg = fingerspelling_config

    # 공용 TFLite 모델을 여러 요청이 동시에 사용하지 않도록 한다.
    app.state.fs_predictor_lock = asyncio.Lock()

    try:
        model_path_value = fingerspelling_config.get(
            "model_path",
            (
                "models/fingerspelling/"
                "multi_hand_gesture_classifier.tflite"
            ),
        )

        model_path = ROOT / model_path_value

        app.state.fs_predictor = (
            FingerspellPredictor.load(
                model_path
            )
        )

        print(
            "[지문자 모델 로드 성공] "
            f"path={model_path}"
        )

    except Exception as error:
        app.state.fs_predictor = None

        print(
            "[경고] 지문자 모델 로드 실패: "
            f"{error}"
        )

        print(
            "[경고] 일반 텍스트 채팅은 사용할 수 있지만, "
            "지문자 인식 기능은 비활성화됩니다."
        )

    yield

    print("[서버 종료] EarLink 서버가 종료됩니다.")


app = FastAPI(
    title="EarLink API",
    description=(
        "이어링크 지문자 인식 및 실시간 채팅 서버"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# Next.js 개발 서버의 HTTP 요청을 허용한다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 방 참가 API
app.include_router(room_router)

# 일반 텍스트 채팅 WebSocket
app.include_router(chat_router)


@app.get("/")
def health_check() -> dict[str, str]:
    """백엔드 서버 실행 상태를 반환한다."""

    return {
        "service": "EarLink API",
        "status": "running",
        "default_room_code": "EAR001",
    }


def validate_translate_participant(
    room_code: str,
    participant_id: int,
) -> None:
    """
    참가자가 실제로 해당 대화방에 속해 있는지 확인한다.

    WebSocket 연결 동안 DB 세션을 계속 유지하지 않고,
    참가자 검증 시점에만 세션을 열고 종료한다.
    """

    with Session(bind=engine) as db:
        get_participant_in_room(
            db=db,
            room_code=room_code,
            participant_id=participant_id,
        )


def create_fingerspelling_session(
    predictor: FingerspellPredictor,
    config: dict[str, Any],
) -> FingerspellingSession:
    """참가자 한 명이 사용할 지문자 인식 세션을 생성한다."""

    return FingerspellingSession(
        predictor,
        confirm_frames=config.get(
            "confirm_frames",
            8,
        ),
        conf_threshold=config.get(
            "conf_threshold",
            0.8,
        ),
    )


@app.websocket(
    "/ws/translate/{room_code}/{participant_id}"
)
async def ws_translate(
    websocket: WebSocket,
    room_code: str,
    participant_id: int,
) -> None:
    """
    손 키포인트를 받아 지문자 인식 결과를 반환한다.

    클라이언트는 영상 프레임이 아닌
    손의 21개 키포인트 좌표만 전송한다.
    """

    # 검증 실패 시 브라우저가 1008 종료 코드를 받을 수 있도록
    # 참가자 검증 전에 WebSocket 연결을 승인한다.
    await websocket.accept()

    try:
        normalized_room_code = normalize_room_code(
            room_code
        )

        # 동기 SQLAlchemy 조회로 이벤트 루프가 막히지 않도록
        # 별도 스레드에서 참가자를 검증한다.
        await asyncio.to_thread(
            validate_translate_participant,
            normalized_room_code,
            participant_id,
        )

    except ValueError as error:
        print(
            "[지문자 연결 거부] "
            f"room={room_code}, "
            f"participant={participant_id}, "
            f"reason={error}"
        )

        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "code": (
                        "PARTICIPANT_VALIDATION_FAILED"
                    ),
                    "detail": str(error),
                }
            )
        except Exception:
            pass

        try:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="대화방 또는 참가자 확인 실패",
            )
        except Exception:
            pass

        return

    except Exception as error:
        print(
            "[지문자 참가자 검증 오류] "
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
                code=status.WS_1011_INTERNAL_ERROR,
                reason="참가자 확인 중 서버 오류 발생",
            )
        except Exception:
            pass

        return

    print(
        "[지문자 연결] "
        f"room={normalized_room_code}, "
        f"participant={participant_id}"
    )

    fs_predictor = getattr(
        websocket.app.state,
        "fs_predictor",
        None,
    )

    fs_predictor_lock = getattr(
        websocket.app.state,
        "fs_predictor_lock",
        None,
    )

    fs_config_value = (
        getattr(
            websocket.app.state,
            "fs_cfg",
            None,
        )
        or {}
    )

    fs_config: dict[str, Any]

    if isinstance(
        fs_config_value,
        dict,
    ):
        fs_config = fs_config_value
    else:
        fs_config = {}

    if fs_predictor is None:
        await websocket.send_json(
            {
                "type": "error",
                "code": "MODEL_UNAVAILABLE",
                "detail": (
                    "지문자 인식 모델을 사용할 수 없습니다."
                ),
            }
        )

        await websocket.close(
            code=status.WS_1011_INTERNAL_ERROR,
            reason="지문자 인식 모델 사용 불가",
        )

        return

    if fs_predictor_lock is None:
        await websocket.send_json(
            {
                "type": "error",
                "code": "MODEL_LOCK_UNAVAILABLE",
                "detail": (
                    "지문자 인식 기능을 초기화하지 "
                    "못했습니다."
                ),
            }
        )

        await websocket.close(
            code=status.WS_1011_INTERNAL_ERROR,
            reason="지문자 인식 초기화 실패",
        )

        return

    # 자모 확정 상태와 한글 조합 상태는
    # WebSocket 참가자마다 별도로 관리한다.
    fs_session = create_fingerspelling_session(
        predictor=fs_predictor,
        config=fs_config,
    )

    try:
        while True:
            raw_message = (
                await websocket.receive_text()
            )

            try:
                message_data = json.loads(
                    raw_message
                )

            except json.JSONDecodeError:
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": "INVALID_JSON",
                        "detail": (
                            "올바른 JSON 형식이 아닙니다."
                        ),
                    }
                )

                continue

            if not isinstance(
                message_data,
                dict,
            ):
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": "INVALID_MESSAGE",
                        "detail": (
                            "메시지는 JSON 객체 "
                            "형식이어야 합니다."
                        ),
                    }
                )

                continue

            message_type = message_data.get(
                "type"
            )

            # 프론트에서 인식된 문장을 채팅 입력창에 적용한 뒤
            # 현재 참가자의 지문자 조합 상태를 초기화한다.
            if message_type == "reset":
                fs_session = (
                    create_fingerspelling_session(
                        predictor=fs_predictor,
                        config=fs_config,
                    )
                )

                await websocket.send_json(
                    {
                        "type": "reset_complete",
                    }
                )

                print(
                    "[지문자 세션 초기화] "
                    f"room={normalized_room_code}, "
                    f"participant={participant_id}"
                )

                continue

            try:
                hand_message = (
                    HandMsg.model_validate(
                        message_data
                    )
                )

            except ValidationError as error:
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": (
                            "INVALID_HAND_MESSAGE"
                        ),
                        "detail": (
                            "손 키포인트 데이터 형식이 "
                            "올바르지 않습니다."
                        ),
                    }
                )

                print(
                    "[잘못된 손 데이터] "
                    f"room={normalized_room_code}, "
                    f"participant={participant_id}, "
                    f"error={error}"
                )

                continue

            dimension_count = (
                len(hand_message.hand) * 2
                if hand_message.hand
                else 0
            )

            echo_message = EchoMsg(
                frame_id=hand_message.frame_id,
                t=hand_message.t,
                n_dims=dimension_count,
            )

            await websocket.send_text(
                echo_message.model_dump_json()
            )

            try:
                # TFLite predictor는 서버 전체에서 공유하므로
                # 동시에 하나의 추론만 실행되도록 한다.
                async with fs_predictor_lock:
                    output = await asyncio.to_thread(
                        fs_session.push,
                        hand_message.hand,
                    )

            except Exception as error:
                print(
                    "[지문자 추론 오류] "
                    f"room={normalized_room_code}, "
                    f"participant={participant_id}, "
                    f"error={error}"
                )

                await websocket.send_json(
                    {
                        "type": "error",
                        "code": "INFERENCE_ERROR",
                        "detail": (
                            "지문자 인식 중 오류가 "
                            "발생했습니다."
                        ),
                    }
                )

                continue

            jamo_message = JamoMsg(
                **output
            )

            await websocket.send_text(
                jamo_message.model_dump_json()
            )

    except WebSocketDisconnect:
        print(
            "[지문자 연결 해제 감지] "
            f"room={normalized_room_code}, "
            f"participant={participant_id}"
        )

    except Exception as error:
        print(
            "[지문자 WebSocket 오류] "
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
                        "지문자 처리 중 서버 오류가 "
                        "발생했습니다."
                    ),
                }
            )
        except Exception:
            pass

        try:
            await websocket.close(
                code=status.WS_1011_INTERNAL_ERROR,
                reason="지문자 처리 중 서버 오류 발생",
            )
        except Exception:
            pass

    finally:
        print(
            "[지문자 연결 종료] "
            f"room={normalized_room_code}, "
            f"participant={participant_id}"
        )


if __name__ == "__main__":
    import uvicorn

    config = load_config()

    server_config = config.get(
        "server"
    )

    if not isinstance(
        server_config,
        dict,
    ):
        raise RuntimeError(
            "config.yaml의 server 설정은 "
            "객체 형식이어야 합니다."
        )

    uvicorn.run(
        "server.main:app",
        host=server_config.get(
            "host",
            "127.0.0.1",
        ),
        port=server_config.get(
            "port",
            8000,
        ),
        reload=True,
    )