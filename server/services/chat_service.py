from typing import Any

from fastapi import WebSocket


class ChatConnectionManager:
    """대화방별 채팅 WebSocket 연결을 관리한다."""

    def __init__(self) -> None:
        # 연결 구조:
        #
        # {
        #     "ABC123": {
        #         1: WebSocket,
        #         2: WebSocket,
        #     },
        #     "XYZ789": {
        #         3: WebSocket,
        #     },
        # }
        #
        # room_code별로 연결을 분리하기 때문에
        # 서로 다른 방의 메시지가 섞이지 않는다.
        self.active_connections: dict[
            str,
            dict[int, WebSocket],
        ] = {}

    @staticmethod
    def normalize_room_code(
        room_code: str,
    ) -> str:
        """대화방 코드를 공백 제거 및 대문자로 정규화한다."""

        normalized_room_code = (
            room_code
            .strip()
            .upper()
        )

        if not normalized_room_code:
            raise ValueError(
                "대화방 코드를 입력해주세요."
            )

        return normalized_room_code

    async def connect(
        self,
        room_code: str,
        participant_id: int,
        websocket: WebSocket,
    ) -> None:
        """
        참가자의 채팅 WebSocket 연결을 등록한다.

        WebSocket accept와 참가자 검증은
        chat_router에서 먼저 처리해야 한다.

        같은 참가자가 새로고침하거나 재접속하면
        기존 연결을 새로운 WebSocket 연결로 교체한다.
        """

        normalized_room_code = (
            self.normalize_room_code(
                room_code
            )
        )

        # websocket.accept()는 chat_router에서
        # 이미 실행했으므로 여기에서는 다시 호출하지 않는다.

        room_connections = (
            self.active_connections.setdefault(
                normalized_room_code,
                {},
            )
        )

        previous_websocket = (
            room_connections.get(
                participant_id
            )
        )

        # 새로운 WebSocket을 먼저 등록한다.
        room_connections[
            participant_id
        ] = websocket

        # 같은 참가자의 기존 연결이 남아 있다면 종료한다.
        #
        # 기존 연결의 finally가 실행되더라도
        # disconnect()에서 WebSocket 객체를 비교하므로
        # 새로운 연결은 제거되지 않는다.
        if (
            previous_websocket
            is not None
            and previous_websocket
            is not websocket
        ):
            try:
                await previous_websocket.close(
                    code=1000,
                    reason=(
                        "새로운 연결로 교체되었습니다."
                    ),
                )

            except Exception:
                # 기존 연결이 이미 종료된 경우 무시한다.
                pass

        print(
            "[채팅 연결] "
            f"room={normalized_room_code}, "
            f"participant={participant_id}, "
            f"count={len(room_connections)}"
        )

    def disconnect(
        self,
        room_code: str,
        participant_id: int,
        websocket: WebSocket,
    ) -> None:
        """
        참가자의 채팅 WebSocket 연결을 제거한다.

        종료된 WebSocket과 현재 등록된 WebSocket이
        같은 경우에만 삭제한다.

        이를 통해 이전 연결의 finally가 재접속한
        새로운 연결을 삭제하는 것을 막는다.
        """

        normalized_room_code = (
            self.normalize_room_code(
                room_code
            )
        )

        room_connections = (
            self.active_connections.get(
                normalized_room_code
            )
        )

        if room_connections is None:
            return

        current_websocket = (
            room_connections.get(
                participant_id
            )
        )

        # 이미 새로운 WebSocket으로 교체됐다면
        # 이전 WebSocket의 종료 요청은 무시한다.
        if (
            current_websocket
            is not websocket
        ):
            return

        room_connections.pop(
            participant_id,
            None,
        )

        # 해당 방에 연결된 참가자가 더 이상 없으면
        # 메모리에서 방 연결 정보도 제거한다.
        if not room_connections:
            self.active_connections.pop(
                normalized_room_code,
                None,
            )

        print(
            "[채팅 연결 종료] "
            f"room={normalized_room_code}, "
            f"participant={participant_id}, "
            f"count={len(room_connections)}"
        )

    async def broadcast(
        self,
        room_code: str,
        message: dict[str, Any],
    ) -> None:
        """해당 대화방의 모든 참가자에게 메시지를 전송한다."""

        normalized_room_code = (
            self.normalize_room_code(
                room_code
            )
        )

        room_connections = (
            self.active_connections.get(
                normalized_room_code,
                {},
            )
        )

        # 전송 도중 연결 정보가 바뀔 수 있으므로
        # 현재 연결 목록의 복사본을 만든다.
        connection_snapshot = list(
            room_connections.items()
        )

        disconnected_connections: list[
            tuple[int, WebSocket]
        ] = []

        for (
            participant_id,
            websocket,
        ) in connection_snapshot:
            try:
                await websocket.send_json(
                    message
                )

            except Exception:
                # 전송에 실패한 WebSocket은
                # 반복문이 끝난 후 제거한다.
                disconnected_connections.append(
                    (
                        participant_id,
                        websocket,
                    )
                )

        # 반복 중 원본 딕셔너리를 수정하지 않고
        # 순회가 끝난 뒤 끊어진 연결을 제거한다.
        for (
            participant_id,
            websocket,
        ) in disconnected_connections:
            self.disconnect(
                room_code=(
                    normalized_room_code
                ),
                participant_id=(
                    participant_id
                ),
                websocket=websocket,
            )

    def is_connected(
        self,
        room_code: str,
        participant_id: int,
    ) -> bool:
        """참가자의 현재 WebSocket 연결 여부를 반환한다."""

        normalized_room_code = (
            self.normalize_room_code(
                room_code
            )
        )

        room_connections = (
            self.active_connections.get(
                normalized_room_code,
                {},
            )
        )

        return (
            participant_id
            in room_connections
        )

    def get_connection_count(
        self,
        room_code: str,
    ) -> int:
        """현재 해당 대화방에 연결된 참가자 수를 반환한다."""

        normalized_room_code = (
            self.normalize_room_code(
                room_code
            )
        )

        room_connections = (
            self.active_connections.get(
                normalized_room_code,
                {},
            )
        )

        return len(
            room_connections
        )


# 서버 전체에서 하나의 연결 관리자를 공유한다.
chat_manager = ChatConnectionManager()