"""/ws/translate 스모크 테스트 — M2 특징 벡터 에코 프로토콜 검증."""

import json

from fastapi.testclient import TestClient

from server.main import app


def test_features_echo():
    client = TestClient(app)
    with client.websocket_connect("/ws/translate") as ws:
        ws.send_text(json.dumps({
            "type": "features", "frame_id": 1, "t": 12.5,
            "dim": 250, "v": [0.0] * 250,
        }))
        echo = json.loads(ws.receive_text())
        assert echo == {"type": "echo", "frame_id": 1, "t": 12.5, "n_dims": 250}


def test_ignores_invalid_and_legacy_messages():
    """쓰레기·구형(M1 keypoints) 메시지는 연결을 끊지 않고 조용히 무시한다."""
    client = TestClient(app)
    with client.websocket_connect("/ws/translate") as ws:
        ws.send_text("not-json")
        ws.send_text(json.dumps({"type": "keypoints", "frame_id": 1,
                                 "pose": [[0.5, 0.5]]}))
        ws.send_text(json.dumps({"type": "features", "frame_id": 7, "t": 1.0,
                                 "v": None}))  # v가 list가 아님 → 무시
        ws.send_text(json.dumps({"type": "features", "frame_id": 8, "t": 2.0,
                                 "dim": 3, "v": [0.1, 0.2, 0.3]}))
        echo = json.loads(ws.receive_text())
        assert echo["frame_id"] == 8          # 첫 응답이 유효 메시지의 것
        assert echo["n_dims"] == 3            # 서버는 실제 배열 길이를 계산
