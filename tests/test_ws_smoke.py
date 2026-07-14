"""/ws/translate 스모크 테스트 — 에코 프로토콜(M2) + 실시간 추론 세션(M5).

기존 에코 테스트 2개는 TestClient 를 컨텍스트 없이 써서 lifespan(TF 모델
로드)을 트리거하지 않는다 — M2 계약 보존을 빠르게 검증한다.
test_full_session 만 `with TestClient(app)` 로 모델을 실제 로드한다.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server.main import app
from shared.features import FEATURE_DIM

ROOT = Path(__file__).resolve().parent.parent


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


# ---------------------------------------------------------------- M5 세션

def _frame(wrist_x):
    """test_segmenter 와 동일한 합성 프레임 (어깨 고정 + 손목/손 이동)."""
    v = np.zeros(FEATURE_DIM, dtype=np.float32)
    v[12:16] = [-0.5, 0.0, 0.5, 0.0]
    v[16:24] = wrist_x
    v[24:108] = wrist_x
    v[FEATURE_DIM - 2] = v[FEATURE_DIM - 1] = 1.0
    return [round(float(x), 4) for x in v]


@pytest.mark.skipif(not list((ROOT / "models").glob("ksl_word_v*.keras")),
                    reason="모델 아티팩트 없음 (git 미포함) — training.train 먼저 실행")
def test_full_session():
    """대기 → 수어 → 대기 스트림에서 signing → idle → result 순서를 검증."""
    rng = np.random.default_rng(11)
    frames = ([_frame(0.1 + rng.uniform(-0.003, 0.003)) for _ in range(10)]
              + [_frame(0.1 + 0.08 * i) for i in range(25)]
              + [_frame(2.1 + rng.uniform(-0.003, 0.003)) for _ in range(15)])

    with TestClient(app) as client:  # 컨텍스트 → lifespan → 모델 로드
        with client.websocket_connect("/ws/translate") as ws:
            received = []
            for i, v in enumerate(frames):
                # t 는 ~15fps 실제 간격 — 서버의 fps 추정·밀도 정규화가 동작하게
                ws.send_text(json.dumps({"type": "features", "frame_id": i,
                                         "t": round(i * 66.7, 1), "v": v}))
                received.append(json.loads(ws.receive_text()))  # echo 1:1 응답
                # status/result 는 비동기 삽입 — 큐에 쌓인 만큼 추가 수신
            # 남은 메시지 수거: result 가 올 때까지 (echo 외 메시지가 섞여 있음)
            events = [m for m in received if m["type"] != "echo"]
            while not any(m["type"] == "result" for m in events):
                m = json.loads(ws.receive_text())
                if m["type"] != "echo":
                    events.append(m)

    states = [m["state"] for m in events if m["type"] == "status"]
    results = [m for m in events if m["type"] == "result"]
    assert states[:2] == ["signing", "idle"]
    assert len(results) == 1
    r = results[0]
    assert set(r) >= {"text", "top1", "confidence", "n_frames", "low_confidence"}
    assert r["top1"] and 0.0 <= r["confidence"] <= 1.0
    assert (r["text"] is None) == r["low_confidence"]
