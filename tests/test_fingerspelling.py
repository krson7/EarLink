"""지문자 인식 단위테스트.

- frame_feature 회귀 (시드 손 입력 → 기준 55차원 golden)
- Debouncer 상태기계
- FingerspellingSession 조합 흐름 (predict 목킹으로 ㅎㅏㄱㄱㅛ→학교)
- 실 tflite 스모크 (모델·TF 없으면 skip)
- WS hand 메시지 라우팅 (echo + jamo)
TF 는 실 tflite 테스트에서만 필요 — 나머지는 순수 로직.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from server.fingerspelling import (
    ACTIONS,
    Debouncer,
    FingerspellingSession,
    frame_feature,
)

ROOT = Path(__file__).resolve().parent.parent
TFLITE = ROOT / "models" / "fingerspelling" / "multi_hand_gesture_classifier.tflite"


# ---------------------------------------------------------------- 전처리 회귀

# default_rng(42) 손 3개에 대한 frame_feature 기준값 (전처리 변경 시 회귀 검출).
GOLDEN = [
    [0.311189, 0.950348, -0.939682, 0.342051, 0.961905, -0.273384, -0.883472, -0.468484, -0.636995, 0.770868, 0.934513, -0.355929, -0.31901, -0.947751, 0.562462, -0.826823, 0.268217, 0.963359, -0.243388, -0.969929, 0.367177, 0.930151, -0.265454, -0.964124, -0.613884, -0.789396, -0.43911, 0.898433, 0.900907, 0.434013, -0.574384, -0.818586, -0.773518, -0.633774, -0.764661, 0.644433, 0.446968, 0.89455, 0.790549, 0.612398, 88.129074, 175.863876, 136.198349, 150.418533, 87.752663, 52.8293, 178.528397, 172.544983, 173.85231, 116.082001, 90.324615, 150.778976, 79.452293, 76.426117, 25.687582],
    [0.258924, 0.965898, -0.652584, -0.757716, 0.893131, -0.449796, -0.964369, -0.26456, 0.238744, 0.971083, -0.576507, 0.817092, -0.757861, -0.652416, -0.574877, -0.81824, -0.196806, 0.980442, -0.331211, 0.943557, 0.312217, -0.950011, -0.289536, -0.957167, -0.983182, 0.182627, 0.988439, -0.151621, 0.964623, -0.263632, -0.998224, 0.059565, -0.758358, 0.651838, 0.907441, 0.420179, 0.273357, -0.961913, 0.530185, -0.847882, 154.269485, 104.006119, 137.928619, 49.017654, 95.518768, 14.184934, 7.992029, 178.850601, 35.023125, 178.198013, 6.564811, 168.129181, 114.473778, 98.981758, 16.153849],
    [0.883042, 0.469295, -0.997572, 0.069637, -0.045834, 0.998949, 0.681779, -0.731559, 0.458603, 0.888641, 0.110446, -0.993882, -0.268626, 0.963245, 0.949163, -0.314786, 0.266581, 0.963813, 0.897316, -0.44139, -0.595251, -0.80354, -0.48857, 0.872525, 0.967975, 0.251045, -0.369428, 0.929259, -0.422169, 0.906517, 0.973316, -0.229467, 0.60608, 0.795404, 0.989186, 0.146667, -0.90558, -0.424176, -0.480448, 0.877023, 148.018341, 83.379845, 139.644226, 146.361984, 170.758514, 123.930374, 100.731682, 100.337944, 114.222832, 97.140991, 3.291234, 128.237274, 44.259605, 163.335266, 86.383881],
]


def test_frame_feature_golden():
    """frame_feature 회귀 고정 — 시드된 손 입력 → 기준 55차원."""
    rng = np.random.default_rng(42)
    for expected in GOLDEN:
        hand = rng.uniform(0, 1, (21, 2))
        got = frame_feature(hand)
        assert got is not None
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-5)


def test_frame_feature_shape():
    hand = np.random.default_rng(0).uniform(0, 1, (21, 2))
    f = frame_feature(hand)
    assert f is not None and f.shape == (55,) and f.dtype == np.float32


# ---------------------------------------------------------------- Debouncer

def test_debouncer_fires_after_stable_frames():
    deb = Debouncer(confirm_frames=3, conf_thr=0.8)
    assert deb.update("ㄱ", 0.9) is None   # 1
    assert deb.update("ㄱ", 0.9) is None   # 2
    assert deb.update("ㄱ", 0.9) == "ㄱ"   # 3 → 확정
    assert deb.update("ㄱ", 0.9) is None   # 이미 fired


def test_debouncer_low_conf_resets():
    deb = Debouncer(confirm_frames=2, conf_thr=0.8)
    deb.update("ㄴ", 0.9)
    assert deb.update("ㄴ", 0.5) is None   # 저신뢰 → 리셋
    assert deb.update("ㄴ", 0.9) is None   # 카운트 다시 1


def test_debouncer_held_single_fires_once():
    """같은 자모를 계속 유지하면 1회만 확정 (연속 홀드 스팸 차단)."""
    deb = Debouncer(confirm_frames=2, conf_thr=0.8)
    assert deb.update("ㄱ", 0.9) is None
    assert deb.update("ㄱ", 0.9) == "ㄱ"   # 첫 확정
    for _ in range(5):                     # 손 안 뗀 채 계속 유지
        assert deb.update("ㄱ", 0.9) is None


def test_debouncer_repeat_after_release():
    """해제(손뗌=None) 후엔 같은 자모 재확정 가능 (연속 ㄱㄱ 처리)."""
    deb = Debouncer(confirm_frames=2, conf_thr=0.8)
    assert deb.update("ㄱ", 0.9) is None
    assert deb.update("ㄱ", 0.9) == "ㄱ"   # 첫 확정
    deb.update(None, 0.0)                  # 손 뗌 → 재확정 무장
    deb.update("ㄱ", 0.9)
    assert deb.update("ㄱ", 0.9) == "ㄱ"   # 재확정


def test_debouncer_repeat_after_different():
    """사이에 다른 자모를 봐도 재확정 가능 (기존 경로 유지)."""
    deb = Debouncer(confirm_frames=2, conf_thr=0.8)
    deb.update("ㄱ", 0.9)
    assert deb.update("ㄱ", 0.9) == "ㄱ"
    deb.update("ㅏ", 0.9)
    deb.update("ㄱ", 0.9)
    assert deb.update("ㄱ", 0.9) == "ㄱ"


# ---------------------------------------------------------------- 조합 흐름

class _MockPredictor:
    """예측 결과를 스크립트대로 반환 (tflite 없이 세션 흐름 테스트)."""
    def __init__(self, script):
        self.actions = ACTIONS
        self._script = script  # [(jamo, conf), ...] 프레임 순서
        self._i = 0

    def predict(self, seq):
        jamo, conf = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return ACTIONS.index(jamo), conf


def _run_session(jamos, confirm_frames=3):
    """서로 다른 자모 시퀀스를 세션에 흘려 최종 조합 문자열을 얻는다.
    각 자모를 confirm_frames 유지 후 사이에 저신뢰 프레임으로 디바운스 리셋."""
    seq = []
    for jamo in jamos:
        seq += [(jamo, 0.95)] * confirm_frames
        seq += [(jamo, 0.3)] * 2
    session = FingerspellingSession(_MockPredictor(seq),
                                    confirm_frames=confirm_frames, conf_threshold=0.8)
    hand = np.random.default_rng(0).uniform(0, 1, (21, 2)).tolist()
    composed = ""
    for _ in range(10 + len(seq)):     # 버퍼(10) 채운 뒤 스크립트 소진
        composed = session.push(hand)["composed"]
    return composed


def test_session_composes_word():
    """세션이 확정 자모를 shared 조합기로 흘려보내 완성형 한글을 만든다."""
    assert _run_session(["ㅁ", "ㅏ", "ㄴ", "ㅕ"]) == "마녀"   # 받침 이동 포함


def test_session_composes_simple():
    assert _run_session(["ㄴ", "ㅏ"]) == "나"


def test_session_repeat_jamo_with_release():
    """연속 같은 자모(ㄱㄱ)도 사이에 손을 떼면(=저신뢰 gap) 재확정된다.
    _run_session 이 자모마다 저신뢰 프레임 2개를 삽입 → release-gate 무장 →
    학교(ㅎㅏㄱㄱㅛ)가 정상 조합. (조합기는 원래 정상 — test_hangul_compose)."""
    assert _run_session(["ㅎ", "ㅏ", "ㄱ", "ㄱ", "ㅛ"]) == "학교"


def test_session_hand_none_clears_buffer():
    session = FingerspellingSession(None)  # predictor 없어도 crash 안 남
    out = session.push(None)
    assert out["composed"] == "" and out["current"] is None


# ---------------------------------------------------------------- 실 tflite 스모크

@pytest.mark.skipif(not TFLITE.exists(), reason="tflite 모델 없음")
def test_real_tflite_predict():
    pytest.importorskip("tensorflow")
    from server.fingerspelling import FingerspellPredictor
    pred = FingerspellPredictor.load(TFLITE)
    assert pred is not None
    idx, conf = pred.predict(np.zeros((10, 55), dtype=np.float32))
    assert 0 <= idx < len(ACTIONS) and 0.0 <= conf <= 1.0


# ---------------------------------------------------------------- WS 라우팅

def test_ws_hand_routing():
    """hand 메시지 → echo 응답 (jamo는 predictor가 있으면 추가)."""
    from fastapi.testclient import TestClient
    from server.main import app

    hand = [[0.5, 0.5]] * 21

    with TestClient(app) as client:
        join_response = client.post(
            "/rooms/join",
            json={
                "accessibility_mode": "HEARING",
            },
        )

        assert join_response.status_code == 201

        participant = join_response.json()
        room_code = participant["room_code"]
        participant_id = participant["participant_id"]

        websocket_url = (
            f"/ws/translate/"
            f"{room_code}/"
            f"{participant_id}"
        )

        with client.websocket_connect(
            websocket_url,
        ) as ws:
            ws.send_text(
                json.dumps(
                    {
                        "type": "hand",
                        "frame_id": 1,
                        "t": 0.0,
                        "hand": hand,
                    }
                )
            )

            messages = []

            for _ in range(2):
                try:
                    messages.append(
                        json.loads(
                            ws.receive_text()
                        )
                    )
                except Exception:
                    break

            message_types = {
                message["type"]
                for message in messages
            }

            assert "echo" in message_types

            for message in messages:
                if message["type"] == "jamo":
                    assert "composed" in message
