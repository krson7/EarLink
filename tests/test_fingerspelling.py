"""지문자 모드 단위테스트 — Track 2.

- vector_normalization 패리티: demo.py 원본과 서버 이식본이 bit 일치
- Debouncer 상태기계
- FingerspellingSession 조합 흐름 (predict 목킹으로 ㅎㅏㄱㄱㅛ→학교)
- 실 tflite 스모크 (모델·TF 없으면 skip)
- WS hand 메시지 라우팅 (echo + jamo)
TF 는 실 tflite 테스트에서만 필요 — 나머지는 순수 로직.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from server.fingerspelling import (
    ACTIONS,
    Debouncer,
    FingerspellingSession,
    frame_feature,
    vector_normalization,
)

ROOT = Path(__file__).resolve().parent.parent
DEMO_PY = ROOT / "demo_jamo_test" / "demo.py"
TFLITE = ROOT / "models" / "fingerspelling" / "multi_hand_gesture_classifier.tflite"


# ---------------------------------------------------------------- 전처리 패리티

def _load_demo_module():
    """demo.py 를 모듈로 로드. demo.py 는 'from hangul_compose import ...' 로
    상대 임포트하므로 그 디렉토리를 일시적으로 sys.path 에 넣는다."""
    import sys
    demo_dir = str(DEMO_PY.parent)
    sys.path.insert(0, demo_dir)
    try:
        spec = importlib.util.spec_from_file_location("demo_ref", DEMO_PY)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    finally:
        sys.path.remove(demo_dir)


@pytest.mark.skipif(not DEMO_PY.exists(), reason="demo.py 없음")
def test_vector_normalization_parity():
    """서버 이식본과 demo.py 원본이 같은 손 입력에 bit 일치하는 55차원."""
    demo = _load_demo_module()
    rng = np.random.default_rng(42)
    for _ in range(5):
        hand = rng.uniform(0, 1, (21, 2))
        srv = frame_feature(hand)
        ref = demo.frame_feature(hand)
        assert srv is not None and ref is not None
        assert np.array_equal(srv, ref)  # 이식이 수치적으로 동일해야 함


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
    """hand 메시지 → echo 응답 (jamo 는 predictor 있으면 추가)."""
    from fastapi.testclient import TestClient
    from server.main import app

    hand = [[0.5, 0.5]] * 21
    with TestClient(app) as client:  # lifespan → 모델 로드
        with client.websocket_connect("/ws/translate") as ws:
            ws.send_text(json.dumps({"type": "hand", "frame_id": 1,
                                     "t": 0.0, "hand": hand}))
            msgs = []
            for _ in range(2):  # echo + (predictor 있으면) jamo
                try:
                    msgs.append(json.loads(ws.receive_text()))
                except Exception:
                    break
            types = {m["type"] for m in msgs}
            assert "echo" in types
            # tflite 로드됐으면 jamo 도 옴 (버퍼 미충족이라 current=None)
            for m in msgs:
                if m["type"] == "jamo":
                    assert "composed" in m
