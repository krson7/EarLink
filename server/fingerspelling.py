"""지문자(fingerspelling) 인식.

손 21점 → 전처리(뼈 단위벡터 20 + 관절 각도 15 = 55차원, 시퀀스 10) → tflite
분류 → 디바운스 → 한글 조합(shared.hangul_compose).

TF/tflite 는 지연 import(FingerspellPredictor.load 내부) — 전처리·상태기계
단위테스트가 TF 없이 돌고, 서버 기동 로그에서 로드 시점이 명확해진다.
재학습 시 config 의 model_path 와 아래 ACTIONS 만 교체하면 된다.
"""

import threading
from collections import deque
from pathlib import Path

import numpy as np

from shared.hangul_compose import HangulComposer

SEQ_LEN = 10  # tflite 입력 시퀀스 길이 (모델 고정)

# tflite 출력 31 클래스 (자모 라벨 순서 — 재학습 시 이 리스트도 교체).
ACTIONS = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ',
           'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ',
           'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ', 'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']


# ---------------------------------------------------------------- 전처리


def vector_normalization(joint):
    """손 21점 (21,2) → 뼈 단위벡터 20 + 관절 각도 15."""
    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :2]
    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :2]
    v = v2 - v1
    v = v / (np.linalg.norm(v, axis=1)[:, np.newaxis] + 1e-9)
    angle = np.arccos(np.einsum('nt,nt->n',
                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
    angle = np.degrees(angle)
    return v, np.array([angle], dtype=np.float32)


def frame_feature(joint):
    """손 21점 → 55차원 프레임 특징. NaN 있으면 None."""
    v, angle = vector_normalization(np.asarray(joint, dtype=np.float64))
    d = np.concatenate([v.flatten(), angle.flatten()]).astype(np.float32)
    return None if np.isnan(d).any() else d


# ---------------------------------------------------------------- 디바운스
# 같은 자모 N프레임 유지 시 확정, 해제(손뗌/저신뢰/다른 자모) 후 재확정 허용.


class Debouncer:
    def __init__(self, confirm_frames, conf_thr):
        self.n = confirm_frames
        self.thr = conf_thr
        self.active = None
        self.count = 0
        self.fired = False
        self.last_confirmed = None
        # 해제(손뗌/저신뢰/다른 자모) 후에만 같은 자모 재확정 허용
        self.released = True

    def update(self, jamo, conf):
        """프레임별 예측 → 확정된 자모 반환(없으면 None)."""
        if jamo is None or conf < self.thr:
            self.active = None
            self.count = 0
            self.fired = False
            self.released = True
            return None
        if jamo != self.active:
            self.active = jamo
            self.count = 1
            self.fired = False
            if jamo != self.last_confirmed:
                self.released = True
            return None
        self.count += 1
        if self.count >= self.n and not self.fired:
            if self.active != self.last_confirmed or self.released:
                self.fired = True
                self.last_confirmed = self.active
                self.released = False
                return self.active
        return None


# ---------------------------------------------------------------- 추론기


class FingerspellPredictor:
    """기동 시 1회 로드되는 tflite 싱글턴.

    tflite Interpreter 는 스레드 안전하지 않다 — set_tensor/invoke/get_tensor
    한 벌을 Lock 으로 직렬화한다(asyncio.to_thread 로 여러 연결이 동시에 호출).
    """

    def __init__(self, interp, inp, out, actions):
        self._interp = interp
        self._inp = inp
        self._out = out
        self.actions = actions
        self._lock = threading.Lock()

    @classmethod
    def load(cls, model_path, actions=ACTIONS):
        """tflite 로드. 파일 없으면 None (에코만으로 기동)."""
        path = Path(model_path)
        if not path.exists():
            return None
        import tensorflow as tf  # 지연 import (~수 초) — 기동 시 1회

        interp = tf.lite.Interpreter(model_path=str(path))
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        predictor = cls(interp, inp, out, actions)
        predictor.predict(np.zeros((SEQ_LEN, 55), dtype=np.float32))  # 워밍업
        print(f"[모델] 지문자 tflite 로드 ({path.name}, {len(actions)}자모)")
        return predictor

    def predict(self, seq):
        """(10,55) → (idx, conf). Lock 으로 tflite 호출 직렬화."""
        with self._lock:
            self._interp.set_tensor(
                self._inp["index"], seq[np.newaxis].astype(self._inp["dtype"]))
            self._interp.invoke()
            y = self._interp.get_tensor(self._out["index"])[0]
        i = int(np.argmax(y))
        return i, float(y[i])


class FingerspellingSession:
    """WebSocket 연결당 1개.

    손 21점을 받아 55차원 시퀀스 버퍼(10)를 채우고, 버퍼가 차면 예측·디바운스,
    확정 자모를 HangulComposer 로 조합한다. push() 는 jamo 메시지 dict 를 반환한다.
    """

    def __init__(self, predictor, confirm_frames=8, conf_threshold=0.8):
        self._predictor = predictor
        self._buf = deque(maxlen=SEQ_LEN)
        self._deb = Debouncer(confirm_frames, conf_threshold)
        self._composer = HangulComposer()

    def reset(self):
        self._buf.clear()
        self._composer.reset()
        self._deb.last_confirmed = None
        self._deb.released = True

    def push(self, hand):
        """손 21점(또는 None) → jamo 메시지 dict."""
        current = None
        confidence = None
        confirmed = None

        if hand is None:
            self._buf.clear()
            self._deb.update(None, 0.0)
        else:
            feat = frame_feature(hand)
            if feat is not None:
                self._buf.append(feat)
            if len(self._buf) == SEQ_LEN and self._predictor is not None:
                idx, conf = self._predictor.predict(
                    np.array(self._buf, dtype=np.float32))
                current = self._predictor.actions[idx]
                confidence = round(conf, 4)
                fired = self._deb.update(current, conf)
                if fired:
                    confirmed = fired
                    self._composer.feed(fired)
            else:
                self._deb.update(None, 0.0)

        return {
            "current": current,
            "confidence": confidence,
            "confirmed": confirmed,
            "composed": self._composer.text,
        }
