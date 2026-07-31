"""지문자(fingerspelling) 인식.

손 21점
→ 전처리(뼈 단위벡터 20 + 관절 각도 15 = 55차원, 시퀀스 10)
→ TFLite 분류
→ 디바운스
→ 한글 조합(shared.hangul_compose)

TensorFlow/TFLite는 FingerspellPredictor.load() 내부에서 지연 import한다.
전처리와 상태기계 단위 테스트는 TensorFlow 없이 실행할 수 있다.

재학습 시 config의 model_path와 ACTIONS를 함께 변경해야 한다.
"""

import threading
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from shared.hangul_compose import HangulComposer


# TFLite 모델의 입력 시퀀스 길이
SEQ_LEN = 10


# TFLite 출력 클래스 순서
# 모델을 다시 학습했다면 출력 라벨 순서도 함께 변경해야 한다.
ACTIONS = [
    "ㄱ",
    "ㄴ",
    "ㄷ",
    "ㄹ",
    "ㅁ",
    "ㅂ",
    "ㅅ",
    "ㅇ",
    "ㅈ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
    "ㅏ",
    "ㅑ",
    "ㅓ",
    "ㅕ",
    "ㅗ",
    "ㅛ",
    "ㅜ",
    "ㅠ",
    "ㅡ",
    "ㅣ",
    "ㅐ",
    "ㅒ",
    "ㅔ",
    "ㅖ",
    "ㅢ",
    "ㅚ",
    "ㅟ",
]


# ---------------------------------------------------------------------------
# 전처리
# ---------------------------------------------------------------------------


def vector_normalization(
    joint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """손 21점 좌표를 뼈 단위벡터와 관절 각도로 변환한다.

    Args:
        joint:
            (21, 2) 또는 (21, 3) 형태의 손 관절 좌표.

    Returns:
        뼈 단위벡터 20개와 관절 각도 15개.
    """

    v1 = joint[
        [
            0,
            1,
            2,
            3,
            0,
            5,
            6,
            7,
            0,
            9,
            10,
            11,
            0,
            13,
            14,
            15,
            0,
            17,
            18,
            19,
        ],
        :2,
    ]

    v2 = joint[
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
        ],
        :2,
    ]

    vectors = v2 - v1

    vector_lengths = np.linalg.norm(
        vectors,
        axis=1,
    )[:, np.newaxis]

    vectors = vectors / (
        vector_lengths + 1e-9
    )

    angles = np.arccos(
        np.clip(
            np.einsum(
                "nt,nt->n",
                vectors[
                    [
                        0,
                        1,
                        2,
                        4,
                        5,
                        6,
                        8,
                        9,
                        10,
                        12,
                        13,
                        14,
                        16,
                        17,
                        18,
                    ],
                    :,
                ],
                vectors[
                    [
                        1,
                        2,
                        3,
                        5,
                        6,
                        7,
                        9,
                        10,
                        11,
                        13,
                        14,
                        15,
                        17,
                        18,
                        19,
                    ],
                    :,
                ],
            ),
            -1.0,
            1.0,
        )
    )

    angles = np.degrees(angles)

    return (
        vectors,
        np.array(
            [angles],
            dtype=np.float32,
        ),
    )


def frame_feature(
    joint: list[list[float]] | np.ndarray,
) -> Optional[np.ndarray]:
    """손 21점을 55차원 프레임 특징으로 변환한다.

    좌표 구조가 잘못됐거나 NaN/무한대가 포함되면 None을 반환한다.
    """

    try:
        joint_array = np.asarray(
            joint,
            dtype=np.float64,
        )

        if (
            joint_array.ndim != 2
            or joint_array.shape[0] != 21
            or joint_array.shape[1] < 2
        ):
            return None

        vectors, angles = vector_normalization(
            joint_array
        )

        feature = np.concatenate(
            [
                vectors.flatten(),
                angles.flatten(),
            ]
        ).astype(np.float32)

        if not np.isfinite(feature).all():
            return None

        return feature

    except (TypeError, ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# 디바운스
# ---------------------------------------------------------------------------


class Debouncer:
    """같은 자모가 일정 프레임 유지될 때 한 번만 확정한다."""

    def __init__(
        self,
        confirm_frames: int,
        conf_thr: float,
    ) -> None:
        if confirm_frames <= 0:
            raise ValueError(
                "confirm_frames는 1 이상이어야 합니다."
            )

        if not 0.0 <= conf_thr <= 1.0:
            raise ValueError(
                "conf_thr는 0.0 이상 1.0 이하여야 합니다."
            )

        self.n = confirm_frames
        self.thr = conf_thr

        self.active: Optional[str] = None
        self.count = 0
        self.fired = False

        self.last_confirmed: Optional[str] = None

        # 손을 떼거나 다른 자모로 바꾼 뒤에만
        # 같은 자모를 다시 확정할 수 있다.
        self.released = True

    def reset(self) -> None:
        """디바운스 상태를 완전히 초기화한다."""

        self.active = None
        self.count = 0
        self.fired = False
        self.last_confirmed = None
        self.released = True

    def update(
        self,
        jamo: Optional[str],
        conf: float,
    ) -> Optional[str]:
        """프레임별 예측을 받아 확정된 자모를 반환한다."""

        if (
            jamo is None
            or conf < self.thr
        ):
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

        if (
            self.count >= self.n
            and not self.fired
        ):
            if (
                self.active
                != self.last_confirmed
                or self.released
            ):
                self.fired = True
                self.last_confirmed = (
                    self.active
                )
                self.released = False

                return self.active

        return None


# ---------------------------------------------------------------------------
# TFLite 추론기
# ---------------------------------------------------------------------------


class FingerspellPredictor:
    """서버 시작 시 한 번 로드되는 TFLite 추론기.

    TFLite Interpreter는 스레드 안전하지 않으므로
    set_tensor → invoke → get_tensor 과정을 Lock으로 직렬화한다.
    """

    def __init__(
        self,
        interp,
        inp,
        out,
        actions: list[str],
    ) -> None:
        self._interp = interp
        self._inp = inp
        self._out = out

        self.actions = actions

        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        actions: list[str] = ACTIONS,
    ) -> Optional["FingerspellPredictor"]:
        """TFLite 모델을 로드한다.

        모델 파일이 없으면 None을 반환한다.
        """

        path = Path(model_path)

        if not path.exists():
            return None

        # TensorFlow는 서버 시작 시점에만 지연 import한다.
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(
            model_path=str(path)
        )

        interpreter.allocate_tensors()

        input_detail = (
            interpreter.get_input_details()[0]
        )

        output_detail = (
            interpreter.get_output_details()[0]
        )

        predictor = cls(
            interpreter,
            input_detail,
            output_detail,
            actions,
        )

        # 첫 추론 지연을 줄이기 위한 워밍업
        predictor.predict(
            np.zeros(
                (SEQ_LEN, 55),
                dtype=np.float32,
            )
        )

        print(
            "[모델] 지문자 tflite 로드 "
            f"({path.name}, {len(actions)}자모)"
        )

        return predictor

    def predict(
        self,
        seq: np.ndarray,
    ) -> tuple[int, float]:
        """(10, 55) 시퀀스를 분류해 인덱스와 신뢰도를 반환한다."""

        sequence = np.asarray(
            seq,
            dtype=np.float32,
        )

        expected_shape = (
            SEQ_LEN,
            55,
        )

        if sequence.shape != expected_shape:
            raise ValueError(
                "지문자 입력 시퀀스 형태가 "
                f"{expected_shape}이 아닙니다: "
                f"{sequence.shape}"
            )

        with self._lock:
            self._interp.set_tensor(
                self._inp["index"],
                sequence[
                    np.newaxis
                ].astype(
                    self._inp["dtype"]
                ),
            )

            self._interp.invoke()

            output = self._interp.get_tensor(
                self._out["index"]
            )[0]

        index = int(np.argmax(output))
        confidence = float(output[index])

        if (
            index < 0
            or index >= len(self.actions)
        ):
            raise ValueError(
                "모델 출력 인덱스가 ACTIONS 범위를 벗어났습니다."
            )

        return index, confidence


# ---------------------------------------------------------------------------
# 참가자별 지문자 세션
# ---------------------------------------------------------------------------


class FingerspellingSession:
    """WebSocket 연결마다 생성되는 지문자 인식 세션.

    손 21점을 받아 55차원 특징을 만든 뒤 10프레임 시퀀스를 구성한다.
    자모가 디바운스로 확정되면 HangulComposer에 전달한다.
    """

    def __init__(
        self,
        predictor: FingerspellPredictor,
        confirm_frames: int = 8,
        conf_threshold: float = 0.8,
    ) -> None:
        self._predictor = predictor

        self._buf: deque[np.ndarray] = (
            deque(
                maxlen=SEQ_LEN,
            )
        )

        self._deb = Debouncer(
            confirm_frames,
            conf_threshold,
        )

        self._composer = HangulComposer()

    def reset(self) -> None:
        """현재 참가자의 지문자 인식 상태를 완전히 초기화한다.

        초기화 대상:
        - 10프레임 특징 버퍼
        - 디바운스 상태
        - 조합 중인 한글
        """

        self._buf.clear()
        self._deb.reset()
        self._composer.reset()

    def push(
        self,
        hand: Optional[list[list[float]]],
    ) -> dict[str, object]:
        """손 21점을 받아 지문자 인식 결과를 반환한다."""

        current: Optional[str] = None
        confidence: Optional[float] = None
        confirmed: Optional[str] = None

        if hand is None:
            # 손이 감지되지 않으면 시퀀스와
            # 현재 디바운스 진행 상태를 해제한다.
            self._buf.clear()
            self._deb.update(
                None,
                0.0,
            )

        else:
            feature = frame_feature(hand)

            if feature is not None:
                self._buf.append(feature)
            else:
                # 잘못된 좌표가 들어오면 이전 프레임과
                # 연결해 예측하지 않도록 버퍼를 비운다.
                self._buf.clear()
                self._deb.update(
                    None,
                    0.0,
                )

            if (
                len(self._buf) == SEQ_LEN
                and self._predictor is not None
            ):
                index, prediction_confidence = (
                    self._predictor.predict(
                        np.array(
                            self._buf,
                            dtype=np.float32,
                        )
                    )
                )

                current = (
                    self._predictor.actions[
                        index
                    ]
                )

                confidence = round(
                    prediction_confidence,
                    4,
                )

                fired = self._deb.update(
                    current,
                    prediction_confidence,
                )

                if fired is not None:
                    confirmed = fired
                    self._composer.feed(
                        fired
                    )

            else:
                self._deb.update(
                    None,
                    0.0,
                )

        return {
            "current": current,
            "confidence": confidence,
            "confirmed": confirmed,
            "composed": self._composer.text,
        }