"""시간축 윈도잉 — 학습(training/train.py)과 추론(server/inference.py) 공용.

CLAUDE.md §5의 "학습과 추론에서 완전히 동일해야 함" 원칙에 따라 한 곳에 둔다.
train.py에서 문자 그대로 이동함 (2026-07, M5) — 수식을 바꾸면 재학습이 필요하다.
TF import 금지 (numpy만) — 서버 상태기계 단위테스트가 TF 없이 돌아야 한다.
"""

import numpy as np

from shared.features import FEATURE_DIM

N_COORD = FEATURE_DIM - 2  # 좌표부 248 (마스크 2차원 제외)


def resample(seq, new_len):
    """시간축 선형 보간. 좌표부는 선형, 마스크 2차원은 최근접(0/1 유지)."""
    new_len = max(2, new_len)
    pos = np.linspace(0.0, len(seq) - 1, new_len)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, len(seq) - 1)
    frac = (pos - i0).astype(np.float32)[:, None]
    out = seq[i0] * (1.0 - frac) + seq[i1] * frac
    out[:, N_COORD:] = seq[np.round(pos).astype(int), N_COORD:]
    return out.astype(np.float32)


def window(seq, win, rng=None):
    """(T, 250) → (win, 250). T>win: 크롭(rng 있으면 랜덤, 없으면 중앙),
    T<win: 끝을 0 패딩. Conv1D는 Keras 마스킹을 지원하지 않아 0 패딩을
    쓴다 — 실제 프레임은 포즈(어깨 포함)가 항상 0이 아니므로 전-0 패딩
    프레임과 혼동될 수 없다."""
    t = len(seq)
    if t >= win:
        start = (int(rng.integers(0, t - win + 1)) if rng is not None and t > win
                 else (t - win) // 2)
        return seq[start:start + win]
    out = np.zeros((win, FEATURE_DIM), dtype=np.float32)
    out[:t] = seq
    return out
