"""모델 로드(최신 버전), 발화 구간 탐지 상태기계, 세그먼트 정규화, 추론.

CLAUDE.md §6: 프레임 간 (x,y) 변화량 평균의 이동평균(여기서는 EMA)이
임곗값을 넘으면 발화 시작, 다시 내려가면 종료로 판정한다. 탐지된 구간만
분류기에 넣고, 임곗값은 config.yaml `segmentation:` 으로 조정한다.

특징 처리 파리티: 클라이언트가 web/features.js(= shared/features.py 패리티
증명됨)로 정규화한 250차원 벡터를 보내므로 서버는 v를 그대로 쓴다.
윈도잉(45프레임·뒤 0패딩)은 학습과 동일한 shared/windowing.py 를 쓴다.

주의: 이 모듈 최상단에서 TF/keras 를 import 하지 않는다 — 지연 import
(Predictor.load_latest 내부). 발화 탐지 상태기계 단위테스트가 TF 없이
돌아야 하고, 서버 기동 로그에서도 로드 시점이 명확해진다.
"""

import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared.features import FEATURE_DIM, N_HAND_POINTS, POSE_SUBSET_INDICES
from shared.windowing import resample, window

# 특징 레이아웃 (train.py 와 동일하게 shared 상수에서 유도)
LH_OFF = len(POSE_SUBSET_INDICES) * 2          # 24
RH_OFF = LH_OFF + N_HAND_POINTS * 2            # 66
MASK_L, MASK_R = FEATURE_DIM - 2, FEATURE_DIM - 1

# 모션 지표에 쓰는 x-열 인덱스 (y는 +1).
# 팔꿈치(13,14)·손목(15,16) = POSE_SUBSET 위치 8~11 → dims 16~23.
_ARM_X = np.arange(16, 24, 2)
_LH_X = np.arange(LH_OFF, LH_OFF + N_HAND_POINTS * 2, 2)
_RH_X = np.arange(RH_OFF, RH_OFF + N_HAND_POINTS * 2, 2)


def _point_dists(d, x_idx):
    return np.sqrt(d[x_idx] ** 2 + d[x_idx + 1] ** 2)


def motion_delta(prev, cur):
    """프레임 간 모션: 포인트별 유클리드 거리 평균 (어깨폭 단위/프레임).

    팔꿈치+손목 4포인트는 항상 포함(유효 프레임은 포즈가 반드시 존재),
    손 21포인트×2는 두 프레임 모두 검출(마스크=1)됐을 때만 — 손 검출
    깜빡임(0↔값 42차원 점프)이 지표에 들어가지 않는다. 얼굴은 제외
    (대기 중 고개 움직임·표정이 오탐을 만든다 — 수어 신호는 팔·손).
    """
    d = cur - prev
    dists = [_point_dists(d, _ARM_X)]
    if prev[MASK_L] > 0 and cur[MASK_L] > 0:
        dists.append(_point_dists(d, _LH_X))
    if prev[MASK_R] > 0 and cur[MASK_R] > 0:
        dists.append(_point_dists(d, _RH_X))
    return float(np.concatenate(dists).mean())


@dataclass(frozen=True)
class SegConfig:
    """발화 구간 탐지 파라미터 (단위: 어깨폭/프레임, 클라이언트 ~15fps 기준).

    기본값은 config.yaml `segmentation:` 과 동기화 유지할 것.
    end_threshold/end_patience 는 수어 중간의 짧은 멈춤(형태소 경계)에서
    발화가 두 토막 나지 않도록 완화되어 있다 (파리티 버그 수정, 2026-07-10).
    """
    ema_alpha: float = 0.4          # 지수이동평균 계수 (유효 창 ~4프레임)
    start_threshold: float = 0.06   # EMA 가 넘으면 발화 시작
    end_threshold: float = 0.02     # EMA 가 내려가면 종료 후보 (히스테리시스)
    end_patience_frames: int = 8    # 연속으로 end 미만이어야 종료 (~0.5초)
    min_segment_frames: int = 8     # 미만이면 블립으로 폐기 (~0.5초)
    max_segment_frames: int = 90    # 초과 시 강제 방출 (~6초)
    lookback_frames: int = 3        # EMA 지연 보상 — 시작 시 직전 프레임 포함
    cooldown_frames: int = 10       # 결과 후 재시작 금지 구간 (~0.7초)
    edge_trim_threshold: float = 0.02   # 세그먼트 양끝 트리밍 모션 임곗값
    edge_trim_margin_frames: int = 2    # 트리밍 시 남기는 여유 프레임

    @classmethod
    def from_yaml(cls, cfg):
        seg = cfg.get("segmentation") or {}
        return cls(**{k: seg[k] for k in cls.__dataclass_fields__ if k in seg})


class UtteranceSession:
    """WebSocket 연결당 1개 — 슬라이딩 버퍼 + 발화 탐지 상태기계.

    push(v, t) 는 발생한 이벤트를 순서대로 반환한다:
      ("status", "signing"|"idle") — 상태 전이 시
      ("segment", (np.ndarray(T, 250), np.ndarray(T,))) — 발화 종료 시.
        두 번째 배열은 프레임별 클라이언트 타임스탬프(ms) — 밀도 정규화용.
    """

    def __init__(self, cfg: SegConfig):
        self.cfg = cfg
        self.state = "idle"                       # idle | signing | cooldown
        self.ema = 0.0
        self.prev = None
        self.recent = deque(maxlen=cfg.lookback_frames + 1)  # IDLE 중 룩백 버퍼
        self.segment = []                         # [(vec, t_ms), ...]
        self.below = 0                            # end 미만 연속 카운터
        self.cool = 0

    def _emit_segment(self):
        seg = np.asarray([v for v, _ in self.segment], dtype=np.float32)
        times = np.asarray([t for _, t in self.segment], dtype=np.float64)
        self.segment = []
        return seg, times

    def push(self, v, t):
        cur = np.asarray(v, dtype=np.float32)
        motion = 0.0 if self.prev is None else motion_delta(self.prev, cur)
        self.prev = cur
        self.ema = self.cfg.ema_alpha * motion + (1 - self.cfg.ema_alpha) * self.ema

        events = []
        if self.state == "cooldown":
            self.cool -= 1
            if self.cool <= 0:
                self.state = "idle"
            self.recent.append((cur, t))
            return events

        if self.state == "idle":
            self.recent.append((cur, t))
            if self.ema >= self.cfg.start_threshold:
                self.state = "signing"
                self.segment = list(self.recent)  # 룩백 포함 (EMA 지연 보상)
                self.below = 0
                events.append(("status", "signing"))
            return events

        # state == "signing"
        self.segment.append((cur, t))
        self.below = self.below + 1 if self.ema < self.cfg.end_threshold else 0
        ended = self.below >= self.cfg.end_patience_frames
        forced = len(self.segment) >= self.cfg.max_segment_frames
        if not (ended or forced):
            return events

        self.state = "cooldown"
        self.cool = self.cfg.cooldown_frames
        self.recent.clear()
        events.append(("status", "idle"))
        if forced or len(self.segment) >= self.cfg.min_segment_frames:
            events.append(("segment", self._emit_segment()))
        else:
            self.segment = []  # 블립 — 조용히 폐기
        return events


# ---------------------------------------------------------------- 세그먼트 정규화

TRAIN_FPS = 30.0        # 학습 데이터 프레임레이트 (§5) — 밀도 정규화 목표
FPS_VALID_RANGE = (5.0, 60.0)   # 밖이면 타임스탬프 불신 → 리스케일 생략
MIN_CANDIDATE_ROWS = 12          # 후보 서브구간 최소 길이 (30fps 기준 0.4초)


def estimate_fps(times):
    """프레임 타임스탬프(ms)로 실제 수신 fps 추정. 신뢰 불가 시 None."""
    if times is None or len(times) < 2:
        return None
    duration = (float(times[-1]) - float(times[0])) / 1000.0
    if duration <= 0:
        return None
    fps = (len(times) - 1) / duration
    return fps if FPS_VALID_RANGE[0] <= fps <= FPS_VALID_RANGE[1] else None


def trim_segment_edges(seg, threshold, margin):
    """세그먼트 양끝의 저모션 프레임(대기·전환 잔여)을 잘라낸다 —
    학습 데이터의 morpheme 구간 트리밍을 근사. 활동 프레임이 없으면 원본 유지."""
    if len(seg) < 3:
        return seg
    motions = np.asarray([motion_delta(seg[i - 1], seg[i])
                          for i in range(1, len(seg))])
    active = np.nonzero(motions >= threshold)[0]
    if len(active) == 0:
        return seg
    lo = max(0, int(active[0]) + 1 - margin)       # motions[i] = seg[i]↔seg[i+1]
    hi = min(len(seg), int(active[-1]) + 2 + margin)
    return seg[lo:hi]


def normalize_segment(seg, win, fps=None):
    """세그먼트 → (win, 250), 학습 분포에 맞춘 밀도 정규화 포함.

    fps 가 유효하면 30fps 등가 길이로 **항상 리샘플**한다 (짧은 세그먼트도 —
    ~15fps 클라이언트의 수어는 리샘플 없이는 학습 대비 절반 밀도가 되어
    시간 스케일 증강(0.8~1.2×) 분포를 완전히 벗어난다: 파리티 버그 원인 1).
    등가 길이 ≤ win 이면 리샘플 후 끝 0패딩(학습과 동일 경로), 초과면
    win 으로 단일 리샘플(이중 보간 방지).
    """
    if fps is not None:
        target = int(round(len(seg) * TRAIN_FPS / fps))
        if target > win:
            return resample(seg, win)
        if target != len(seg):
            seg = resample(seg, target)
    elif len(seg) > win:
        return resample(seg, win)
    return window(seg, win)


class Predictor:
    """기동 시 1회 로드되는 전역 싱글턴. predict 는 순수 동기 함수 —
    호출측(main.py)이 asyncio.to_thread 로 이벤트 루프 밖에서 돌린다."""

    # 다중 윈도우 후보: (시작, 길이) 그리드 (30fps 등가 행 단위).
    # 탐지 세그먼트에는 손 올리기/내리기 전환 동작이 수어의 1~2배 길이로
    # 섞여 있고(고모션이라 에지 트리밍으로 못 자름), 구간 추정이 어긋나도
    # 복원되도록 다양한 서브구간을 시도해 최고 confidence 를 채택한다.
    CAND_START_STRIDE = 5           # 시작 위치 간격
    CAND_LENGTHS = (25, 30, 35, 40, 45)  # 학습 클립 길이 분포(24~47행)를 커버

    def __init__(self, model, inv_labels, version, win, conf_thr, seg_cfg=None):
        self._model = model
        self._inv = inv_labels
        self.version = version
        self.win = win
        self.conf_thr = conf_thr
        self.seg_cfg = seg_cfg or SegConfig()

    @classmethod
    def load_latest(cls, models_dir: Path, win: int, conf_thr: float,
                    seg_cfg=None):
        """models/ksl_word_v{n}.keras 중 최신 버전 로드 (CLAUDE.md §7).
        없으면 None — 서버는 발화 탐지 전용 모드로 기동한다."""
        import json

        versions = []
        for p in models_dir.glob("ksl_word_v*.keras"):
            m = re.fullmatch(r"ksl_word_v(\d+)\.keras", p.name)
            if m:
                versions.append((int(m.group(1)), p))
        if not versions:
            return None
        version, model_path = max(versions)
        labels_path = models_dir / f"ksl_word_v{version}.labels.json"
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        inv = {idx: word for word, idx in labels.items()}

        import keras  # 지연 import (~수 초) — 기동 시 1회

        model = keras.models.load_model(model_path)
        t0 = time.time()
        # 워밍업: 첫 호출의 그래프 트레이싱 비용(수백 ms~수 초)을 선지불
        model(np.zeros((1, win, FEATURE_DIM), dtype=np.float32), training=False)
        print(f"[모델] ksl_word_v{version} 로드 + 워밍업 "
              f"{(time.time() - t0) * 1000:.0f}ms ({len(inv)}단어)")
        return cls(model, inv, version, win, conf_thr, seg_cfg)

    def _candidates(self, seg, times):
        """세그먼트 → 45프레임 후보 텐서 목록.

        ① 양끝 저모션 트리밍(morpheme 트리밍 근사) → ② 30fps 등가 리샘플
        → ③ (시작 × 길이) 그리드 서브구간 — 각 후보는 학습 포맷(수어가
        0행에서 시작 + 뒤 0패딩)과 같은 모양으로 정규화된다.
        """
        trimmed = trim_segment_edges(seg, self.seg_cfg.edge_trim_threshold,
                                     self.seg_cfg.edge_trim_margin_frames)
        fps = estimate_fps(times)
        if fps is not None:
            # times 는 트리밍 전 기준이지만 fps 는 구간 평균이라 그대로 유효
            target = max(2, int(round(len(trimmed) * TRAIN_FPS / fps)))
            s = resample(trimmed, target) if target != len(trimmed) else trimmed
        else:
            s = trimmed

        candidates = []
        n = len(s)
        for start in range(0, max(1, n - MIN_CANDIDATE_ROWS), self.CAND_START_STRIDE):
            for length in self.CAND_LENGTHS:
                if start + length <= n:
                    candidates.append(normalize_segment(s[start:start + length],
                                                        self.win))
        if not candidates:  # 짧은 세그먼트 — 전체를 하나의 후보로
            candidates.append(normalize_segment(s, self.win))
        return candidates

    def predict(self, seg, times=None):
        """다중 후보 배치 추론 → 최고 confidence 후보 채택."""
        from server.schemas import ResultMsg

        x = np.stack(self._candidates(seg, times)).astype(np.float32)
        # model.predict 대신 직접 호출 — 반복 호출 오버헤드 회피 (한 배치)
        probs = np.asarray(self._model(x, training=False))
        best = int(probs.max(axis=1).argmax())
        idx = int(probs[best].argmax())
        conf = float(probs[best, idx])
        low = conf < self.conf_thr
        return ResultMsg(
            text=None if low else self._inv[idx],  # 저신뢰 → "인식 안 됨"
            top1=self._inv[idx],
            confidence=round(conf, 4),
            n_frames=len(seg),
            low_confidence=low,
        )
