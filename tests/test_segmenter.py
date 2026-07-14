"""발화 구간 탐지 상태기계 단위테스트 — TF 불필요 (server.inference 는 지연 import).

합성 250차원 프레임으로 전이 순서·블립 폐기·강제 종료·쿨다운·손 검출
깜빡임 면역을 검증한다. 프레임 시퀀스는 위치 연속성을 지킨다 (대기 구간이
원위치로 순간이동하면 인위적 모션 스파이크가 생겨 비현실적이다).
"""

import numpy as np

from server.inference import (
    SegConfig,
    UtteranceSession,
    estimate_fps,
    motion_delta,
    normalize_segment,
    trim_segment_edges,
)
from shared.features import FEATURE_DIM

MASK_L, MASK_R = FEATURE_DIM - 2, FEATURE_DIM - 1

CFG = SegConfig()  # 기본값 (config.yaml 과 동일)


def frame(wrist_x, hands=True):
    """합성 프레임: 팔꿈치·손목(dims 16~23)과 손 블록에 wrist_x."""
    v = np.zeros(FEATURE_DIM, dtype=np.float32)
    v[12:16] = [-0.5, 0.0, 0.5, 0.0]        # 어깨 (고정)
    v[16:24] = wrist_x
    if hands:
        v[24:108] = wrist_x
        v[MASK_L] = v[MASK_R] = 1.0
    return v.tolist()


def idle_frames(n, base, jitter=0.003, seed=7):
    """base 위치에서 미세 지터만 있는 대기 프레임."""
    rng = np.random.default_rng(seed)
    return [frame(base + rng.uniform(-jitter, jitter)) for _ in range(n)]


def signing_frames(n, start, step=0.08):
    """프레임당 step(어깨폭 단위)씩 손이 이동 — 수어 동작 수준.
    끝 위치는 start + step*(n-1)."""
    return [frame(start + step * i) for i in range(n)]


def push_t(session, f):
    """~15fps(66.7ms 간격) 타임스탬프를 자동 증가시키며 push."""
    t = getattr(session, "_test_t", 0.0)
    session._test_t = t + 66.7
    return session.push(f, t)


def run(session, frames):
    events = []
    for f in frames:
        events += push_t(session, f)
    return events


def test_transition_and_segment():
    """대기 → 수어 → (끝 위치에서) 대기: signing/idle 전이와 세그먼트 1개."""
    s = UtteranceSession(CFG)
    end_pos = 0.1 + 0.08 * 24
    events = run(s, idle_frames(10, base=0.1)
                 + signing_frames(25, start=0.1)
                 + idle_frames(12, base=end_pos))
    kinds = [k for k, _ in events]
    assert kinds == ["status", "status", "segment"]
    assert events[0][1] == "signing" and events[1][1] == "idle"
    seg, times = events[2][1]
    assert seg.dtype == np.float32 and seg.shape[1] == FEATURE_DIM
    assert len(times) == len(seg) and np.all(np.diff(times) > 0)
    # 세그먼트 ≈ 수어 프레임 + 룩백 + EMA 감쇠·patience 꼬리
    assert 22 <= len(seg) <= 25 + CFG.lookback_frames + CFG.end_patience_frames + 4


def test_short_blip_discarded():
    """min_segment_frames 미만 블립은 전이만 있고 세그먼트는 없다.

    기본 파라미터에서는 트리거 후 EMA 감쇠+patience 꼬리만으로 세그먼트가
    최소 길이에 도달하므로, 폐기 로직 검증에는 min 을 올린 설정을 쓴다.
    """
    cfg = SegConfig(min_segment_frames=20)
    s = UtteranceSession(cfg)
    # 한 프레임 손 튐(twitch) 후 즉시 원위치 — 세그먼트 ~12프레임 < 20
    events = run(s, idle_frames(10, base=0.1)
                 + [frame(0.4)]
                 + idle_frames(20, base=0.1, seed=8))
    kinds = [k for k, _ in events]
    assert "segment" not in kinds
    assert kinds.count("status") == 2  # signing → idle 전이는 발생


def test_max_length_forced_emit():
    """max_segment_frames 도달 시 강제 방출."""
    s = UtteranceSession(CFG)
    events = run(s, idle_frames(5, base=0.1) + signing_frames(120, start=0.1))
    segments = [p for k, p in events if k == "segment"]
    assert len(segments) == 1
    assert len(segments[0][0]) == CFG.max_segment_frames


def test_cooldown_blocks_restart():
    """쿨다운 중에는 동작이 계속돼도 새 발화가 시작되지 않는다."""
    s = UtteranceSession(CFG)
    end_pos = 0.1 + 0.08 * 24
    run(s, idle_frames(5, base=0.1) + signing_frames(25, start=0.1)
        + idle_frames(12, base=end_pos))
    assert s.state == "cooldown"
    # 쿨다운이 끝날 때까지 고모션 프레임을 밀어넣어도 signing 전이가 없어야 한다
    pos = end_pos
    while s.state == "cooldown":
        pos += 0.08
        events = push_t(s, frame(pos))
        assert not any(k == "status" and p == "signing" for k, p in events)
    assert s.state == "idle"  # 쿨다운 만료 프레임 자체는 전이를 일으키지 않는다


def test_motion_ignores_hand_flicker():
    """손 검출 깜빡임(마스크 0↔1, 42차원 0↔값 점프)이 모션에 안 들어간다."""
    a = np.asarray(frame(0.3, hands=True), dtype=np.float32)
    b = np.asarray(frame(0.3, hands=False), dtype=np.float32)
    assert motion_delta(a, b) < 1e-6  # 팔은 정지, 손은 한쪽만 검출 → 제외


def test_idle_jitter_never_triggers():
    s = UtteranceSession(CFG)
    assert run(s, idle_frames(100, base=0.1)) == []


# -------------------------------------------------- 밀도 정규화·트리밍 (파리티 수정)

def test_estimate_fps():
    assert abs(estimate_fps(np.arange(10) * 66.7) - 15.0) < 0.1
    assert abs(estimate_fps(np.arange(10) * 33.3) - 30.0) < 0.2
    assert estimate_fps(np.arange(10) * 1.0) is None      # 1000fps — 불신
    assert estimate_fps(np.arange(10) * 500.0) is None    # 2fps — 불신
    assert estimate_fps(np.array([1.0])) is None


def test_normalize_segment_upsamples_low_fps():
    """15fps 세그먼트(20프레임)는 30fps 등가(~39행)로 리샘플 후 패딩돼야 한다."""
    seg = np.asarray([frame(0.1 + 0.05 * i) for i in range(20)], dtype=np.float32)
    x = normalize_segment(seg, 45, fps=15.0)
    assert x.shape == (45, FEATURE_DIM)
    real = int((np.abs(x[:, :248]).sum(axis=1) > 0).sum())
    assert 38 <= real <= 40          # 20프레임 × 30/15 ≈ 40행 (fps 미지정이면 20행)
    assert set(np.unique(x[:real, 248:250])) == {1.0}  # 마스크 보존


def test_normalize_segment_without_fps_keeps_length():
    seg = np.asarray([frame(0.1 + 0.05 * i) for i in range(20)], dtype=np.float32)
    x = normalize_segment(seg, 45)  # fps 미상 → 리스케일 생략(기존 동작)
    real = int((np.abs(x[:, :248]).sum(axis=1) > 0).sum())
    assert real == 20


def test_trim_segment_edges():
    """양끝의 저모션(대기) 프레임이 잘리고 활동 구간+여유만 남아야 한다."""
    idle_head = [frame(0.1) for _ in range(6)]
    active = [frame(0.1 + 0.1 * i) for i in range(10)]
    idle_tail = [frame(1.0) for _ in range(6)]
    seg = np.asarray(idle_head + active + idle_tail, dtype=np.float32)
    out = trim_segment_edges(seg, threshold=0.02, margin=2)
    assert 10 <= len(out) <= 10 + 2 * 2 + 2   # 활동 10 + 양끝 여유
    assert len(out) < len(seg)
    # 전부 저모션이면 원본 유지
    still = np.asarray([frame(0.1) for _ in range(8)], dtype=np.float32)
    assert len(trim_segment_edges(still, 0.02, 2)) == 8
