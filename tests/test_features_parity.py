"""shared/features.py ↔ web/features.js 특징 일치 검증.

같은 더미 키포인트 입력(파이썬에서 시드 고정 생성)을 양쪽에 넣어
출력이 허용오차 1e-5 내로 일치하는지 확인한다. 동일한 float64 연산
순서를 쓰므로 실제 diff는 정확히 0이어야 한다.

이 테스트는 M2의 수용 게이트다 — node가 없으면 skip 하지 않고 실패시킨다.
"""

import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from shared.features import (
    FACE_SUBSET_INDICES,
    FEATURE_DIM,
    POSE_SUBSET_INDICES,
    extract_features,
)

TESTS_DIR = Path(__file__).resolve().parent
RUNNER = TESTS_DIR / "features_runner.mjs"
TOLERANCE = 1e-5

# ---------------------------------------------------------------- 더미 입력
# 파이썬에서 한 번 생성해 같은 JSON을 양쪽에 공급한다 (JS에서 랜덤 재생성 금지).
# JSON float 왕복은 무손실이므로 양쪽이 완전히 같은 double 을 본다.

_rng = random.Random(20260709)


def _pts(n, lo=0.0, hi=1.0):
    return [[_rng.uniform(lo, hi), _rng.uniform(lo, hi)] for _ in range(n)]


def _full_case():
    return {
        "pose": _pts(33),
        "left_hand": _pts(21),
        "right_hand": _pts(21),
        "face": _pts(478),
    }


def _make_cases():
    cases = []

    cases.append({"name": "all_present", **_full_case()})

    c = _full_case()
    c["left_hand"] = None
    cases.append({"name": "left_hand_missing", **c})

    c = _full_case()
    c["left_hand"] = None
    c["right_hand"] = None
    cases.append({"name": "both_hands_missing", **c})

    c = _full_case()
    c["face"] = None
    cases.append({"name": "face_missing", **c})

    cases.append({"name": "pose_only", "pose": _pts(33),
                  "left_hand": None, "right_hand": None, "face": None})

    # MediaPipe는 [0,1] 밖 좌표도 내보낸다 (화면 밖 추정)
    cases.append({"name": "out_of_frame_coords",
                  "pose": _pts(33, -0.2, 1.2), "left_hand": _pts(21, -0.2, 1.2),
                  "right_hand": _pts(21, -0.2, 1.2), "face": _pts(478, -0.2, 1.2)})

    c = _full_case()
    c["pose"] = None
    cases.append({"name": "pose_missing", **c})  # → None/null

    c = _full_case()
    c["pose"][12] = list(c["pose"][11])  # 어깨 두 점이 동일 → 스케일 퇴화
    cases.append({"name": "degenerate_shoulders", **c})  # → None/null

    return cases


CASES = _make_cases()
INVALID_CASES = {"pose_missing", "degenerate_shoulders"}


# ---------------------------------------------------------------- JS 실행

@pytest.fixture(scope="module")
def js_output():
    node = shutil.which("node")
    if node is None:
        pytest.fail("node 가 필요합니다 (패리티 게이트는 skip 하지 않음). brew install node")
    proc = subprocess.run(
        [node, str(RUNNER)],
        input=json.dumps({"cases": CASES}),
        text=True, capture_output=True, timeout=60,
    )
    assert proc.returncode == 0, f"features_runner.mjs 실패:\n{proc.stderr}"
    return json.loads(proc.stdout)


# ---------------------------------------------------------------- 패리티

def test_constants_parity(js_output):
    """인덱스 상수와 차원이 순서까지 완전히 같아야 한다 (컬럼 의미 고정)."""
    assert js_output["constants"]["POSE_SUBSET_INDICES"] == POSE_SUBSET_INDICES
    assert js_output["constants"]["FACE_SUBSET_INDICES"] == FACE_SUBSET_INDICES
    assert js_output["constants"]["FEATURE_DIM"] == FEATURE_DIM


@pytest.mark.parametrize("idx", range(len(CASES)), ids=[c["name"] for c in CASES])
def test_parity(js_output, idx):
    case = CASES[idx]
    py = extract_features(case["pose"], case["left_hand"],
                          case["right_hand"], case["face"])
    js = js_output["results"][idx]

    if case["name"] in INVALID_CASES:
        assert py is None and js is None
        return

    assert py is not None and js is not None
    assert len(py) == FEATURE_DIM and len(js) == FEATURE_DIM
    diffs = [abs(a - b) for a, b in zip(py, js)]
    max_diff = max(diffs)
    assert max_diff <= TOLERANCE, (
        f"{case['name']}: max diff {max_diff:.3e} (index {diffs.index(max_diff)})"
    )


# ---------------------------------------------------------------- 구조 검증
# (패리티가 깨졌을 때 원인 진단을 돕는 파이썬 단독 검증)

def _py(name):
    case = next(c for c in CASES if c["name"] == name)
    return extract_features(case["pose"], case["left_hand"],
                            case["right_hand"], case["face"])


def test_masks_all_present():
    v = _py("all_present")
    assert v[248] == 1.0 and v[249] == 1.0


def test_left_hand_missing_zero_filled():
    v = _py("left_hand_missing")
    assert v[24:66] == [0.0] * 42       # 왼손 블록
    assert v[66:108] != [0.0] * 42      # 오른손은 채워짐
    assert v[248] == 0.0 and v[249] == 1.0


def test_face_missing_zero_filled():
    v = _py("face_missing")
    assert v[108:248] == [0.0] * 140


def test_normalization_shoulder_frame():
    """어깨 중점이 원점, 어깨 간 거리가 1이 되도록 정규화됐는지 확인."""
    case = next(c for c in CASES if c["name"] == "all_present")
    v = extract_features(case["pose"], None, None, None)
    # 레이아웃상 어깨(11,12)는 포즈 서브셋의 7·8번째 포인트 (인덱스 12~15)
    lx, ly, rx, ry = v[12], v[13], v[14], v[15]
    assert abs((lx + rx) / 2) < 1e-12 and abs((ly + ry) / 2) < 1e-12
    dist = ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5
    assert abs(dist - 1.0) < 1e-12
