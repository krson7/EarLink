"""키포인트 선별·정규화 — 학습(training/)과 추론(server/)이 공유하는 특징 파이프라인.

web/features.js 와 완전히 동일한 로직이어야 한다. 이 파일을 수정하면 반드시
features.js 도 함께 수정하고 tests/test_features_parity.py 를 실행할 것.
연산 순서를 바꾸지 말 것 — 같은 IEEE-754 float64 연산 순서를 유지해야
Python↔JS 결과가 비트 단위로 일치한다. (math.hypot 금지: sqrt(dx*dx+dy*dy) 고정)

특징 계약 (M2에서 확정 — 이후 마일스톤의 학습 데이터가 이 레이아웃에 의존):
  124포인트 = 포즈 12 + 양손 42 + 얼굴 70
  벡터 250차원 = [포즈 24][왼손 42][오른손 42][얼굴 140][마스크L][마스크R]

CLAUDE.md §5의 포즈 목록(13개 인덱스)과 표기("상체 12개, 총 124포인트")가
모순되어, nose(0)를 제외한 12개로 확정했다(사용자 결정, 2026-07-09).
머리 위치 정보는 눈(2,5)·귀(7,8)가 대체하고, KSL-Guide 원논문의
얼굴 70점(OpenPose 표준) 구성을 유지한다.
"""

import math

# MediaPipe Pose 33포인트 중 상체 12개.
# 눈(2,5), 귀(7,8), 입(9,10), 어깨(11,12), 팔꿈치(13,14), 손목(15,16)
POSE_SUBSET_INDICES = [2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# 얼굴 70포인트 — MediaPipe canonical face mesh 인덱스 기준.
# 선정 근거 (비수지 신호 우선순위):
#   - 입술: 입 모양(한국어 입모양 동반)이 가장 핵심 비수지 신호이므로
#     canonical FACEMESH_LIPS 40점(바깥 20 + 안쪽 20) 전체 유지
#   - 눈썹: 눈썹 올림 = 의문문 표지이므로 canonical FACEMESH_*_EYEBROW
#     10점×2(위/아래 라인) 전체 유지
#   - 눈: 개폐(크게 뜨기/찡그림) 감지가 목적이므로 canonical 16점 링에서
#     외안각·내안각·아랫꺼풀 중앙·윗꺼풀 내측·윗꺼풀 중앙 5점×2로 축소
#   → 40 + 20 + 10 = 70 (총 124포인트를 맞추는 구성)
# 주의: 이 리스트의 순서가 특징 벡터의 컬럼 의미를 영구히 결정한다.
#       절대 정렬·재배치하지 말 것.
_FACE_RIGHT_EYEBROW = [46, 52, 53, 55, 63, 65, 66, 70, 105, 107]
_FACE_LEFT_EYEBROW = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336]
_FACE_RIGHT_EYE = [33, 133, 145, 158, 159]
_FACE_LEFT_EYE = [263, 362, 374, 385, 386]
_FACE_LIPS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88,
              91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308,
              310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]
FACE_SUBSET_INDICES = (_FACE_RIGHT_EYEBROW + _FACE_LEFT_EYEBROW
                       + _FACE_RIGHT_EYE + _FACE_LEFT_EYE + _FACE_LIPS)

N_HAND_POINTS = 21
FEATURE_DIM = 250  # 12*2 + 21*2*2 + 70*2 + 마스크 2
MIN_SHOULDER_DIST = 1e-6
# 얼굴 입력 최소 길이: 최대 인덱스 415 + 1. (468/478포인트 출력 모두 수용)
MIN_FACE_LEN = 416

# 임포트 시 자기검증 — 인덱스 오타/누락 방지
assert len(POSE_SUBSET_INDICES) == 12
assert len(FACE_SUBSET_INDICES) == 70 and len(set(FACE_SUBSET_INDICES)) == 70
assert len(POSE_SUBSET_INDICES) * 2 + N_HAND_POINTS * 2 * 2 + len(FACE_SUBSET_INDICES) * 2 + 2 == FEATURE_DIM


def extract_features(pose, left_hand, right_hand, face):
    """키포인트 그룹 → 250차원 특징 벡터. 프레임 무효 시 None.

    입력: 그룹별 [x, y] 쌍의 리스트 또는 None
      - pose: 33개 (MediaPipe Pose 전체)
      - left_hand / right_hand: 21개
      - face: 468 또는 478개 (canonical face mesh [+ 홍채])

    정규화: 양어깨(11,12) 중점을 원점으로 평행이동한 뒤 어깨 간 거리로
    나눠 스케일을 통일한다. 포즈가 없거나 어깨 거리가 퇴화하면 프레임 무효.

    결측: 손이 없으면 해당 42차원을 0으로 채우고 마스크 플래그를 0.0으로
    둔다(검출 시 1.0). 얼굴이 없으면 140차원을 0으로 채운다 — 스펙(§5)이
    마스크를 손에만 요구하므로 플래그는 없지만, 전부 0인 얼굴 블록이
    사실상 마스크 역할을 한다.
    """
    if pose is None or len(pose) < 17:
        return None

    cx = (pose[11][0] + pose[12][0]) / 2.0
    cy = (pose[11][1] + pose[12][1]) / 2.0
    dx = pose[11][0] - pose[12][0]
    dy = pose[11][1] - pose[12][1]
    s = math.sqrt(dx * dx + dy * dy)
    if s < MIN_SHOULDER_DIST:
        return None

    features = []

    for i in POSE_SUBSET_INDICES:
        features.append((pose[i][0] - cx) / s)
        features.append((pose[i][1] - cy) / s)

    left_ok = left_hand is not None and len(left_hand) >= N_HAND_POINTS
    right_ok = right_hand is not None and len(right_hand) >= N_HAND_POINTS
    for hand, ok in ((left_hand, left_ok), (right_hand, right_ok)):
        if ok:
            for i in range(N_HAND_POINTS):
                features.append((hand[i][0] - cx) / s)
                features.append((hand[i][1] - cy) / s)
        else:
            features.extend([0.0] * (N_HAND_POINTS * 2))

    if face is not None and len(face) >= MIN_FACE_LEN:
        for i in FACE_SUBSET_INDICES:
            features.append((face[i][0] - cx) / s)
            features.append((face[i][1] - cy) / s)
    else:
        features.extend([0.0] * (len(FACE_SUBSET_INDICES) * 2))

    features.append(1.0 if left_ok else 0.0)
    features.append(1.0 if right_ok else 0.0)
    return features
