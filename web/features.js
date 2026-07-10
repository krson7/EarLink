// 키포인트 선별·정규화 — shared/features.py 와 완전히 동일한 로직.
//
// 이 파일을 수정하면 반드시 features.py 도 함께 수정하고
// tests/test_features_parity.py 를 실행할 것 (허용오차 1e-5 게이트).
// 연산 순서를 바꾸지 말 것 — 같은 IEEE-754 float64 연산 순서를 유지해야
// Python↔JS 결과가 비트 단위로 일치한다. (Math.hypot 금지: sqrt(dx*dx+dy*dy) 고정)
// 반올림 금지 — 전송용 4자리 반올림은 app.js에서만 한다.
//
// 브라우저 API를 사용하지 않는다 (패리티 테스트에서 Node로도 실행됨).
// 인덱스·선정 근거의 상세 주석은 shared/features.py 참조.

// MediaPipe Pose 33포인트 중 상체 12개 — 눈/귀/입/어깨/팔꿈치/손목 (nose 제외)
export const POSE_SUBSET_INDICES = [2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

// 얼굴 70포인트 — canonical face mesh. 순서가 특징 컬럼 의미를 결정하므로
// 절대 정렬·재배치하지 말 것. (오른눈썹10 + 왼눈썹10 + 오른눈5 + 왼눈5 + 입술40)
const FACE_RIGHT_EYEBROW = [46, 52, 53, 55, 63, 65, 66, 70, 105, 107];
const FACE_LEFT_EYEBROW = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336];
const FACE_RIGHT_EYE = [33, 133, 145, 158, 159];
const FACE_LEFT_EYE = [263, 362, 374, 385, 386];
const FACE_LIPS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88,
                   91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308,
                   310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415];
export const FACE_SUBSET_INDICES = [
  ...FACE_RIGHT_EYEBROW, ...FACE_LEFT_EYEBROW,
  ...FACE_RIGHT_EYE, ...FACE_LEFT_EYE, ...FACE_LIPS,
];

export const N_HAND_POINTS = 21;
export const FEATURE_DIM = 250; // 12*2 + 21*2*2 + 70*2 + 마스크 2
export const MIN_SHOULDER_DIST = 1e-6;
export const MIN_FACE_LEN = 416; // 최대 인덱스 415 + 1 (468/478 출력 모두 수용)

// 임포트 시 자기검증 — features.py 의 assert 와 동일
if (POSE_SUBSET_INDICES.length !== 12
    || FACE_SUBSET_INDICES.length !== 70
    || new Set(FACE_SUBSET_INDICES).size !== 70
    || POSE_SUBSET_INDICES.length * 2 + N_HAND_POINTS * 2 * 2
       + FACE_SUBSET_INDICES.length * 2 + 2 !== FEATURE_DIM) {
  throw new Error("features.js 상수 자기검증 실패");
}

/**
 * 키포인트 그룹 → 250차원 특징 벡터. 프레임 무효 시 null.
 * 입력: 그룹별 [x, y] 쌍의 배열 또는 null (pose 33 / hand 21 / face 468|478)
 * 동작 명세는 shared/features.py 의 docstring 과 동일.
 */
export function extractFeatures(pose, leftHand, rightHand, face) {
  if (pose == null || pose.length < 17) {
    return null;
  }

  const cx = (pose[11][0] + pose[12][0]) / 2.0;
  const cy = (pose[11][1] + pose[12][1]) / 2.0;
  const dx = pose[11][0] - pose[12][0];
  const dy = pose[11][1] - pose[12][1];
  const s = Math.sqrt(dx * dx + dy * dy);
  if (s < MIN_SHOULDER_DIST) {
    return null;
  }

  const features = [];

  for (const i of POSE_SUBSET_INDICES) {
    features.push((pose[i][0] - cx) / s);
    features.push((pose[i][1] - cy) / s);
  }

  const leftOk = leftHand != null && leftHand.length >= N_HAND_POINTS;
  const rightOk = rightHand != null && rightHand.length >= N_HAND_POINTS;
  for (const [hand, ok] of [[leftHand, leftOk], [rightHand, rightOk]]) {
    if (ok) {
      for (let i = 0; i < N_HAND_POINTS; i++) {
        features.push((hand[i][0] - cx) / s);
        features.push((hand[i][1] - cy) / s);
      }
    } else {
      for (let i = 0; i < N_HAND_POINTS * 2; i++) features.push(0.0);
    }
  }

  if (face != null && face.length >= MIN_FACE_LEN) {
    for (const i of FACE_SUBSET_INDICES) {
      features.push((face[i][0] - cx) / s);
      features.push((face[i][1] - cy) / s);
    }
  } else {
    for (let i = 0; i < FACE_SUBSET_INDICES.length * 2; i++) features.push(0.0);
  }

  features.push(leftOk ? 1.0 : 0.0);
  features.push(rightOk ? 1.0 : 0.0);
  return features;
}
