// 패리티 테스트 러너 — stdin으로 케이스 JSON을 받아 web/features.js 를 실행하고
// 상수와 결과를 stdout으로 출력한다. test_features_parity.py 가 호출한다.
// (정적 import 경로는 이 파일의 위치 기준으로 해석되므로 cwd와 무관하다)
import { extractFeatures, POSE_SUBSET_INDICES, FACE_SUBSET_INDICES, FEATURE_DIM }
  from "../web/features.js";

const chunks = [];
for await (const c of process.stdin) chunks.push(c);
const { cases } = JSON.parse(Buffer.concat(chunks).toString("utf-8"));

const results = cases.map((c) =>
  extractFeatures(c.pose, c.left_hand, c.right_hand, c.face));

process.stdout.write(JSON.stringify({
  constants: { POSE_SUBSET_INDICES, FACE_SUBSET_INDICES, FEATURE_DIM },
  results,
}));
