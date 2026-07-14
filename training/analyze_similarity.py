"""M3: 후보 단어 유사도 스크리닝 — 학습 도구가 아니다.

추출된 40개 특징 시퀀스를 쌍별로 비교해 동작이 지나치게 유사한 단어 쌍을
리포트한다. 사용자가 이 리포트를 보고 겹치는 단어를 쳐내 최종 30개를
확정하기 위한 선별 도구다 (REAL01 단일 화자 데이터 — 학습용 아님).

지표:
- dtw_norm: 정규화 DTW 거리 (프레임 간 유클리드 비용, (T1+T2)로 나눔).
  특징이 이미 어깨 기준 정규화되어 있어 유클리드가 직접적인 포즈 거리다.
  z-score 는 거의 움직이지 않는 얼굴 140차원을 부풀려 손 동작 가중치를
  훼손하므로 쓰지 않는다.
- cosine_mean: 시퀀스를 평균 풀링한 248차원 벡터 간 코사인 유사도.
마스크 2차원(FEATURE_DIM-2:)은 두 지표 모두에서 제외한다.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from shared.features import FEATURE_DIM
from training.build_dataset import DEFAULT_OUT, load_candidates, resolve_data_root

N_COORD_DIMS = FEATURE_DIM - 2  # 마스크 플래그 제외한 좌표부


def dtw_distance(a, b):
    """정규화 DTW: 프레임 간 유클리드 비용, (T1+T2)로 나눔. O(T1*T2)."""
    # 비용 행렬 벡터화: dist(i,j) = sqrt(|a_i|^2 + |b_j|^2 - 2 a_i·b_j)
    aa = (a * a).sum(axis=1)
    bb = (b * b).sum(axis=1)
    cost = np.sqrt(np.maximum(aa[:, None] + bb[None, :] - 2.0 * (a @ b.T), 0.0))

    n, m = cost.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1],
                                               D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def cosine_of_means(a, b):
    """평균 풀링한 벡터 간 코사인 유사도."""
    va, vb = a.mean(axis=0), b.mean(axis=0)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(va @ vb / denom) if denom > 0 else 0.0


def load_sequences(out_dir, candidates):
    """manifest 의 ok 항목을 로드. CSV 에 없는 stem 이 있으면 실패 (스테일 방어)."""
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    by_stem = {c.stem: c for c in candidates}
    stale = [s for s, m in manifest.items()
             if m.get("status") == "ok" and s not in by_stem]
    if stale:
        sys.exit(f"[오류] 후보 CSV에 없는 manifest 항목 {stale} — "
                 "스테일 산출물. data/processed 를 재생성하세요.")

    sequences = {}
    for stem, m in manifest.items():
        if m.get("status") != "ok":
            continue
        arr = np.load(out_dir / f"{stem}.npy")[:, :N_COORD_DIMS]
        c = by_stem[stem]
        sequences[c.korean] = (c.word_id, arr)
    return sequences


def main():
    parser = argparse.ArgumentParser(description="후보 단어 특징 시퀀스 유사도 리포트")
    parser.add_argument("--data-root", help="데이터셋 루트 (기본: config.yaml)")
    parser.add_argument("--processed", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--out", type=Path,
                        default=DEFAULT_OUT / "similarity_report.csv")
    args = parser.parse_args()

    candidates = load_candidates(resolve_data_root(args.data_root))
    sequences = load_sequences(args.processed, candidates)
    words = list(sequences)  # CSV(선정순번) 순서 유지 (dict 삽입 순서)

    lengths = {w: len(sequences[w][1]) for w in words}
    n_short = sum(1 for t in lengths.values() if t < 60)
    print(f"[유사도] {len(words)}개 시퀀스, {len(words) * (len(words) - 1) // 2}쌍 비교 "
          f"(프레임 {min(lengths.values())}~{max(lengths.values())})")
    if n_short:
        print(f"[경고] 60프레임 미만 시퀀스 {n_short}/{len(words)} — "
              "M4 윈도우 설계 재검토 필요 (보고만)")

    rows = []
    for i, wa in enumerate(words):
        for wb in words[i + 1:]:
            ida, a = sequences[wa]
            idb, b = sequences[wb]
            rows.append({
                "word_a": wa, "word_b": wb,
                "word_id_a": ida, "word_id_b": idb,
                "dtw_norm": round(dtw_distance(a, b), 4),
                "cosine_mean": round(cosine_of_means(a, b), 4),
                "frames_a": len(a), "frames_b": len(b),
            })
    rows.sort(key=lambda r: r["dtw_norm"])

    print(f"\n동작 유사 상위 {args.top_k}쌍 (DTW 오름차순 = 유사할수록 위):")
    print(f"{'순위':>4} {'단어A':<8} {'단어B':<8} {'DTW↓':>8} {'cos평균':>8}")
    for rank, r in enumerate(rows[:args.top_k], 1):
        print(f"{rank:>4} {r['word_a']:<8} {r['word_b']:<8} "
              f"{r['dtw_norm']:>8.4f} {r['cosine_mean']:>8.4f}")

    # utf-8-sig: 엑셀에서 한글이 깨지지 않도록 BOM 포함 저장
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n→ 전체 {len(rows)}쌍 저장: {args.out}")


if __name__ == "__main__":
    main()
