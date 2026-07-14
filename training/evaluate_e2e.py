"""종단(end-to-end) 재현율 평가 — 원본 영상을 실시간 추론 파이프라인에 통과.

각 영상을 클라이언트처럼 시뮬레이션(~15fps 샘플링 + 4자리 반올림)해서
발화 탐지 → 다중 윈도우 추론까지 서버 경로 그대로 돌리고, 영상별 최고
confidence 결과가 정답 라벨과 일치하는지 센다. B안 파이프라인 정상의
기준(원본 재현율 90%+)이며, A안(다화자) 재학습 후에도 같은 방식으로 쓴다.
"""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import yaml

from server.inference import Predictor, SegConfig, UtteranceSession
from shared.features import extract_features
from training.build_dataset import load_candidates, resolve_data_root
from training.extract_keypoints import DEFAULT_MODELS_DIR, HolisticExtractor

ROOT = Path(__file__).resolve().parent.parent
FINAL_CSV = ROOT / "training" / "earlink_final_30words.csv"
VIDEO_FPS = 30.0
CLIENT_STEP = 2          # 매 2프레임 샘플 ≈ 15fps (web/app.js DETECT_INTERVAL_MS=66)
FLUSH_MAX = 60           # EOF 후 마지막 벡터 반복 상한 (실사용에선 카메라가 계속 돎)


def simulate_client(video_path, step=CLIENT_STEP):
    """클라이언트 시뮬레이션: step 간격 샘플 프레임만 랜드마커에 통과,
    특징은 4자리 반올림(전송 규약). [(t_ms, vec), ...] 반환."""
    ex = HolisticExtractor(DEFAULT_MODELS_DIR)
    cap = cv2.VideoCapture(str(video_path))
    out, idx, prev_ts = [], 0, -1
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % step == 0:
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ts = int(round(idx * 1000.0 / VIDEO_FPS))
            if ts <= prev_ts:
                ts = prev_ts + 1
            prev_ts = ts
            v = extract_features(*ex.detect(img, ts))
            if v is not None:  # 클라이언트는 무효 프레임을 보내지 않음
                out.append((float(ts), [round(float(x), 4) for x in v]))
        idx += 1
    cap.release()
    ex.close()
    return out


def run_pipeline(feats, seg_cfg, predictor):
    """특징 스트림 → 발화 탐지 → 추론. 세그먼트별 ResultMsg 리스트 반환."""
    session = UtteranceSession(seg_cfg)
    results = []

    def handle(events):
        for kind, payload in events:
            if kind == "segment":
                seg, times = payload
                results.append(predictor.predict(seg, times))

    last_v = last_t = None
    for t, v in feats:
        handle(session.push(v, t))
        last_v, last_t = v, t
    n_flush = 0
    while session.state == "signing" and n_flush < FLUSH_MAX and last_v is not None:
        last_t += 1000.0 / (VIDEO_FPS / CLIENT_STEP)
        handle(session.push(last_v, last_t))
        n_flush += 1
    return results


def main():
    parser = argparse.ArgumentParser(description="원본 영상 종단 재현율 평가")
    parser.add_argument("--ids", help="쉼표 구분 WORD ID (기본: 최종 30개 전체)")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    seg_cfg = SegConfig.from_yaml(cfg)
    predictor = Predictor.load_latest(
        ROOT / "models", win=cfg["model"]["window_frames"],
        conf_thr=cfg["model"]["confidence_threshold"], seg_cfg=seg_cfg)
    if predictor is None:
        raise SystemExit("[오류] 모델 없음 — training.train 먼저 실행")

    candidates = load_candidates(resolve_data_root(None), FINAL_CSV)
    if args.ids:
        wanted = set(args.ids.split(","))
        candidates = [c for c in candidates if c.word_id in wanted]

    n_correct = 0
    failures = []
    for c in candidates:
        feats = simulate_client(c.path)
        results = run_pipeline(feats, seg_cfg, predictor)
        if results:
            best = max(results, key=lambda r: r.confidence)
            ok = best.top1 == c.korean
            n_correct += ok
            mark = "✓" if ok else "✗"
            segs = " / ".join(f"{r.top1} {r.confidence:.2f}" for r in results)
            print(f"{mark} {c.korean:<6}({c.word_id})  최종: {best.top1} "
                  f"{best.confidence:.3f}  [세그먼트 {len(results)}개: {segs}]")
            if not ok:
                failures.append((c.korean, best.top1, best.confidence))
        else:
            print(f"✗ {c.korean:<6}({c.word_id})  세그먼트 미탐지")
            failures.append((c.korean, "(미탐지)", 0.0))

    n = len(candidates)
    print(f"\n종단 재현율: {n_correct}/{n} = {n_correct / n * 100:.1f}%"
          f"  (목표 90%+)")
    if failures:
        print("실패:", ", ".join(f"{w}→{p}({cf:.2f})" for w, p, cf in failures))


if __name__ == "__main__":
    main()
