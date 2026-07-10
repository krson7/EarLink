"""M3: 정면 WORD 영상 → MediaPipe Tasks 재추출 → shared.features 정규화 → .npy.

추출기 파리티 원칙 (A안):
- 제공된 OpenPose JSON(라벨링데이터/)은 사용하지 않는다 — 학습과 추론이
  같은 추출기(MediaPipe)를 쓰도록 mp4에서 재추출한다.
- mediapipe==0.10.35 는 web/app.js 의 MP_VERSION 과 동일해야 한다.
- 모델 URL은 web/app.js 의 MODELS 와 동일 경로를 미러링한다.
- combo 모드의 handedness 규약은 web/app.js 폴백과 동일: 라벨을 그대로
  슬롯에 배치하고 좌표는 절대 반전하지 않는다.

추출기 모드는 실행(run) 단위로 하나만 쓴다 — 영상/프레임 중간 전환 금지
(혼합 추출기 데이터셋 방지). Python Holistic 은 불안정 사례가 보고되어
(mediapipe#5181) 시작 시 실제 detect 프로브로 가용성을 확인한다.
세그폴트는 포착할 수 없으므로, 3개 ID 단계 실행이 조기 검출 장치다.
"""

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from shared.features import FEATURE_DIM, extract_features
from training.build_dataset import (
    DEFAULT_OUT,
    resolve_data_root,
    scan_frontal_videos,
)

ROOT = Path(__file__).resolve().parent.parent

# web/app.js MODELS 와 동일 경로 (상호 참조 고정 — 한쪽을 바꾸면 다른 쪽도 확인).
# combo 의 pose 는 lite: 포즈는 정규화 앵커(어깨)라서 추론(웹 폴백 lite)과
# 다른 모델을 쓰면 스케일이 계통적으로 어긋난다. 파리티 > 오프라인 품질.
_MODEL_BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "holistic": f"{_MODEL_BASE}/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task",
    "pose": f"{_MODEL_BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "hand": f"{_MODEL_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face": f"{_MODEL_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}
DEFAULT_MODELS_DIR = ROOT / "models" / "mediapipe"
MASK_L = FEATURE_DIM - 2
MASK_R = FEATURE_DIM - 1


def ensure_model(key, models_dir):
    """모델 .task 파일을 캐시에서 반환, 없으면 다운로드 (원자적 저장)."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / Path(MODELS[key]).name
    if not path.exists():
        print(f"[다운로드] {MODELS[key]}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODELS[key], tmp)
        os.replace(tmp, path)
        print(f"  → {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return str(path)


def _first(lms):
    """Holistic 결과가 평면/중첩 리스트 어느 쪽이든 수용 (web/app.js first() 미러)."""
    if not lms:
        return None
    if isinstance(lms, (list, tuple)) and not hasattr(lms[0], "x"):
        return lms[0] if lms[0] else None
    return lms


def to_xy(lms):
    """랜드마크 객체 리스트 → [x, y] 쌍 리스트 (app.js toXY 미러, 반올림 없음)."""
    if not lms:
        return None
    return [[p.x, p.y] for p in lms]


# ---------------------------------------------------------------- 추출기
# 공통 인터페이스: detect(img, ts_ms) → (pose, left, right, face) [x,y] 리스트들


class HolisticExtractor:
    mode = "holistic"

    def __init__(self, models_dir):
        self._lm = vision.HolisticLandmarker.create_from_options(
            vision.HolisticLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=ensure_model("holistic", models_dir)),
                running_mode=vision.RunningMode.VIDEO,
            ))

    def detect(self, img, ts_ms):
        r = self._lm.detect_for_video(img, ts_ms)
        return (to_xy(_first(r.pose_landmarks)),
                to_xy(_first(r.left_hand_landmarks)),
                to_xy(_first(r.right_hand_landmarks)),
                to_xy(_first(r.face_landmarks)))

    def close(self):
        self._lm.close()


class ComboExtractor:
    mode = "combo"

    def __init__(self, models_dir):
        base = lambda key: BaseOptions(model_asset_path=ensure_model(key, models_dir))
        video = vision.RunningMode.VIDEO
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(base_options=base("pose"),
                                         running_mode=video, num_poses=1))
        self._hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(base_options=base("hand"),
                                         running_mode=video, num_hands=2))
        self._face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(base_options=base("face"),
                                         running_mode=video, num_faces=1))

    def detect(self, img, ts_ms):
        p = self._pose.detect_for_video(img, ts_ms)
        h = self._hands.detect_for_video(img, ts_ms)
        f = self._face.detect_for_video(img, ts_ms)
        left = right = None
        for i, lm in enumerate(h.hand_landmarks):
            # web/app.js 폴백과 동일 규약: handedness 라벨을 그대로 슬롯에 배치
            if h.handedness[i][0].category_name == "Left":
                left = lm
            else:
                right = lm
        return (to_xy(p.pose_landmarks[0] if p.pose_landmarks else None),
                to_xy(left), to_xy(right),
                to_xy(f.face_landmarks[0] if f.face_landmarks else None))

    def close(self):
        self._pose.close()
        self._hands.close()
        self._face.close()


def _first_frame_image(video_path):
    cap = cv2.VideoCapture(str(video_path))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"[오류] 프로브 영상 디코딩 실패: {video_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def resolve_extractor_mode(requested, models_dir, probe_video):
    """auto 면 Holistic 생성 + 실제 detect 프로브 후 모드 확정 (실행 단위 고정)."""
    if requested in ("auto", "holistic"):
        try:
            ex = HolisticExtractor(models_dir)
            ex.detect(_first_frame_image(probe_video), 0)
            ex.close()
            return "holistic"
        except Exception as e:  # noqa: BLE001 — 어떤 초기화/추론 예외든 폴백 대상
            if requested == "holistic":
                raise
            print(f"[경고] Holistic 사용 불가 → combo로 전환: {e}")
    return "combo"


def create_extractor(mode, models_dir):
    """영상마다 새로 생성 — VIDEO 모드 타임스탬프는 인스턴스 단위 단조 증가
    필수(영상 간 리셋 불가)이고, 트래커 상태를 격리해 결과가 처리 순서와
    무관해진다(재현성·resume 안전)."""
    return HolisticExtractor(models_dir) if mode == "holistic" else ComboExtractor(models_dir)


# ---------------------------------------------------------------- 영상 처리


def process_video(video_path, extractor):
    """영상 전체 프레임(30fps 기준)을 추출·정규화해 (T, 250) float32 반환."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (1.0 <= fps <= 120.0):
        fps = 30.0  # 메타데이터 손상 대비 — CLAUDE.md §5 기준값
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows = []
    n_frames = n_invalid = left_miss = right_miss = 0
    prev_ts = -1
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(round(n_frames * 1000.0 / fps))
        if ts <= prev_ts:  # detect_for_video 는 단조 증가 타임스탬프 필수
            ts = prev_ts + 1
        prev_ts = ts

        v = extract_features(*extractor.detect(img, ts))
        if v is None:
            n_invalid += 1  # 포즈 미검출 프레임은 드롭 (계수만 남김)
        else:
            left_miss += v[MASK_L] == 0.0
            right_miss += v[MASK_R] == 0.0
            rows.append(v)
        n_frames += 1
    cap.release()

    n_valid = len(rows)
    stats = {
        "n_frames": n_frames,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        # 미검출률은 유효 프레임 대비 비율
        "left_miss_ratio": round(left_miss / n_valid, 4) if n_valid else None,
        "right_miss_ratio": round(right_miss / n_valid, 4) if n_valid else None,
        "fps": round(fps, 3),
        "width": width,
        "height": height,
    }
    return np.asarray(rows, dtype=np.float32), stats


# ---------------------------------------------------------------- manifest


def load_manifest(out_dir):
    path = out_dir / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_manifest(out_dir, manifest):
    path = out_dir / "manifest.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)


def save_npy(path, arr):
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


# ---------------------------------------------------------------- 진단


def compare_extractors(video_path, models_dir, n_probe_frames=30):
    """holistic vs combo 실측 대조 (web/app.js 가 M3로 이월한 항목).

    같은 프레임을 두 추출기에 넣어 포즈/왼손/오른손/얼굴 그룹별 평균 L2와
    손 교차 슬롯(왼↔오) L2를 비교한다. 동일 슬롯 L2 << 교차 슬롯 L2 면
    combo 의 handedness 라벨이 holistic 의 좌/우 슬롯 규약과 일치한다는
    뜻이다. 보고만 하고 아무것도 저장하지 않는다.
    """
    results = {}
    for mode in ("holistic", "combo"):
        ex = create_extractor(mode, models_dir)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        for idx in range(n_probe_frames):
            ok, bgr = cap.read()
            if not ok:
                break
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            frames.append(ex.detect(img, int(round(idx * 1000.0 / fps))))
        cap.release()
        ex.close()
        results[mode] = frames

    def mean_l2(pairs):
        """양쪽 다 검출된 프레임의 포인트 평균 L2. (거리, 프레임 수) 반환."""
        dists = [
            float(np.linalg.norm(np.asarray(a) - np.asarray(b), axis=1).mean())
            for a, b in pairs
            if a is not None and b is not None and len(a) == len(b)
        ]
        return (sum(dists) / len(dists), len(dists)) if dists else (None, 0)

    h, c = results["holistic"], results["combo"]
    groups = [("포즈", 0, 0), ("왼손", 1, 1), ("오른손", 2, 2), ("얼굴", 3, 3)]
    print(f"[대조] {Path(video_path).stem} 앞 {len(h)}프레임 — holistic ↔ combo")
    print(f"  {'그룹':<12s}{'평균 L2':>12s}{'비교 프레임':>10s}")
    rows = {}
    for name, hi, ci in groups:
        d, n = mean_l2([(hf[hi], cf[ci]) for hf, cf in zip(h, c)])
        rows[name] = d
        print(f"  {name:<12s}{d if d is None else f'{d:12.5f}'}{n:>10d}")
    cross_l, n_xl = mean_l2([(hf[1], cf[2]) for hf, cf in zip(h, c)])
    cross_r, n_xr = mean_l2([(hf[2], cf[1]) for hf, cf in zip(h, c)])
    print(f"  {'왼↔오(교차)':<11s}{cross_l if cross_l is None else f'{cross_l:12.5f}'}{n_xl:>10d}")
    print(f"  {'오↔왼(교차)':<11s}{cross_r if cross_r is None else f'{cross_r:12.5f}'}{n_xr:>10d}")

    same = [rows["왼손"], rows["오른손"]]
    cross = [cross_l, cross_r]
    if all(v is not None for v in same + cross):
        ok = same[0] < cross[0] and same[1] < cross[1]
        print(f"  → 판정: {'handedness 규약 일치 (라벨 그대로 사용 OK)' if ok else '뒤집힘 의심 — 규약 재검토 필요!'}")
    else:
        print("  → 판정 불가: 손이 양쪽 모두 검출된 프레임 부족")


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description="정면 WORD 영상 MediaPipe 재추출 → npy")
    parser.add_argument("--data-root", help="데이터셋 루트 (기본: config.yaml)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ids", help="쉼표 구분 WORD ID (예: WORD1501,WORD1502)")
    parser.add_argument("--limit", type=int, help="앞에서부터 N개 WORD ID만 처리")
    parser.add_argument("--extractor", choices=("auto", "holistic", "combo"),
                        default="auto")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--force", action="store_true", help="기존 npy 재추출")
    parser.add_argument("--compare-extractors", action="store_true",
                        help="holistic vs combo 손 슬롯 대조 진단만 실행")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    entries = scan_frontal_videos(data_root)
    if args.ids:
        wanted = set(args.ids.split(","))
        entries = [e for e in entries if e.label in wanted]
    if args.limit:
        labels = sorted({e.label for e in entries})[:args.limit]
        entries = [e for e in entries if e.label in set(labels)]
    if not entries:
        sys.exit("[오류] 처리할 영상이 없습니다 — --ids/--limit/DATA_ROOT 확인.")

    if args.compare_extractors:
        compare_extractors(entries[0].path, args.models_dir)
        return

    mode = resolve_extractor_mode(args.extractor, args.models_dir, entries[0].path)
    print(f"[추출기] {mode} (mediapipe {mp.__version__}, CPU)")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.out)

    # 혼합 추출기 데이터셋 방지: 기존 항목과 모드가 다르면 중단
    existing_modes = {m["extractor_mode"] for m in manifest.values()
                      if m.get("status") == "ok"}
    if existing_modes and existing_modes != {mode} and not args.force:
        sys.exit(f"[오류] 기존 manifest 추출기({existing_modes})와 현재 모드({mode})가 "
                 "다릅니다 — 혼합 방지. 전체 재추출은 --force와 함께 실행하세요.")

    n_done = n_skip = n_fail = 0
    for e in entries:
        npy_path = args.out / f"{e.stem}.npy"
        if (not args.force and npy_path.exists()
                and manifest.get(e.stem, {}).get("status") == "ok"):
            n_skip += 1
            print(f"[스킵] {e.stem} (완료됨 — 재추출은 --force)")
            continue

        extractor = create_extractor(mode, args.models_dir)
        try:
            arr, stats = process_video(e.path, extractor)
        finally:
            extractor.close()

        entry = {
            "label": e.label,
            **stats,
            "extractor_mode": mode,
            "mediapipe_version": mp.__version__,
            "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if len(arr) == 0:
            entry["status"] = "empty"
            n_fail += 1
            print(f"[실패] {e.stem}: 유효 프레임 0")
        else:
            save_npy(npy_path, arr)
            entry["status"] = "ok"
            n_done += 1
            print(f"{e.stem}  {e.label}  shape={arr.shape}  "
                  f"무효 {stats['n_invalid']}  "
                  f"왼손누락 {stats['left_miss_ratio'] * 100:.1f}%  "
                  f"오른손누락 {stats['right_miss_ratio'] * 100:.1f}%")
        manifest[e.stem] = entry
        save_manifest(args.out, manifest)

    print(f"── 완료 {n_done} / 스킵 {n_skip} / 실패 {n_fail} → "
          f"{args.out / 'manifest.json'} 갱신 ──")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
