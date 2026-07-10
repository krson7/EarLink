"""M3: DATA_ROOT 스캔 → 정면(_F) WORD 영상 목록 + label_map.json 생성.

이 샘플(New_sample)에는 형태소/한국어 라벨 CSV가 없다 → 라벨은 WORD ID
문자열("WORD1501" 등)이며 한국어 매핑은 추후 부착한다. 원래 설계(CLAUDE.md
§4)의 역할 중 윈도우화(60프레임/stride 15)는 M4 학습 시점으로 이관했다 —
영상 단위 (T, 250) npy가 정본 아티팩트다.

주의: 라벨링데이터/(제공 OpenPose JSON)는 절대 읽지 않는다 — 학습-추론
추출기 일치 원칙에 따라 MediaPipe로 재추출한다(extract_keypoints.py).
스캔을 원천데이터/ 하위로 제한해 구조적으로도 접근하지 않는다.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple

import yaml

ROOT = Path(__file__).resolve().parent.parent

# 정면(F) WORD 영상만. 파일명 약어는 WORD (Cowork 분석으로 확인, WRD 아님).
FRONTAL_RE = re.compile(r"^NIA_SL_(WORD\d{4})_REAL\d{2}_F\.mp4$")
SOURCE_SUBDIR = "원천데이터"  # 이 하위만 스캔 (라벨링데이터/ 접근 차단)
DEFAULT_OUT = ROOT / "data" / "processed"
EXPECTED_LABELS = 20  # New_sample 기준


class VideoEntry(NamedTuple):
    path: Path
    label: str   # 예: "WORD1501"
    stem: str    # 예: "NIA_SL_WORD1501_REAL01_F"


def resolve_data_root(cli_value=None):
    """우선순위: --data-root > $DATA_ROOT > config.yaml data.data_root."""
    candidates = (
        cli_value,
        os.environ.get("DATA_ROOT"),
        yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        .get("data", {}).get("data_root"),
    )
    for value in candidates:
        if value:
            root = Path(value).expanduser()
            if not root.is_dir():
                sys.exit(f"[오류] DATA_ROOT가 디렉토리가 아닙니다: {root}")
            return root
    sys.exit("[오류] DATA_ROOT 미설정 — config.yaml data.data_root 또는 "
             "$DATA_ROOT, --data-root로 지정하세요.")


def scan_frontal_videos(data_root):
    """원천데이터/ 하위의 정면 WORD 영상을 (라벨, 파일명) 정렬로 반환."""
    source = data_root / SOURCE_SUBDIR
    if not source.is_dir():
        sys.exit(f"[오류] {source} 가 없습니다 — DATA_ROOT 확인 필요.")
    entries = [
        VideoEntry(p, m.group(1), p.stem)
        for p in source.rglob("*.mp4")
        if (m := FRONTAL_RE.fullmatch(p.name))
    ]
    return sorted(entries, key=lambda e: (e.label, e.stem))


def main():
    parser = argparse.ArgumentParser(description="정면 WORD 영상 스캔 + label_map 생성")
    parser.add_argument("--data-root", help="데이터셋 루트 (기본: config.yaml)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="출력 디렉토리 (기본: data/processed)")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    entries = scan_frontal_videos(data_root)
    labels = sorted({e.label for e in entries})
    label_map = {label: i for i, label in enumerate(labels)}

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "label_map.json"
    out_path.write_text(json.dumps(label_map, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"[스캔] {data_root / SOURCE_SUBDIR}")
    print(f"[결과] 정면 영상 {len(entries)}개, 라벨 {len(labels)}개 → {out_path}")
    for label in labels:
        n = sum(1 for e in entries if e.label == label)
        print(f"  {label}: 영상 {n}개")
    if len(labels) != EXPECTED_LABELS:
        print(f"[경고] 라벨 수가 기대값({EXPECTED_LABELS})과 다릅니다: {len(labels)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
