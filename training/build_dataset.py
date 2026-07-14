"""M3: 후보 CSV → 라벨(한국어)·트리밍 구간 + label_map.json 생성.

라벨·트리밍 소스는 training/earlink_30words_candidates.csv (후보 40단어,
Cowork 분석 산출물). 이 CSV는 **utf-8-sig**(BOM 포함)로 읽는다 — AI Hub
NIA 원본 CSV(cp949, CLAUDE.md §10)와 다른 우리 산출물이기 때문.
라벨은 korean_word가 정본이고 순서는 CSV(선정순번)를 그대로 고정한다.

윈도우화(60프레임/stride 15)는 M4 학습 시점으로 이관 — 영상 단위
(T, 250) npy가 정본 아티팩트다.

주의: 라벨링데이터/(제공 라벨·키포인트)는 절대 읽지 않는다 — 학습-추론
추출기 일치 원칙에 따라 MediaPipe로 재추출한다(extract_keypoints.py).
스캔은 경로에 원천데이터/가 포함된 mp4로 제한해 구조적으로도 접근하지 않는다.
"""

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import NamedTuple

import yaml

ROOT = Path(__file__).resolve().parent.parent

# 정면(F) WORD 영상만. 파일명 약어는 WORD (Cowork 분석으로 확인, WRD 아님).
FRONTAL_RE = re.compile(r"^NIA_SL_(WORD\d{4})_REAL\d{2}_F\.mp4$")
SOURCE_DIR_NAME = "원천데이터"  # 경로에 이 디렉토리가 포함된 영상만 스캔
CANDIDATES_CSV = ROOT / "training" / "earlink_30words_candidates.csv"
DEFAULT_OUT = ROOT / "data" / "processed"
WINDOW_FRAMES = 60  # CLAUDE.md §5 — M4 윈도우 설계 점검용 (보고 전용)


class Candidate(NamedTuple):
    word_id: str      # 예: "WORD1637"
    korean: str       # 예: "없다" — 라벨 정본
    filename: str     # 예: "NIA_SL_WORD1637_REAL01_F.mp4"
    start_sec: float  # 트리밍 시작 (대기 자세 제거)
    end_sec: float    # 트리밍 끝
    stem: str         # npy/manifest 키
    path: Path        # DATA_ROOT 하위에서 해석된 절대 경로


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


def _nfc(s):
    """macOS 파일시스템은 한글 경로를 NFD(자모 분해형)로 저장하므로,
    소스 코드의 NFC 문자열과 비교하려면 반드시 NFC로 정규화해야 한다."""
    return unicodedata.normalize("NFC", s)


def scan_frontal_videos(data_root):
    """경로에 원천데이터/가 포함된 정면 WORD 영상을 파일명 정렬로 반환."""
    videos = [
        p for p in data_root.rglob("*.mp4")
        if any(_nfc(part) == SOURCE_DIR_NAME for part in p.parts)
        and FRONTAL_RE.fullmatch(p.name)
    ]
    return sorted(videos, key=lambda p: p.name)


def load_candidates(data_root, csv_path=CANDIDATES_CSV):
    """후보 CSV → 선정순번 순 Candidate 목록. 검증 실패는 모아서 한 번에 보고."""
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"[오류] 후보 CSV가 비어 있습니다: {csv_path}")

    errors = []
    for col in ("word_id", "korean_word", "video_filename"):
        values = [r[col] for r in rows]
        dups = sorted({v for v in values if values.count(v) > 1})
        if dups:
            errors.append(f"{col} 중복: {dups}")
    for r in rows:
        if not (0 <= float(r["start_sec"]) < float(r["end_sec"])):
            errors.append(f"{r['word_id']}: 구간 이상 "
                          f"({r['start_sec']}~{r['end_sec']})")
        if not FRONTAL_RE.fullmatch(r["video_filename"]):
            errors.append(f"{r['word_id']}: 정면 패턴 불일치 {r['video_filename']}")

    name_index = {}
    for p in scan_frontal_videos(data_root):
        if p.name in name_index:
            errors.append(f"동일 파일명 중복 발견: {p} / {name_index[p.name]}")
        name_index[p.name] = p
    missing = [r["video_filename"] for r in rows
               if r["video_filename"] not in name_index]
    if missing:
        errors.append(f"DATA_ROOT에 없는 영상 {len(missing)}개: {missing}")

    if errors:
        sys.exit("[오류] 후보 CSV 검증 실패:\n  - " + "\n  - ".join(errors))

    return [
        Candidate(
            word_id=r["word_id"],
            korean=r["korean_word"],
            filename=r["video_filename"],
            start_sec=float(r["start_sec"]),
            end_sec=float(r["end_sec"]),
            stem=Path(r["video_filename"]).stem,
            path=name_index[r["video_filename"]],
        )
        for r in rows
    ]


def build_label_map(candidates):
    """korean_word → 인덱스. CSV(선정순번) 순서 고정 — 정렬 금지."""
    return {c.korean: i for i, c in enumerate(candidates)}


def main():
    parser = argparse.ArgumentParser(description="후보 CSV 파싱 + label_map 생성")
    parser.add_argument("--data-root", help="데이터셋 루트 (기본: config.yaml)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    n_scanned = len(scan_frontal_videos(data_root))
    print(f"[스캔] {data_root} 하위 {SOURCE_DIR_NAME}/ 정면 영상: {n_scanned}개")

    candidates = load_candidates(data_root)
    label_map = build_label_map(candidates)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "label_map.json"
    out_path.write_text(json.dumps(label_map, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    durations = [c.end_sec - c.start_sec for c in candidates]
    n_short = sum(1 for d in durations if d * 30.0 < WINDOW_FRAMES)
    print(f"[후보] {len(candidates)}단어 → {out_path}")
    print(f"[트리밍] 길이 {min(durations):.2f}~{max(durations):.2f}초")
    if n_short:
        print(f"[경고] 60프레임(2초) 미만 클립 {n_short}/{len(candidates)}개 — "
              "M4 윈도우 설계 재검토 필요 (보고만)")


if __name__ == "__main__":
    main()
