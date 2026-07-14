"""M4: Conv1D-LSTM 학습 — B안(REAL01 단일 화자) 파이프라인 완주 검증.

목적은 정확도가 아니라 "학습 → 저장 → 재로드 추론"의 전 과정이 도는 것의
검증이다. 클래스당 원본 영상이 1개뿐이라 과적합은 예상된 정상 결과이며,
val 지표는 같은 원본의 증강본에서 홀드아웃한 **누수 있는 참고용 수치**다
(깨끗한 분할은 다화자 데이터 확보 후에나 가능). 이 모델은 M5 웹앱 연결
테스트용이다.

윈도우는 45프레임(1.5초) — CLAUDE.md §5의 60에서 축소(사용자 결정,
2026-07-10): 트리밍된 최종 30단어 클립 중 22개가 45프레임 미만이라 60은
대부분 패딩이 된다. config.yaml model.window_frames 와 일치해야 한다.

입력은 data/processed/의 npy + training/earlink_final_30words.csv 뿐이다.
training.build_dataset 는 import 하지 않는다 — load_candidates 가
DATA_ROOT(원본 영상)를 요구하므로, 학습이 영상 데이터에 의존하게 된다.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from shared.features import (
    FACE_SUBSET_INDICES,
    FEATURE_DIM,
    N_HAND_POINTS,
    POSE_SUBSET_INDICES,
)
from shared.windowing import resample, window

ROOT = Path(__file__).resolve().parent.parent
FINAL_CSV = ROOT / "training" / "earlink_final_30words.csv"
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
VERSION = 1          # 기본값 — --version 으로 재정의
DENSITY_AUG_P = 0.3  # 15fps 수신→30fps 복원(보간)을 흉내내는 밀도 왕복 증강 확률

SEED = 42
WINDOW = 45              # 1.5초 @30fps (§5의 60에서 축소 — 모듈 docstring 참조)
N_AUG = 20               # 원본당 증강 배수
SIGMA_NOISE = 0.01       # 좌표 노이즈 σ (어깨거리 단위 — 정규화 후 어깨폭 1.0)
TIME_SCALE_RANGE = (0.8, 1.2)
FLIP_P = 0.5
VAL_PER_CLASS = 4        # 클래스당 증강 마지막 4개를 val 로 (참고용 — 누수 있음)
BATCH, EPOCHS, LR = 32, 60, 1e-3

# 특징 레이아웃 오프셋 (shared/features.py 계약에서 유도)
N_COORD = FEATURE_DIM - 2                    # 248
LH_OFF = len(POSE_SUBSET_INDICES) * 2        # 24
RH_OFF = LH_OFF + N_HAND_POINTS * 2          # 66
FACE_OFF = RH_OFF + N_HAND_POINTS * 2        # 108
MASK_L, MASK_R = N_COORD, N_COORD + 1        # 248, 249

# ---------------------------------------------------------------- 좌우 반전
# 미러 쌍은 MediaPipe canonical_face_model.obj 의 x-대칭으로 기하 검증 완료
# (2026-07-10, 전 쌍 x_a == -x_b / y·z 동일, 위반 0). 입술 미드라인
# {0,13,14,17}은 자기 자신이 미러다.

POSE_MIRROR = {2: 5, 5: 2, 7: 8, 8: 7, 9: 10, 10: 9,
               11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15}
_FACE_MIRROR_PAIRS = [
    # 눈썹
    (46, 276), (52, 282), (53, 283), (55, 285), (63, 293),
    (65, 295), (66, 296), (70, 300), (105, 334), (107, 336),
    # 눈
    (33, 263), (133, 362), (145, 374), (158, 385), (159, 386),
    # 입술
    (37, 267), (39, 269), (40, 270), (61, 291), (78, 308), (80, 310),
    (81, 311), (82, 312), (84, 314), (87, 317), (88, 318), (91, 321),
    (95, 324), (146, 375), (178, 402), (181, 405), (185, 409), (191, 415),
]
FACE_MIRROR = ({a: b for a, b in _FACE_MIRROR_PAIRS}
               | {b: a for a, b in _FACE_MIRROR_PAIRS}
               | {i: i for i in (0, 13, 14, 17)})


def build_flip_permutation():
    """250차원 좌우 반전용 (perm, sign). 적용: seq[:, perm] * sign.

    포즈는 서브셋 내 미러 쌍 스왑, 손은 왼/오 블록 통째 교환(랜드마크
    순서는 handedness 무관), 얼굴은 미러 쌍 스왑, 마스크는 L↔R.
    좌표 원점이 어깨 중점이므로 x 열 부호 반전만으로 미러가 완성된다.
    """
    perm = np.arange(FEATURE_DIM)
    pose_pos = {lm: i for i, lm in enumerate(POSE_SUBSET_INDICES)}
    for i, lm in enumerate(POSE_SUBSET_INDICES):
        j = pose_pos[POSE_MIRROR[lm]]
        perm[2 * i] = 2 * j
        perm[2 * i + 1] = 2 * j + 1
    for k in range(N_HAND_POINTS * 2):
        perm[LH_OFF + k] = RH_OFF + k
        perm[RH_OFF + k] = LH_OFF + k
    face_pos = {lm: i for i, lm in enumerate(FACE_SUBSET_INDICES)}
    for i, lm in enumerate(FACE_SUBSET_INDICES):
        j = face_pos[FACE_MIRROR[lm]]
        perm[FACE_OFF + 2 * i] = FACE_OFF + 2 * j
        perm[FACE_OFF + 2 * i + 1] = FACE_OFF + 2 * j + 1
    perm[MASK_L], perm[MASK_R] = MASK_R, MASK_L
    sign = np.ones(FEATURE_DIM, dtype=np.float32)
    sign[0:N_COORD:2] = -1.0  # x 열만 부호 반전 (마스크 2차원 제외)
    return perm, sign


_PERM, _SIGN = build_flip_permutation()
# 임포트 시 자기검증: perm 은 치환이자 대합(두 번 적용 = 항등)이어야 한다
assert np.array_equal(np.sort(_PERM), np.arange(FEATURE_DIM))
assert np.array_equal(_PERM[_PERM], np.arange(FEATURE_DIM))
assert set(FACE_MIRROR) == set(FACE_SUBSET_INDICES)


def flip(seq):
    """좌우 반전 (라벨 유지). 두 번 적용하면 원본과 동일해야 한다."""
    return (seq[:, _PERM] * _SIGN).astype(np.float32)


# ---------------------------------------------------------------- 증강


def add_noise(seq, rng):
    """좌표부에만 가우시안 노이즈. 결측 블록은 재-제로화해
    '전부 0 = 결측' 계약(M5 추론과의 분포 일치)을 보존한다."""
    out = seq.copy()
    out[:, :N_COORD] += rng.normal(
        0.0, SIGMA_NOISE, (len(seq), N_COORD)).astype(np.float32)
    out[:, LH_OFF:LH_OFF + 42] *= seq[:, MASK_L:MASK_L + 1]
    out[:, RH_OFF:RH_OFF + 42] *= seq[:, MASK_R:MASK_R + 1]
    face_present = np.abs(seq[:, FACE_OFF:FACE_OFF + 140]).sum(axis=1) > 0
    out[:, FACE_OFF:FACE_OFF + 140] *= face_present[:, None].astype(np.float32)
    return out


# resample/window 는 shared/windowing.py 로 이동 (M5 — 추론과 공용, 수식 불변)


def augment_once(seq, lead, sign, rng):
    """컨텍스트 지터 → 시간스케일 → 밀도 왕복(p=0.3) → 좌우반전(p=0.5)
    → 노이즈 → 윈도우. 패딩이 마지막이라 노이즈가 패딩을 오염시키지 않는다.

    컨텍스트 지터(v2): 수어 본체 양쪽의 전환 동작(손 올리기/내리기)을 랜덤
    길이로 포함시킨다 — 실시간 발화 탐지 세그먼트에는 전환부가 섞이므로,
    이를 분포 안에 넣지 않으면 과적합 모델이 전환 조각에 과신 오답을 낸다
    (M5 파리티 버그 잔여 원인). 밀도 왕복은 ~15fps 클라이언트 수신 후
    30fps 등가 복원(선형 보간)의 스무딩을 흉내낸다."""
    tail = len(seq) - lead - sign
    r1 = int(rng.integers(0, lead + 1)) if lead > 0 else 0
    r2 = int(rng.integers(0, tail + 1)) if tail > 0 else 0
    out = seq[lead - r1: lead + sign + r2]

    scale = rng.uniform(*TIME_SCALE_RANGE)
    out = resample(out, int(round(len(out) * scale)))
    if rng.random() < DENSITY_AUG_P and len(out) >= 4:
        out = resample(out[::2], len(out))  # 반 밀도 샘플 → 보간 복원
    if rng.random() < FLIP_P:
        out = flip(out)
    out = add_noise(out, rng)
    return window(out, WINDOW, rng)


# ---------------------------------------------------------------- 데이터셋


def load_final_words():
    """최종 30단어 CSV(utf-8-sig — BOM 미처리 시 첫 키가 '\\ufeff번호') 파싱."""
    with open(FINAL_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30, f"최종 CSV 행 수 이상: {len(rows)}"
    words = [r["korean_word"] for r in rows]
    assert len(set(words)) == 30, "korean_word 중복"
    return rows


def load_sequences(rows, processed=PROCESSED):
    """CSV stem 으로 npy 30개만 선별 (glob 금지 — processed 에는 다른 npy가
    있을 수 있다). manifest 로 status/라벨 일치를 검증한다.

    반환: [(seq, lead, sign), ...] — lead/sign 은 컨텍스트 여유 추출
    (--margin-sec) 시 수어 본체의 위치. 여유 없이 추출된 npy 는 lead=0,
    sign=전체 길이 (기존과 동일 동작)."""
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    seqs = []
    for r in rows:
        stem = Path(r["video_filename"]).stem
        m = manifest.get(stem)
        if not m or m.get("status") != "ok":
            sys.exit(f"[오류] {stem}: manifest에 ok 항목이 없음 — 재추출 필요")
        if m.get("label") != r["korean_word"]:
            sys.exit(f"[오류] {stem}: manifest 라벨({m.get('label')}) ≠ "
                     f"CSV({r['korean_word']})")
        seq = np.load(processed / f"{stem}.npy")
        lead = int(m.get("lead_frames", 0))
        sign = int(m.get("sign_frames", len(seq) - lead))
        seqs.append((seq, lead, sign))
    return seqs


def build_dataset(rows, seqs, rng):
    """클래스당 21샘플(원본 중앙 크롭 1 + 증강 20) = 630.
    분할: [원본 + 증강 1..16] → train(510), [증강 17..20] → val(120).
    ※ val 은 같은 원본의 증강본 — 누수이며 참고용 수치다."""
    x_train, y_train, x_val, y_val = [], [], [], []
    for label_idx, (seq, lead, sign) in enumerate(seqs):
        samples = [window(seq[lead:lead + sign], WINDOW)]  # 원본: 수어 본체만
        samples += [augment_once(seq, lead, sign, rng) for _ in range(N_AUG)]
        train_part = samples[:1 + N_AUG - VAL_PER_CLASS]
        val_part = samples[1 + N_AUG - VAL_PER_CLASS:]
        x_train += train_part
        y_train += [label_idx] * len(train_part)
        x_val += val_part
        y_val += [label_idx] * len(val_part)
    return (np.stack(x_train), np.asarray(y_train, dtype=np.int64),
            np.stack(x_val), np.asarray(y_val, dtype=np.int64))


# ---------------------------------------------------------------- 모델


def configure_device(tf, force_cpu):
    """tensorflow-metal 이 있으면 활용, 없거나 실패하면 CPU. 절대 중단하지 않음."""
    if force_cpu:
        tf.config.set_visible_devices([], "GPU")
        return "CPU (--cpu 지정)"
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return "CPU (GPU 장치 없음 — tensorflow-metal 미설치)"
        with tf.device("/GPU:0"):  # 미니 프로브: 플러그인이 실제로 동작하는지
            tf.linalg.matmul(tf.ones((2, 2)), tf.ones((2, 2))).numpy()
        return f"GPU ({gpus[0].name})"
    except Exception as exc:  # noqa: BLE001 — 플러그인 실패는 종류 불문 CPU 폴백
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            sys.exit("[오류] GPU 초기화 후 폴백 불가 — --cpu 로 재실행하세요.")
        return f"CPU (GPU 프로브 실패: {exc})"


def build_model(keras, n_classes):
    """CLAUDE.md §7 구조 그대로. 입력 (45, 250).
    padding='same' — 스펙은 커널 크기만 규정하며, 짧은 클립의 가장자리
    프레임 손실을 피하는 쪽을 택했다."""
    model = keras.Sequential([
        keras.Input(shape=(WINDOW, FEATURE_DIM)),
        keras.layers.Conv1D(64, 3, padding="same", activation="relu"),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(128, 3, padding="same", activation="relu"),
        keras.layers.Bidirectional(keras.layers.LSTM(64)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(LR),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description="M4 Conv1D-LSTM 학습 (B안 완주 검증)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--cpu", action="store_true",
                        help="GPU 를 완전히 숨기고 CPU 로 실행 (재현성 보장 경로)")
    parser.add_argument("--processed", type=Path, default=PROCESSED,
                        help="npy 디렉토리 (컨텍스트 증강은 data/processed_ctx)")
    parser.add_argument("--version", type=int, default=VERSION,
                        help="모델 버전 번호 (ksl_word_v{n}.keras)")
    args = parser.parse_args()
    version = args.version

    import tensorflow as tf  # 무거운 임포트는 인자 파싱 뒤에
    import keras

    device = configure_device(tf, args.cpu)
    tf.keras.utils.set_random_seed(SEED)
    rng = np.random.default_rng(SEED)
    print(f"[디바이스] {device}  (tensorflow {tf.__version__})")

    rows = load_final_words()
    seqs = load_sequences(rows, args.processed)
    label_map = {r["korean_word"]: i for i, r in enumerate(rows)}  # CSV 순서 고정

    # flip 자기검증 스모크: 실제 데이터에 2회 적용 = 원본
    assert np.allclose(flip(flip(seqs[0][0])), seqs[0][0], atol=1e-6)

    x_train, y_train, x_val, y_val = build_dataset(rows, seqs, rng)
    print(f"[데이터] train {len(x_train)} / val {len(x_val)} "
          f"(클래스 30, 윈도우 {WINDOW}프레임, 증강 {N_AUG}배)")
    print("[주의] val 은 같은 원본의 증강본 홀드아웃 — 누수 있는 참고용 지표"
          " (클래스당 원본 1개라 깨끗한 분할 불가)")

    np.savez_compressed(PROCESSED / f"dataset_v{version}.npz",
                        x_train=x_train, y_train=y_train,
                        x_val=x_val, y_val=y_val)

    model = build_model(keras, n_classes=len(rows))
    model.summary()

    t0 = time.time()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        batch_size=BATCH, epochs=args.epochs, shuffle=True,
                        verbose=2)
    duration = time.time() - t0

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"ksl_word_v{version}.keras"
    model.save(model_path)
    labels_path = MODELS_DIR / f"ksl_word_v{version}.labels.json"
    labels_path.write_text(json.dumps(label_map, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    history_path = MODELS_DIR / f"ksl_word_v{version}.history.json"
    history_path.write_text(json.dumps({
        # float32 는 JSON 직렬화 불가 → float 캐스팅
        **{k: [float(v) for v in vs] for k, vs in history.history.items()},
        "epochs": args.epochs, "seed": SEED, "device": device,
        "n_train": len(x_train), "n_val": len(x_val),
        "window": WINDOW, "duration_sec": round(duration, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 왕복 검증 (B안의 핵심: 학습 → 저장 → 재로드 추론이 도는가) ──
    before = model.predict(x_val[:BATCH], verbose=0).argmax(axis=1)
    reloaded = keras.models.load_model(model_path)
    after = reloaded.predict(x_val[:BATCH], verbose=0).argmax(axis=1)
    assert np.array_equal(before, after), "재로드 후 예측 불일치!"
    inv = {v: k for k, v in label_map.items()}
    print(f"[왕복 검증] 저장 → 재로드 → 추론 OK — val 첫 배치 예측 일치 "
          f"({BATCH}개, 예: '{inv[int(after[0])]}')")

    size_mb = model_path.stat().st_size / 1e6
    print(f"[저장] {model_path} ({size_mb:.1f} MB)")
    print(f"[저장] {labels_path} / {history_path}")
    print(f"[완료] {args.epochs} epoch, {duration:.0f}초, "
          f"train acc {history.history['accuracy'][-1]:.3f}, "
          f"val acc {history.history['val_accuracy'][-1]:.3f} (참고용)")


if __name__ == "__main__":
    main()
