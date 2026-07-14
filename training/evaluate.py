"""M4: 학습 결과 평가 — train/val 정확도, 정확도 곡선, 혼동행렬 이미지.

train.py 가 저장한 dataset_v{n}.npz 를 그대로 평가한다 (시드 재생성 방식은
증강 코드가 수정되면 조용히 어긋나므로 npz 고정). val 은 같은 원본의
증강본 홀드아웃 — 누수 있는 참고용 지표임을 리포트에 명시한다.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"

# macOS 한글 폰트 후보 (미설정 시 축 라벨이 tofu(□)로 렌더된다)
KOREAN_FONTS = ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic")


def setup_korean_font(plt):
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in KOREAN_FONTS:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    else:
        print(f"[경고] 한글 폰트({KOREAN_FONTS}) 미발견 — 라벨이 깨질 수 있음")
    plt.rcParams["axes.unicode_minus"] = False  # 마이너스 부호 tofu 방지


def load_artifacts(version):
    import keras

    model_path = MODELS_DIR / f"ksl_word_v{version}.keras"
    for p in (model_path,
              MODELS_DIR / f"ksl_word_v{version}.labels.json",
              MODELS_DIR / f"ksl_word_v{version}.history.json",
              PROCESSED / f"dataset_v{version}.npz"):
        if not p.exists():
            sys.exit(f"[오류] {p} 없음 — training.train 을 먼저 실행하세요.")
    model = keras.models.load_model(model_path)
    labels = json.loads((MODELS_DIR / f"ksl_word_v{version}.labels.json")
                        .read_text(encoding="utf-8"))
    history = json.loads((MODELS_DIR / f"ksl_word_v{version}.history.json")
                         .read_text(encoding="utf-8"))
    data = np.load(PROCESSED / f"dataset_v{version}.npz")
    return model, labels, history, data, model_path


def confusion_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion(plt, cm, words, out_png):
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(words)), labels=words, rotation=90, fontsize=8)
    ax.set_yticks(range(len(words)), labels=words, fontsize=8)
    ax.set_xlabel("예측")
    ax.set_ylabel("정답")
    ax.set_title("혼동행렬 (val — 누수 있는 참고용)")
    for i in range(len(words)):        # 셀 값 표기는 비대각 오류만 (가독성)
        for j in range(len(words)):
            if cm[i, j] and i != j:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="red", fontsize=7)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_curves(plt, history, out_png):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["accuracy"]) + 1)
    ax1.plot(epochs, history["accuracy"], label="train")
    ax1.plot(epochs, history["val_accuracy"], label="val (참고용)")
    ax1.set_title("정확도")
    ax1.set_xlabel("epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["loss"], label="train")
    ax2.plot(epochs, history["val_loss"], label="val (참고용)")
    ax2.set_title("손실")
    ax2.set_xlabel("epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def top_confusions(cm, words, k=10):
    """비대각 혼동 셀을 횟수 내림차순으로 (정답→예측, 횟수)."""
    pairs = [(words[i], words[j], int(cm[i, j]))
             for i in range(len(words)) for j in range(len(words))
             if i != j and cm[i, j] > 0]
    return sorted(pairs, key=lambda x: -x[2])[:k]


def main():
    parser = argparse.ArgumentParser(description="M4 학습 결과 평가·시각화")
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")  # 헤드리스 렌더링
    import matplotlib.pyplot as plt
    setup_korean_font(plt)

    model, labels, history, data, model_path = load_artifacts(args.version)
    words = list(labels)  # CSV(선정순번) 순서 == 인덱스 순서

    _, train_acc = model.evaluate(data["x_train"], data["y_train"], verbose=0)
    _, val_acc = model.evaluate(data["x_val"], data["y_val"], verbose=0)
    y_pred = model.predict(data["x_val"], verbose=0).argmax(axis=1)
    cm = confusion_matrix(data["y_val"], y_pred, len(words))

    curves_png = MODELS_DIR / f"ksl_word_v{args.version}.curves.png"
    confusion_png = MODELS_DIR / f"ksl_word_v{args.version}.confusion.png"
    plot_curves(plt, history, curves_png)
    plot_confusion(plt, cm, words, confusion_png)

    size_mb = model_path.stat().st_size / 1e6
    confusions = top_confusions(cm, words)
    print("=== M4 학습 결과 (B안: REAL01 단일 화자 — 파이프라인 검증용, "
          "정확도는 목표 아님) ===")
    print(f"모델      : {model_path} ({size_mb:.1f} MB)")
    print(f"라벨맵    : {MODELS_DIR / f'ksl_word_v{args.version}.labels.json'} "
          f"({len(words)}단어)")
    print(f"train acc : {train_acc:.3f}   val acc : {val_acc:.3f}")
    print("※ val 은 동일 원본의 증강본(원본 1개/클래스 → 깨끗한 분할 불가) — "
          "누수 있는 참고용 수치")
    print(f"곡선      : {curves_png}")
    print(f"혼동행렬  : {confusion_png}")
    if confusions:
        pairs = "  ".join(f"{a}→{b}({n})" for a, b, n in confusions)
        print(f"혼동 상위 쌍: {pairs}")
    else:
        print("혼동 상위 쌍: 없음 (val 전량 정답)")


if __name__ == "__main__":
    main()
