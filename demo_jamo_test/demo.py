"""Rahites 지문자 모델 체험 데모 (프로젝트 밖 자립형).

  python demo.py --webcam                       # 웹캠 실시간
  python demo.py --video sample.mp4 --truth 가락로   # AI Hub 영상 재생하며 인식

MediaPipe HandLandmarker(Tasks API)로 손 21점을 뽑아 Rahites 의 전처리
(Vector_Normalization: 20 뼈 단위벡터 + 15 각도 = 55차원, 시퀀스 10)로 변환하고,
그들의 학습 tflite 가중치로 자모를 추론한다. 한글은 PIL(맑은고딕)로 오버레이한다.

키: c=스트림 리셋, q=종료, space=일시정지(영상 모드).
"""
import argparse
import platform
import sys
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from hangul_compose import HangulComposer

HERE = Path(__file__).resolve().parent
TFLITE = HERE / "models" / "multi_hand_gesture_classifier.tflite"
HAND_TASK = HERE / "hand_landmarker.task"
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task")

SEQ_LEN = 10
ACTIONS = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
           'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']


# ------------------------------------------------------- 전처리 (Rahites utils.py 그대로)

def vector_normalization(joint):
    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :2]
    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :2]
    v = v2 - v1
    v = v / (np.linalg.norm(v, axis=1)[:, np.newaxis] + 1e-9)
    angle = np.arccos(np.einsum('nt,nt->n',
                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
    angle = np.degrees(angle)
    return v, np.array([angle], dtype=np.float32)


def frame_feature(joint):
    v, angle = vector_normalization(joint.astype(np.float64))
    d = np.concatenate([v.flatten(), angle.flatten()]).astype(np.float32)
    return None if np.isnan(d).any() else d


# ------------------------------------------------------- 모델·랜드마커

def ensure_hand_task():
    if not HAND_TASK.exists():
        print(f"[다운로드] hand_landmarker.task → {HAND_TASK}")
        tmp = HAND_TASK.with_suffix(".part")
        urllib.request.urlretrieve(HAND_URL, tmp)
        tmp.replace(HAND_TASK)


def load_tflite():
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(TFLITE))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

    def predict(seq):  # seq (10,55) → (idx, conf)
        interp.set_tensor(inp["index"], seq[np.newaxis].astype(inp["dtype"]))
        interp.invoke()
        y = interp.get_tensor(out["index"])[0]
        i = int(np.argmax(y))
        return i, float(y[i])
    return predict


def best_hand(result):
    """검출된 손 중 최고 confidence 손의 21점 (21,2) 정규화 좌표. 없으면 None."""
    if not result.hand_landmarks:
        return None
    scores = [h[0].score for h in result.handedness]
    lm = result.hand_landmarks[int(np.argmax(scores))]
    return np.array([[p.x, p.y] for p in lm], dtype=np.float64)


# ------------------------------------------------------- 오버레이 (PIL 한글)

def load_font(size):
    if platform.system() == "Windows":
        cands = [r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\malgunbd.ttf"]
    elif platform.system() == "Darwin":
        cands = ["/System/Library/Fonts/AppleGothic.ttf",
                 "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
                 "/Library/Fonts/AppleGothic.ttf"]
    else:
        cands = ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]
    for c in cands:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


class Fonts:
    def __init__(self):
        self.big = load_font(52)
        self.mid = load_font(34)
        self.small = load_font(28)


def annotate(frame_bgr, state, fonts, truth=None):
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)

    def text(pos, s, font, fill=(255, 255, 0)):
        d.text(pos, s, font=font, fill=fill,
               stroke_width=3, stroke_fill=(0, 0, 0))

    cur = state["cur"]
    cur_str = f"{cur[0]}  {cur[1]:.2f}" if cur else "…"
    text((16, 12), cur_str, fonts.big, (80, 220, 120))
    text((16, 84), "확정: " + "".join(state["stream"]), fonts.mid)
    text((16, 130), "조합: " + state["composed"], fonts.mid, (120, 200, 255))
    if truth:
        text((16, img.height - 48), f"정답: {truth}", fonts.small, (255, 180, 180))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ------------------------------------------------------- 확정 상태기계

class Debouncer:
    def __init__(self, confirm_frames, conf_thr):
        self.n = confirm_frames
        self.thr = conf_thr
        self.active = None      # 현재 안정 자모
        self.count = 0
        self.fired = False
        self.last_confirmed = None
        self.seen_different = False

    def update(self, jamo, conf):
        """프레임별 예측 → 확정된 자모 반환(없으면 None)."""
        if jamo is None or conf < self.thr:
            self.active = None
            self.count = 0
            self.fired = False
            return None
        if jamo != self.active:
            self.active = jamo
            self.count = 1
            self.fired = False
            if jamo != self.last_confirmed:
                self.seen_different = True
            return None
        self.count += 1
        if self.count >= self.n and not self.fired:
            if self.active != self.last_confirmed or self.seen_different:
                self.fired = True
                self.last_confirmed = self.active
                self.seen_different = False
                return self.active
        return None


# ------------------------------------------------------- 메인 루프

def run(args):
    ensure_hand_task()
    predict = load_tflite()
    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_TASK)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2))
    fonts = Fonts()

    if args.webcam:
        cap = cv2.VideoCapture(0)
        flip_default = True
    else:
        cap = cv2.VideoCapture(args.video)
        flip_default = False
    if not cap.isOpened():
        sys.exit(f"[오류] 입력 열기 실패: {'webcam' if args.webcam else args.video}")
    flip = args.flip if args.flip is not None else flip_default

    buf = deque(maxlen=SEQ_LEN)
    deb = Debouncer(args.confirm_frames, args.conf)
    composer = HangulComposer()
    state = {"cur": None, "stream": [], "composed": ""}
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    paused = False
    saved = 0
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            if flip:
                frame = cv2.flip(frame, 1)
            ts = int(idx * 1000 / 30)
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            joint = best_hand(res)
            cur = None
            if joint is not None:
                feat = frame_feature(joint)
                if feat is not None:
                    buf.append(feat)
            else:
                buf.clear()
            if len(buf) == SEQ_LEN:
                i, conf = predict(np.array(buf, dtype=np.float32))
                cur = (ACTIONS[i], conf)
                confirmed = deb.update(ACTIONS[i], conf)
                if confirmed:
                    state["stream"].append(confirmed)
                    composer.feed(confirmed)
                    state["composed"] = composer.text
            else:
                deb.update(None, 0.0)
            state["cur"] = cur
            idx += 1

        out = annotate(frame, state, fonts, truth=args.truth)

        # 주석 프레임 저장 (확정 순간 + 주기적)
        if save_dir and not paused and (deb.fired or idx % 20 == 0) and saved < 400:
            cv2.imwrite(str(save_dir / f"f{idx:05d}.png"), out)
            saved += 1

        if args.no_window:
            continue  # 헤드리스 — 다음 프레임 (영상 끝나면 위에서 break)

        # 화면 맞춤 리사이즈 후 표시
        h, w = out.shape[:2]
        if h > args.max_height:
            s = args.max_height / h
            out = cv2.resize(out, (int(w * s), int(h * s)))
        cv2.imshow("Rahites jamo demo", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            state["stream"].clear(); composer.reset(); state["composed"] = ""
            deb.last_confirmed = None; deb.seen_different = False
        if key == ord(' ') and not args.webcam:
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    if save_dir:
        print(f"[저장] 주석 프레임 {saved}개 → {save_dir}")
    print(f"확정 스트림: {''.join(state['stream'])}  |  조합: {state['composed']}")


def main():
    ap = argparse.ArgumentParser(description="Rahites 지문자 체험 데모")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--webcam", action="store_true")
    g.add_argument("--video", type=str)
    ap.add_argument("--truth", type=str, default=None, help="영상 정답 단어(하단 표시)")
    ap.add_argument("--confirm-frames", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.8)
    ap.add_argument("--flip", type=lambda s: s.lower() in ("1", "true", "yes"), default=None,
                    help="좌우반전 강제(기본: 웹캠 on, 영상 off)")
    ap.add_argument("--max-height", type=int, default=900)
    ap.add_argument("--no-window", action="store_true", help="헤드리스(imshow 없이 저장만)")
    ap.add_argument("--save-dir", type=str, default=None, help="주석 프레임 저장 폴더")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
