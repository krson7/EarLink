// 이어링크(EarLink) M5 — 카메라 → MediaPipe 키포인트 → 특징 벡터 전송
//                        → 서버 발화 탐지·추론 결과 표시 + 한국어 TTS.
//
// 개인정보 설계 원칙: 서버로는 정규화된 특징 벡터 JSON만 보낸다.
// 영상 프레임(이미지/blob)을 전송하는 코드를 이 파일에 추가하지 않는다.

import { extractFeatures, FEATURE_DIM } from "./features.js";

// http://0.0.0.0:8000 은 브라우저가 보안 컨텍스트로 취급하지 않아 카메라가
// 차단된다 (uvicorn 로그의 0.0.0.0 링크를 그대로 여는 실수 방지) → localhost로 이동.
if (location.hostname === "0.0.0.0") {
  location.replace(location.href.replace("//0.0.0.0", "//localhost"));
}

// ---------------------------------------------------------------- 상수

const MP_VERSION = "0.10.35"; // 문제 시 "0.10.21"로 하향 (이 상수 한 곳만 수정)
const CDN_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const MODELS = {
  holistic: "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task",
  pose: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
  hand: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
  face: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
};
const DETECT_INTERVAL_MS = 66;        // 검출·전송 ~15fps (그리기는 검출 시점에만)
const WS_MAX_BUFFERED = 64 * 1024;    // 송신 버퍼 초과 시 프레임 드롭 (백프레셔)

// ---------------------------------------------------------------- DOM

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const banner = document.getElementById("banner");
const flipBtn = document.getElementById("flipBtn");
const recDot = document.getElementById("recDot");
const results = document.getElementById("results");
const ttsBtn = document.getElementById("ttsBtn");
const hud = {
  dot: document.getElementById("statusDot"),
  status: document.getElementById("statusText"),
  frames: document.getElementById("hudFrames"),
  rtt: document.getElementById("hudRtt"),
  dims: document.getElementById("hudDims"),
  feat: document.getElementById("hudFeat"),
  mode: document.getElementById("hudMode"),
};

function setStatus(text, ok) {
  hud.status.textContent = text;
  hud.dot.className = ok === true ? "ok" : ok === false ? "bad" : "";
}
function showBanner(msg) { banner.textContent = msg; banner.hidden = false; }

// ---------------------------------------------------------------- TTS (한국어)

const tts = {
  enabled: true,
  voice: null,
  pick() {
    // ko 보이스 탐색 — 기기 내장(localService) 우선
    const voices = speechSynthesis.getVoices().filter((v) => v.lang.startsWith("ko"));
    this.voice = voices.find((v) => v.localService) ?? voices[0] ?? null;
  },
  speak(text) {
    if (!this.enabled || !("speechSynthesis" in window)) return;
    speechSynthesis.cancel(); // 백로그 방지 — 직전 단어 끊김은 허용
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "ko-KR";
    if (this.voice) u.voice = this.voice;
    speechSynthesis.speak(u);
  },
};
if ("speechSynthesis" in window) {
  tts.pick(); // Safari 는 voiceschanged 가 안 올 수 있어 즉시 시도도 병행
  speechSynthesis.addEventListener?.("voiceschanged", () => tts.pick());
  // iOS: speak() 는 사용자 제스처 후에만 허용 — 카메라 권한 탭은 DOM 제스처가
  // 아니므로 최초 화면 터치 1회로 언락한다 (빈 발화 → 즉시 취소).
  document.addEventListener("pointerdown", () => {
    speechSynthesis.speak(new SpeechSynthesisUtterance(""));
    speechSynthesis.cancel();
  }, { once: true });
}
ttsBtn.addEventListener("click", () => {
  tts.enabled = !tts.enabled;
  if (!tts.enabled) speechSynthesis?.cancel();
  ttsBtn.textContent = tts.enabled ? "🔊" : "🔇";
});

// ---------------------------------------------------------------- 결과 로그

const MAX_RESULT_ENTRIES = 50;

function appendResult(m) {
  const el = document.createElement("div");
  const pct = Math.round(m.confidence * 100);
  if (m.text === null) {
    // 서버 계약: text=null ⇒ 저신뢰 "인식 안 됨" (TTS 금지)
    el.className = "entry miss";
    el.innerHTML = `인식 안 됨 <small>(추정: ${m.top1} ${pct}%)</small>`;
  } else {
    el.className = "entry";
    el.innerHTML = `${m.text} <small>${pct}%</small>`;
  }
  results.append(el);
  while (results.children.length > MAX_RESULT_ENTRIES) results.firstChild.remove();
  results.scrollTop = results.scrollHeight;
}

// ---------------------------------------------------------------- MediaPipe 로드

setStatus("모델 로딩…");
const { FilesetResolver, DrawingUtils, HolisticLandmarker,
        PoseLandmarker, HandLandmarker, FaceLandmarker } =
  await import(`${CDN_BASE}/vision_bundle.mjs`);
const vision = await FilesetResolver.forVisionTasks(`${CDN_BASE}/wasm`);

// ---------------------------------------------------------------- 랜드마커 팩토리
// 공통 인터페이스: { mode, detect(video, ts) → {pose, leftHand, rightHand, face}, close() }

// 웹 Holistic 결과는 중첩 배열(NormalizedLandmark[][])이지만, 알려진 불안정성에
// 대비해 평면 배열도 수용한다.
function first(lm) {
  if (!lm || lm.length === 0) return null;
  return Array.isArray(lm[0]) ? (lm[0].length ? lm[0] : null) : lm;
}

async function createHolisticLandmarker() {
  let lastErr;
  for (const delegate of ["GPU", "CPU"]) { // GPU 실패 시 CPU 재시도
    try {
      const h = await HolisticLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODELS.holistic, delegate },
        runningMode: "VIDEO",
      });
      return {
        mode: "holistic",
        detect(v, ts) {
          const r = h.detectForVideo(v, ts);
          return {
            pose: first(r.poseLandmarks),
            leftHand: first(r.leftHandLandmarks),
            rightHand: first(r.rightHandLandmarks),
            face: first(r.faceLandmarks),
          };
        },
        close: () => h.close(),
      };
    } catch (e) {
      lastErr = e;
      console.warn(`Holistic(${delegate}) 초기화 실패:`, e);
    }
  }
  throw lastErr;
}

async function createFallbackLandmarker() {
  let lastErr;
  for (const delegate of ["GPU", "CPU"]) {
    try {
      const opts = (path, extra) => ({
        baseOptions: { modelAssetPath: path, delegate },
        runningMode: "VIDEO",
        ...extra,
      });
      const pose = await PoseLandmarker.createFromOptions(vision, opts(MODELS.pose, { numPoses: 1 }));
      const hands = await HandLandmarker.createFromOptions(vision, opts(MODELS.hand, { numHands: 2 }));
      const face = await FaceLandmarker.createFromOptions(vision, opts(MODELS.face, { numFaces: 1 }));
      return {
        mode: "fallback",
        detect(v, ts) {
          const p = pose.detectForVideo(v, ts);
          const h = hands.detectForVideo(v, ts);
          const f = face.detectForVideo(v, ts);
          let leftHand = null, rightHand = null;
          h.landmarks.forEach((lm, i) => {
            // M2 규약: 랜드마커의 handedness 라벨을 그대로 슬롯에 배치한다
            // (좌표는 절대 반전하지 않음 — 미러는 CSS 표시 전용).
            // Holistic 라벨과의 실측 대조는 M3(영상 재추출)에서 수행.
            if (h.handedness?.[i]?.[0]?.categoryName === "Left") leftHand = lm;
            else rightHand = lm;
          });
          return {
            pose: p.landmarks[0] ?? null,
            leftHand,
            rightHand,
            face: f.faceLandmarks[0] ?? null,
          };
        },
        close() { pose.close(); hands.close(); face.close(); },
      };
    } catch (e) {
      lastErr = e;
      console.warn(`폴백 조합(${delegate}) 초기화 실패:`, e);
    }
  }
  throw lastErr;
}

async function createLandmarker() {
  try {
    return await createHolisticLandmarker();
  } catch {
    console.warn("Holistic 사용 불가 → Pose+Hand+Face 폴백으로 전환");
    return createFallbackLandmarker();
  }
}

let landmarker = null;
let detectFailures = 0;
let swapping = false;

// Holistic은 실행 중에도 실패할 수 있다 → 연속 3회 예외 시 폴백으로 교체
function safeDetect(v, ts) {
  try {
    const r = landmarker.detect(v, ts);
    detectFailures = 0;
    return r;
  } catch (e) {
    console.warn("검출 실패:", e);
    if (landmarker.mode === "holistic" && ++detectFailures >= 3 && !swapping) {
      swapping = true;
      landmarker.close();
      landmarker = null; // 교체 중 검출 일시정지 (루프의 null 가드가 보호)
      createFallbackLandmarker()
        .then((l) => { landmarker = l; hud.mode.textContent = l.mode; })
        .catch((err) => showBanner(`랜드마커 초기화 실패: ${err.message ?? err}`))
        .finally(() => { swapping = false; });
    }
    return null;
  }
}

// ---------------------------------------------------------------- 카메라

let currentFacing = "user";
let stream = null;

function applyMirror() {
  // 전면 카메라만 미러(CSS 전용) — 좌표 자체는 절대 반전하지 않아
  // 오버레이 정렬이 유지되고 서버 전송 좌표도 원본 그대로다.
  video.classList.toggle("mirror", currentFacing === "user");
  canvas.classList.toggle("mirror", currentFacing === "user");
}

async function startCamera(facing) {
  stream?.getTracks().forEach((t) => t.stop()); // 이전 트랙 반드시 정지
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: { ideal: facing }, // exact 금지: 단일 카메라 데스크톱에서도 무해
      width: { ideal: 640 },
      height: { ideal: 480 },
    },
  });
  video.srcObject = stream;
  await video.play();
  currentFacing = facing;
  applyMirror();
  lastVideoTime = -1;
}

video.addEventListener("loadedmetadata", () => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
});

// 카메라 권한 획득 후에만 장치 목록이 채워진다 → startCamera 성공 뒤 호출
async function maybeShowFlipButton() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices.filter((d) => d.kind === "videoinput");
  flipBtn.hidden = cams.length < 2;
}

flipBtn.addEventListener("click", async () => {
  flipBtn.disabled = true;
  try {
    await startCamera(currentFacing === "user" ? "environment" : "user");
  } catch (e) {
    showBanner(`카메라 전환 실패: ${e.message ?? e}`);
  } finally {
    flipBtn.disabled = false;
  }
});

// ---------------------------------------------------------------- WebSocket

class WSManager {
  constructor(url) {
    this.url = url;
    this.backoff = 500;
    this.connect();
  }
  connect() {
    this.ws = new WebSocket(this.url);
    this.ws.onopen = () => { this.backoff = 500; setStatus("연결됨", true); };
    this.ws.onmessage = (ev) => {
      let m;
      try { m = JSON.parse(ev.data); } catch { return; }
      if (m.type === "echo") {
        hud.frames.textContent = m.frame_id;
        hud.rtt.textContent = Math.max(0, Math.round(performance.now() - m.t));
        hud.dims.textContent = m.n_dims;
      } else if (m.type === "status") {
        recDot.hidden = m.state !== "signing";
      } else if (m.type === "result") {
        appendResult(m);
        if (m.text !== null) tts.speak(m.text);
      }
    };
    this.ws.onclose = () => {
      setStatus("재연결 중…", false);
      setTimeout(() => this.connect(), this.backoff);
      this.backoff = Math.min(this.backoff * 2, 8000); // 0.5s → 8s 지수 백오프
    };
    this.ws.onerror = () => this.ws.close();
  }
  sendFeatures(v, frameId, t) {
    if (this.ws.readyState !== WebSocket.OPEN) return;
    if (this.ws.bufferedAmount > WS_MAX_BUFFERED) return; // 밀리면 프레임 드롭
    this.ws.send(JSON.stringify({
      type: "features",
      frame_id: frameId,
      t,
      dim: v.length,
      // 4자리 반올림은 전송 전용 압축 — features.js 내부에는 반올림이 없다
      v: v.map((x) => Math.round(x * 1e4) / 1e4),
    }));
  }
}

// ---------------------------------------------------------------- 그리기

let drawingUtils = null;

function drawResult(res) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (res.pose) {
    drawingUtils.drawConnectors(res.pose, PoseLandmarker.POSE_CONNECTIONS,
      { color: "#4ade80", lineWidth: 2 });
    drawingUtils.drawLandmarks(res.pose, { color: "#4ade80", radius: 2 });
  }
  for (const [hand, color] of [[res.leftHand, "#60a5fa"], [res.rightHand, "#f97316"]]) {
    if (!hand) continue;
    drawingUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS,
      { color, lineWidth: 2 });
    drawingUtils.drawLandmarks(hand, { color, radius: 2 });
  }
  if (res.face) {
    // 윤곽만 그린다 — 478포인트 전체 테셀레이션은 fps를 크게 떨어뜨린다
    drawingUtils.drawConnectors(res.face, FaceLandmarker.FACE_LANDMARKS_CONTOURS,
      { color: "#f472b6", lineWidth: 1 });
  }
}

// ---------------------------------------------------------------- 메인 루프

let lastVideoTime = -1;
let lastDetectMs = 0;
let frameId = 0;
let wsm = null;

// MediaPipe 랜드마크 객체({x, y, …}) → features 모듈 입력 형식([x, y] 쌍).
// 반올림 없이 원본 정밀도 그대로 특징을 계산한다.
const toXY = (lm) => (lm ? lm.map((p) => [p.x, p.y]) : null);

function loop(nowMs) {
  requestAnimationFrame(loop);
  if (!landmarker || video.readyState < 2) return;
  if (video.currentTime === lastVideoTime) return;        // 새 프레임 없음
  if (nowMs - lastDetectMs < DETECT_INTERVAL_MS) return;  // ~15fps 스로틀
  lastVideoTime = video.currentTime;
  lastDetectMs = nowMs;

  // detectForVideo 타임스탬프는 단조 증가 필수 — 카메라 전환 시 초기화되는
  // video.currentTime 대신 performance.now()를 쓴다.
  const ts = Math.round(performance.now());
  const res = safeDetect(video, ts);
  if (!res) return;
  drawResult(res);

  const v = extractFeatures(
    toXY(res.pose), toXY(res.leftHand), toXY(res.rightHand), toXY(res.face));
  if (!v) {
    hud.feat.textContent = "무효 (포즈 미검출)";
    return; // 무효 프레임은 전송하지 않는다
  }
  // 디버그: 차원 수 + 정규화 후 값 범위 (마지막 마스크 2개 제외한 좌표부만)
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < FEATURE_DIM - 2; i++) {
    if (v[i] < mn) mn = v[i];
    if (v[i] > mx) mx = v[i];
  }
  hud.feat.textContent = `${v.length} [${mn.toFixed(2)}, ${mx.toFixed(2)}]`;
  wsm.sendFeatures(v, ++frameId, ts);
}

// ---------------------------------------------------------------- 부팅

try {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error(`카메라 API를 사용할 수 없습니다. 현재 주소(${location.origin})가 `
      + "HTTPS 또는 http://localhost 인지 확인하세요.");
  }
  landmarker = await createLandmarker();
  hud.mode.textContent = landmarker.mode;

  setStatus("카메라 여는 중…");
  await startCamera("user");
  await maybeShowFlipButton();

  drawingUtils = new DrawingUtils(ctx);
  wsm = new WSManager(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/translate`);
  requestAnimationFrame(loop);
} catch (e) {
  console.error(e);
  const msg = e.name === "NotAllowedError"
    ? "카메라 권한이 거부되었습니다. 브라우저 설정에서 카메라를 허용한 뒤 새로고침하세요."
    : (e.name === "NotReadableError" || e.name === "AbortError")
    ? "카메라를 시작할 수 없습니다. 카메라를 사용 중인 다른 앱/탭(Safari 등)을 닫고 새로고침하세요."
    : e.name === "NotFoundError"
    ? "사용 가능한 카메라를 찾지 못했습니다."
    : `초기화 실패: ${e.message ?? e}`;
  showBanner(msg);
  setStatus("오류", false);
}
