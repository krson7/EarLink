"""이어링크(EarLink) FastAPI 서버 — 지문자 인식.

- `/ws/translate`: 손 21점 수신 → 자모 추론·조합 결과 전송
- `/`: web/ 정적 파일 서빙

개인정보 설계 원칙: 클라이언트는 손 키포인트 JSON만 전송한다.
영상 프레임을 수신·저장하는 코드를 이 서버에 추가하지 않는다.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from server.fingerspelling import FingerspellingSession, FingerspellPredictor
from server.schemas import EchoMsg, HandMsg, JamoMsg

ROOT = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    fs = cfg.get("fingerspelling") or {}
    app.state.fs_cfg = fs
    # tflite 로드(TF 지연 import)는 기동 시 1회. 실패해도 부팅은 막지 않는다.
    try:
        app.state.fs_predictor = FingerspellPredictor.load(
            ROOT / fs.get("model_path",
                          "models/fingerspelling/multi_hand_gesture_classifier.tflite"))
    except Exception as e:  # noqa: BLE001
        print(f"[경고] 지문자 모델 로드 실패: {e}")
        app.state.fs_predictor = None
    if app.state.fs_predictor is None:
        print("[경고] 지문자 tflite 없음 — 인식 비활성 (에코만 동작).")
    yield


app = FastAPI(title="이어링크 지문자 인식 (연구 시제품)", lifespan=lifespan)


@app.websocket("/ws/translate")
async def ws_translate(ws: WebSocket) -> None:
    """손 21점 스트림 → 에코(HUD용) + 자모 추론·조합 결과."""
    await ws.accept()
    fs_predictor = getattr(ws.app.state, "fs_predictor", None)
    fs = getattr(ws.app.state, "fs_cfg", None) or {}
    fs_session = FingerspellingSession(
        fs_predictor,
        confirm_frames=fs.get("confirm_frames", 8),
        conf_threshold=fs.get("conf_threshold", 0.8),
    )
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = HandMsg.model_validate(json.loads(raw))
            except (json.JSONDecodeError, ValidationError):
                continue  # 잘못된 메시지는 조용히 무시 (연결 유지)

            n = len(msg.hand) * 2 if msg.hand else 0
            await ws.send_text(EchoMsg(
                frame_id=msg.frame_id, t=msg.t, n_dims=n).model_dump_json())
            # tflite 호출(락)이 있어 스레드로 — 이벤트 루프 비차단.
            out = await asyncio.to_thread(fs_session.push, msg.hand)
            await ws.send_text(JamoMsg(**out).model_dump_json())
    except WebSocketDisconnect:
        pass


# "/" 마운트는 catch-all이므로 반드시 WebSocket 라우트 등록 뒤에 온다.
app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    uvicorn.run(
        "server.main:app",
        host=cfg["server"]["host"],
        port=cfg["server"]["port"],
        reload=True,
    )
