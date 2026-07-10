"""이어링크(EarLink) FastAPI 서버 — M2.

- `/ws/translate`: 특징 벡터(250차원) 에코 (M5에서 발화 탐지 + 추론으로 대체)
- `/`: web/ 정적 파일 서빙

개인정보 설계 원칙: 클라이언트는 키포인트 JSON만 전송한다.
영상 프레임을 수신·저장하는 코드를 이 서버에 추가하지 않는다.
"""

import json
from pathlib import Path

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="이어링크 KSL 수어 인식 (연구 시제품)")

@app.websocket("/ws/translate")
async def ws_translate(ws: WebSocket) -> None:
    """M2: 특징 벡터 메시지를 요약해 에코한다. M5에서 슬라이딩 윈도우 + 추론으로 대체."""
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") != "features" or not isinstance(msg.get("v"), list):
                continue
            await ws.send_text(json.dumps({
                "type": "echo",
                "frame_id": msg.get("frame_id"),
                # 클라이언트 타임스탬프를 그대로 반환 → 클라이언트가 무상태로 RTT 계산
                "t": msg.get("t"),
                # 클라이언트의 dim 주장이 아니라 실제 배열 길이를 계산
                "n_dims": len(msg["v"]),
            }))
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
