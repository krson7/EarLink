"""이어링크(EarLink) FastAPI 서버 — M5.

- `/ws/translate`: 특징 벡터 수신 → 에코 + 발화 구간 탐지 → 구간 분류 결과 전송
- `/`: web/ 정적 파일 서빙

개인정보 설계 원칙: 클라이언트는 정규화된 특징 벡터 JSON만 전송한다.
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

from server.inference import Predictor, SegConfig, UtteranceSession
from server.schemas import EchoMsg, FeaturesMsg, ResultMsg, StatusMsg
from shared.features import FEATURE_DIM

ROOT = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    app.state.seg_cfg = SegConfig.from_yaml(cfg)
    # TF import + 모델 로드(~수 초)는 기동 시 1회 (CLAUDE.md §7: 최신 버전 로드)
    app.state.predictor = Predictor.load_latest(
        ROOT / "models",
        win=cfg["model"]["window_frames"],
        conf_thr=cfg["model"]["confidence_threshold"],
        seg_cfg=app.state.seg_cfg,
    )
    if app.state.predictor is None:
        print("[경고] models/ksl_word_v*.keras 없음 — 발화 탐지만 동작 (결과 미전송). "
              "python -m training.train 으로 모델을 먼저 학습하세요.")
    yield


app = FastAPI(title="이어링크 KSL 수어 인식 (연구 시제품)", lifespan=lifespan)


@app.websocket("/ws/translate")
async def ws_translate(ws: WebSocket) -> None:
    """특징 벡터 스트림 → 에코(HUD용) + 발화 탐지 + 세그먼트 분류 결과."""
    await ws.accept()
    # getattr 기본값: lifespan 없이도(예: 에코만 검증하는 테스트) 핸들러가 동작
    session = UtteranceSession(
        getattr(ws.app.state, "seg_cfg", None) or SegConfig())  # 연결별 버퍼·상태
    predictor: Predictor = getattr(ws.app.state, "predictor", None)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = FeaturesMsg.model_validate(json.loads(raw))
            except (json.JSONDecodeError, ValidationError):
                continue  # 잘못된 메시지는 조용히 무시 (연결 유지)

            # 에코는 모든 유효 features 메시지에 (M2 계약 — HUD frames/RTT/dims)
            await ws.send_text(EchoMsg(
                frame_id=msg.frame_id, t=msg.t, n_dims=len(msg.v),
            ).model_dump_json())

            if len(msg.v) != FEATURE_DIM:
                continue  # 계약 차원이 아닌 프레임은 탐지기에 넣지 않음

            for kind, payload in session.push(msg.v, msg.t):
                if kind == "status":
                    await ws.send_text(StatusMsg(state=payload).model_dump_json())
                elif kind == "segment" and predictor is not None:
                    seg, times = payload
                    # 추론(후보 배치 1회 + 드물게 재트레이싱)은 스레드로 —
                    # 이벤트 루프를 막지 않아 다른 연결의 에코/RTT 가 유지된다
                    result: ResultMsg = await asyncio.to_thread(
                        predictor.predict, seg, times)
                    await ws.send_text(result.model_dump_json())
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
