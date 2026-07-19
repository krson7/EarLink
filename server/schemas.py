"""WebSocket 메시지 Pydantic 스키마 — 프로토콜의 자기문서화.

수신 (client → server):
  HandMsg  — 손 21점 원시 좌표 (키포인트만 — 개인정보 불변, 이미지 아님)

발신 (server → client):
  EchoMsg  — 프레임 수신 확인 (frames/RTT/dims HUD용)
  JamoMsg  — 지문자 인식 결과: 현재 자모 + 확정 자모 + 조합된 한글(라이브)
"""

from typing import Literal, Optional

from pydantic import BaseModel


class EchoMsg(BaseModel):
    type: Literal["echo"] = "echo"
    frame_id: int
    t: float
    n_dims: int


class HandMsg(BaseModel):
    type: Literal["hand"]
    frame_id: int
    t: float
    hand: Optional[list[list[float]]]  # 21×2 원시 손 좌표, 또는 null(손 미검출)


class JamoMsg(BaseModel):
    type: Literal["jamo"] = "jamo"
    current: Optional[str]        # 현재 프레임 top-1 자모 (버퍼 미충족 시 None)
    confidence: Optional[float]
    confirmed: Optional[str]      # 이번 프레임에 디바운스 확정된 자모 (없으면 None)
    composed: str                 # 지금까지 조합된 한글 (라이브, ㅎ→하→학)
