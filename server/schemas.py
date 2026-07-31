"""WebSocket 메시지 Pydantic 스키마 — 프로토콜의 자기문서화.

수신 (client → server):
  HandMsg   — 손 21점 원시 좌표
  ResetMsg  — 현재 지문자 인식 및 한글 조합 상태 초기화

발신 (server → client):
  EchoMsg           — 프레임 수신 확인
  JamoMsg           — 지문자 인식 결과
  ResetCompleteMsg  — 지문자 상태 초기화 완료
"""

from typing import Literal, Optional

from pydantic import BaseModel


class EchoMsg(BaseModel):
    """손 좌표 프레임 수신 확인 메시지."""

    type: Literal["echo"] = "echo"
    frame_id: int
    t: float
    n_dims: int


class HandMsg(BaseModel):
    """클라이언트가 전송하는 손 키포인트 메시지."""

    type: Literal["hand"]
    frame_id: int
    t: float

    # 21×2 원시 손 좌표 또는 null
    hand: Optional[list[list[float]]]


class JamoMsg(BaseModel):
    """서버가 반환하는 지문자 인식 결과."""

    type: Literal["jamo"] = "jamo"

    # 현재 프레임의 top-1 자모
    # 시퀀스 버퍼가 아직 채워지지 않았다면 None
    current: Optional[str]

    # 현재 자모의 모델 신뢰도
    confidence: Optional[float]

    # 이번 프레임에서 디바운스로 확정된 자모
    confirmed: Optional[str]

    # 지금까지 조합된 한글
    composed: str


class ResetMsg(BaseModel):
    """현재 WebSocket의 지문자 인식 상태 초기화 요청."""

    type: Literal["reset"]


class ResetCompleteMsg(BaseModel):
    """지문자 인식 상태 초기화 완료 응답."""

    type: Literal["reset_complete"] = "reset_complete"