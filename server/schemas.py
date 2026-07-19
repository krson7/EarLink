"""WebSocket 메시지 Pydantic 스키마 — 프로토콜의 자기문서화.

수신 (client → server):
  FeaturesMsg  — 정규화된 특징 벡터. v 길이는 강제하지 않는다: 에코는 모든
                 유효 features 메시지에 응답하는 것이 M2 계약이고(HUD 디버깅),
                 길이 250(FEATURE_DIM)인 프레임만 발화 탐지기에 투입한다.

지문자 모드 (Track 2):
  HandMsg      — 손 21점 원시 좌표 (키포인트만 — 개인정보 불변, 이미지 아님)
  JamoMsg      — 지문자 인식 결과: 현재 자모 + 확정 자모 + 조합된 한글(라이브)

발신 (server → client):
  EchoMsg      — 프레임 수신 확인 (frames/RTT/dims HUD용, M2와 동일)
  StatusMsg    — 발화 상태 전이 시에만 (signing/idle)
  ResultMsg    — 세그먼트 분류 결과. confidence < 임곗값이면 text=None
                 ("인식 안 됨") — 화면 표시·TTS 문자열은 text 하나로 계약해
                 저신뢰 단어가 실수로 발화될 경로를 차단한다.
"""

from typing import Literal, Optional

from pydantic import BaseModel


class FeaturesMsg(BaseModel):
    type: Literal["features"]
    frame_id: int
    t: float
    v: list[float]


class EchoMsg(BaseModel):
    type: Literal["echo"] = "echo"
    frame_id: int
    t: float
    n_dims: int


class StatusMsg(BaseModel):
    type: Literal["status"] = "status"
    state: Literal["signing", "idle"]


class ResultMsg(BaseModel):
    type: Literal["result"] = "result"
    text: Optional[str]        # None = 인식 안 됨 (저신뢰)
    top1: str
    confidence: float
    n_frames: int
    low_confidence: bool


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
