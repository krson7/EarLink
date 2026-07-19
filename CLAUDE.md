# 이어링크(EarLink) 지문자 인식 웹앱 — 개발 명세

이 문서는 이 프로젝트의 단일 기준 문서다. 구현 결정이 충돌하면 이 문서가 우선한다.
"사업"이 아닌 "연구" 용어를 문서·주석 전반에 사용한다.

## 1. 개요

농인이 스마트폰 브라우저 카메라 앞에서 한글 지문자(손으로 자모)를 하면, 자모를
실시간 인식·조합해 완성형 한글로 화면에 표시하고 TTS로 읽어주는 웹앱. 자모를
이어붙여 학습하지 않은 임의의 단어를 표현한다(open-vocabulary).

## 2. 아키텍처

```
[스마트폰/PC 브라우저]                    [FastAPI 서버]
 getUserMedia 카메라
 MediaPipe Tasks(JS/WASM)로               WebSocket /ws/translate
 손 21점 추출              ──손 키포인트──▶  55차원 시퀀스 버퍼(10)
 (영상은 절대 서버로 안 보냄)               tflite 자모 분류 + 디바운스
 조합 한글 표시 + Web Speech TTS ◀──자모──  한글 조합 오토마타
```

- 손 키포인트 추출은 **브라우저에서만**. 프레임 이미지를 서버로 보내는 코드를
  작성하지 않는다 (개인정보·대역폭 설계 원칙).
- 서버는 GPU 없이 CPU 추론.

## 3. 기술 스택

- 백엔드: Python 3.11+, FastAPI, uvicorn, TensorFlow(tflite 추론)
- 프런트: 순수 HTML/JS (빌드 도구 없이 단일 페이지), @mediapipe/tasks-vision CDN
  - Holistic Landmarker 사용(오른손 좌표만). 미지원 시 Pose+Hand+Face 조합 폴백

## 4. 특징(Feature) 설계

`server/fingerspelling.py`의 `frame_feature`가 손 21점을 55차원으로 변환한다:
- 뼈 단위벡터 20개(×2 = 40) — 인접 관절 벡터를 크기로 나눠 정규화(위치·크기 불변)
- 관절 각도 15개(도) — 손가락 굽힘
- 시퀀스 길이 10프레임을 모아 tflite에 입력.
- 오른손만 사용하며 좌표를 반전하지 않는다 — 모델이 오른손 원좌표로 학습됐다.
  왼손을 넣으면 x-반사로 뼈벡터가 오염된다.

## 5. 확정·조합 로직

- **디바운스**(`Debouncer`): 같은 자모를 N프레임(config `confirm_frames`, ~0.6초)
  유지해야 확정. 스쳐가는 손모양은 유지 시간이 짧아 걸러진다. 확정 후 손을 뗐다
  (저신뢰/미검출) 다시 하면 재확정 허용 → 연속 같은 자모(학교의 ㄱㄱ) 처리.
- **한글 조합**(`shared/hangul_compose.py`): 자모 시퀀스를 완성형으로 조합.
  유니코드 조합 공식(초성×588 + 중성×28 + 종성 + 0xAC00) + 받침 이동
  (ㅎㅏㄱㄱㅛ → 학교). 초성/중성/종성 상태머신.

## 6. 디렉토리 구조

```
EarLink/
├ CLAUDE.md
├ README.md
├ config.yaml
├ server/
│  ├ main.py            # FastAPI 앱, /ws/translate, / 정적 서빙
│  ├ fingerspelling.py  # 전처리·tflite 추론·디바운스·세션
│  └ schemas.py         # WebSocket 메시지 스키마
├ shared/
│  └ hangul_compose.py  # 자모 → 완성형 한글 조합 오토마타
├ web/
│  ├ index.html
│  └ app.js             # 카메라, MediaPipe, WebSocket, TTS
├ tests/
│  ├ test_fingerspelling.py
│  └ test_hangul_compose.py
├ conftest.py
└ models/fingerspelling/
   └ multi_hand_gesture_classifier.tflite
```

## 7. 모델

- tflite 분류기: 입력 (1, 10, 55) → 출력 31 자모 softmax.
- 라벨은 `server/fingerspelling.ACTIONS`(자음 14 + 모음 17).
- 재학습 시 `config.yaml`의 `fingerspelling.model_path`와 `ACTIONS`만 교체하면
  아키텍처 변경 없이 반영된다.

## 8. 실행

```
uvicorn server.main:app --reload
```
- 모바일 카메라는 HTTPS에서만 열린다. 개발은 localhost(예외 허용), 스마트폰
  테스트는 `uvicorn --ssl-keyfile --ssl-certfile`(mkcert) 또는 cloudflared/ngrok 터널.
- 테스트: `pytest -q`.

## 9. 제약·주의

- 프레임 이미지 서버 전송 금지. 손 키포인트 JSON만.
- 커밋은 한국어 메시지.
