# 이어링크(EarLink) 수어 인식 웹앱 — 개발 명세

이 문서는 Claude Code가 이 프로젝트를 구현할 때 따라야 할 단일 기준 문서다.
모든 구현 결정이 이 문서와 충돌하면 이 문서가 우선한다.

## 1. 프로젝트 개요

농인이 스마트폰 브라우저 카메라 앞에서 수어를 하면, 실시간으로 한국어 텍스트로
변환해 화면에 표시하고 TTS로 읽어주는 웹앱. 연구(이어링크) 시제품이며,
"사업"이 아닌 "연구" 용어를 문서·주석 전반에 사용한다.

## 2. 아키텍처 (확정 — 변경 금지)

```
[스마트폰/PC 브라우저]                        [FastAPI 서버]
 getUserMedia 카메라
 MediaPipe Tasks(JS/WASM)로                   WebSocket /ws/translate
 Holistic 키포인트 추출          ──키포인트──▶  슬라이딩 윈도우 버퍼
 (영상은 절대 서버로 보내지 않음)               발화 시작/종료 탐지
 결과 텍스트 표시 + Web Speech TTS ◀──글로스──  Conv1D-LSTM 추론 (TF/Keras)
```

- 키포인트 추출은 **브라우저에서만** 수행한다. 프레임 이미지를 서버로 전송하는
  코드를 작성하지 않는다 (개인정보·대역폭 설계 원칙).
- 서버는 GPU 없이 CPU 추론이 가능해야 한다.

## 3. 기술 스택

- 백엔드: Python 3.11+, FastAPI, uvicorn, websockets, TensorFlow/Keras (추론·학습 공용)
- 프런트: 순수 HTML/JS (빌드 도구 없이 단일 페이지), @mediapipe/tasks-vision CDN 사용
  - Holistic Landmarker 사용. 미지원 환경 폴백으로 Pose + Hand + Face Landmarker 조합 허용
- 학습 파이프라인: Python 스크립트 (opencv-python, mediapipe, numpy, tensorflow)

## 4. 데이터

- 원천: AI Hub "수어 영상"(KSL-Guide) 데이터셋. 정면 뷰(`*_F.mp4`)만 사용.
- 라벨: `NIA_SEN_train.csv`, `NIA_SEN_val.csv`
  - **인코딩 CP949** — 반드시 `encoding='cp949'`로 읽는다.
  - 컬럼: `Num, Filename, Kor` (Kor = 띄어쓰기로 구분된 글로스 시퀀스, 어휘 324개)
- 1차 목표 어휘: 글로스 빈도 상위 30개 단어의 **단일 글로스 영상**부터 학습
  (문장 CSLR은 2차 단계). CSV에서 글로스 1개짜리 행 + KSL-Guide-Word 영상 활용.
- 데이터 경로는 `.env` 또는 `config.yaml`의 `DATA_ROOT`로 주입 (하드코딩 금지).

## 5. 특징(Feature) 설계 — 학습과 추론에서 완전히 동일해야 함

이 로직은 `shared/features.py` 한 곳에 구현하고, JS에는 동일 로직을 이식한 뒤
**교차 검증 테스트**(같은 입력 → 같은 출력, 허용오차 1e-5)를 둔다.

1. 키포인트 선별 (KSL-Guide 논문의 124포인트 구성을 MediaPipe로 재현):
   - Pose: 상체 12개 — nose(0), 어깨(11,12), 팔꿈치(13,14), 손목(15,16),
     눈(2,5), 귀(7,8), 입꼬리 대용 mouth(9,10)
   - 양손: 21 × 2 = 42개 전부
   - Face Mesh: 눈썹·눈·입 윤곽 중심으로 약 70개 서브셋
     (비수지 신호: 눈썹 올림=의문문, 입 모양이 핵심. 인덱스는 MediaPipe
     canonical face mesh 기준으로 선정하고 상수 리스트로 고정)
   - 총 124개 포인트 × (x, y) = 248차원/프레임 (z는 1차 버전에서 제외)
2. 정규화: 양어깨(11,12) 중점을 원점으로 평행이동 → 어깨 간 거리로 나눠 스케일
   통일. 손·얼굴 포인트도 같은 기준으로 정규화.
3. 결측 처리: 손이 화면 밖이면 해당 42/84차원을 0으로 채우고 마스크 플래그 추가.
4. 시퀀스: 30fps 기준 슬라이딩 윈도우 60프레임(2초), stride 15.

## 6. 발화 구간 탐지 (서버측)

키포인트 프레임 간 (x,y) 변화량 평균을 계산, 이동평균 임곗값을 넘는 시점을
발화 시작, 다시 내려가는 시점을 발화 종료로 판정 (선행연구 기법).
탐지된 구간만 분류기에 넣는다. 임곗값은 config로 조정 가능하게.

## 7. 모델

- 구조: Conv1D(64, k=3) → MaxPool → Conv1D(128, k=3) → Bi-LSTM(64) →
  Dense(64, relu) → Dense(N_CLASSES, softmax)
- 입력: (60, 248+마스크)
- 저장: `models/ksl_word_v{n}.keras` + 라벨맵 JSON. 서버 기동 시 최신 버전 로드.
- 평가: val 정확도, F1(macro), 혼동행렬 이미지 저장.

## 8. 디렉토리 구조

```
earlink-ksl/
├ CLAUDE.md                  # 이 문서
├ config.yaml
├ server/
│  ├ main.py                 # FastAPI 앱, /ws/translate, / (정적 서빙)
│  ├ inference.py            # 모델 로드, 윈도우 버퍼, 발화 탐지
│  └ schemas.py
├ shared/
│  └ features.py             # 키포인트 선별·정규화 (학습/추론 공용)
├ web/
│  ├ index.html
│  ├ app.js                  # 카메라, MediaPipe, WebSocket, TTS
│  └ features.js             # features.py와 동일 로직
├ training/
│  ├ extract_keypoints.py    # 영상 폴더 → MediaPipe 재추출 → .npy
│  ├ build_dataset.py        # CSV(cp949) 파싱, 상위 30 글로스 필터, 윈도우화
│  ├ train.py
│  └ evaluate.py
├ tests/
│  ├ test_features_parity.py # Python↔JS 특징 일치 검증 (node 실행 비교)
│  └ test_ws_smoke.py
└ models/
```

## 9. 구현 순서 (마일스톤 — 순서대로, 각 단계 동작 확인 후 다음 진행)

1. **M1 스캐폴드**: FastAPI + 정적 페이지. 브라우저에서 카메라 열고 Holistic
   키포인트를 캔버스에 실시간 시각화. WebSocket 에코까지.
2. **M2 특징 파이프라인**: shared/features.py + web/features.js + 일치 테스트.
3. **M3 데이터 전처리**: extract_keypoints.py로 영상 일부(10개 글로스 × 20영상
   샘플)만 먼저 처리해 파이프라인 검증 → 전체 30 글로스 처리.
4. **M4 학습**: train.py, val 정확도 ≥ 80% 목표. 미달 시 증강(시간 스케일,
   좌표 노이즈, 좌우 반전+라벨 유지) 추가.
5. **M5 실시간 연결**: 서버 추론 + 발화 탐지 + 프런트 결과 표시 + Web Speech TTS.
6. **M6 마감**: README(실행법, HTTPS 안내), 데모 스크립트.

## 10. 제약·주의사항

- 모바일 카메라는 HTTPS에서만 열린다. 개발은 localhost(예외 허용),
  스마트폰 테스트는 `uvicorn --ssl-keyfile --ssl-certfile`(mkcert) 또는
  cloudflared/ngrok 터널 사용법을 README에 기재.
- CSV는 항상 cp949. 출력 파일·소스코드는 UTF-8.
- MediaPipe legacy solution(`mp.solutions.holistic`) 대신 **Tasks API** 우선.
  단, Python 쪽 재추출 스크립트에서 Tasks API의 holistic이 불안정하면
  pose+hands+face_mesh 조합으로 동일 124포인트를 구성해도 된다 (인덱스 상수 공유).
- 프레임 이미지 서버 전송 금지. 키포인트 JSON(또는 Float32Array 바이너리)만.
- 커밋은 마일스톤 단위로, 한국어 커밋 메시지.
