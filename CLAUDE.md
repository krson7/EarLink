# 이어링크(EarLink) 수어 인식 웹앱 — 개발 명세

이 문서는 Claude Code가 이 프로젝트를 구현할 때 따라야 할 단일 기준 문서다.
모든 구현 결정이 이 문서와 충돌하면 이 문서가 우선한다.

## 1. 프로젝트 개요

농인이 스마트폰 브라우저 카메라 앞에서 수어를 하면, 실시간으로 한국어 텍스트로
변환해 화면에 표시하고 TTS로 읽어주는 웹앱. 연구(이어링크) 시제품이며,
"사업"이 아닌 "연구" 용어를 문서·주석 전반에 사용한다.

**트랙 구조** (2026-07-10 확정): "핵심 어휘 단어 인식 + 지문자 기반 개방
어휘"의 2트랙.
- **Track 1 — 단어 인식** (§4~§9): 학습된 30단어 실시간 인식. B안(단일
  화자)으로 파이프라인 실현성 입증 완료, 다화자(A안) 재학습은 보류.
- **Track 2 — 지문자·지숫자** (§11): 자모를 이어붙여 학습하지 않은 임의의
  단어·숫자를 표현(open-vocabulary). 현재 우선 진행 트랙.

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
- (현재 구현 주석) 와이어의 "키포인트"는 M2부터 브라우저에서 §5 정규화까지
  마친 **250차원 특징 벡터**로 전송된다 — 원시 좌표도 보내지 않으며,
  개인정보 원칙은 동일하게 유지된다.

## 3. 기술 스택

- 백엔드: Python 3.11+, FastAPI, uvicorn, websockets, TensorFlow/Keras (추론·학습 공용)
- 프런트: 순수 HTML/JS (빌드 도구 없이 단일 페이지), @mediapipe/tasks-vision CDN 사용
  - Holistic Landmarker 사용. 미지원 환경 폴백으로 Pose + Hand + Face Landmarker 조합 허용
- 학습 파이프라인: Python 스크립트 (mediapipe==0.10.35, numpy, tensorflow==2.21.0)
  - cv2는 mediapipe의 전이 의존성(opencv-contrib-python)을 쓴다 —
    **opencv-python을 별도 설치하지 말 것** (동일 cv2 패키지 이중 설치 충돌)

## 4. 데이터 — Track 1 (2026-07-10 실데이터 구조 반영)

- 원천: AI Hub "수어 영상"(KSL-Guide) 데이터셋. 정면 뷰(`*_F.mp4`)만 사용.
  - 파일명 패턴: `NIA_SL_WORD####_REAL##_[F/U/D/R/L].mp4` (약어는 **WORD**, WRD 아님)
  - 영상 스캔은 DATA_ROOT 하위 `원천데이터/` 트리로만 제한.
  - 제공되는 OpenPose 키포인트 JSON(`라벨링데이터/`)은 읽지도 지우지도 않는다 —
    학습-추론 추출기 일치 원칙(A안)에 따라 mp4에서 MediaPipe로 재추출한다.
- 라벨: 실데이터 샘플에 NIA_SEN CSV가 없어, morpheme 라벨에서 추출한 자체
  CSV가 정본이다 — `training/earlink_30words_candidates.csv`(후보 40) →
  유사도 스크리닝(`analyze_similarity.py`) → `training/earlink_final_30words.csv`(최종 30).
  - 인코딩 **utf-8-sig** (자체 산출물 — §10의 cp949는 AI Hub 원본 CSV에만 해당)
  - 핵심 컬럼: `word_id, korean_word(라벨 정본), video_filename, start_sec, end_sec`
  - 각 영상은 `start_sec~end_sec`(morpheme 구간)로 트리밍한 뒤 추출한다
    (앞뒤 대기 자세 제거).
- 1차 목표 어휘: 최종 30단어의 **단일 단어 영상** (문장 CSLR은 2차 단계).
- 데이터 경로는 `.env` 또는 `config.yaml`의 `DATA_ROOT`로 주입 (하드코딩 금지).

## 5. 특징(Feature) 설계 — Track 1, 학습과 추론에서 완전히 동일해야 함

이 로직은 `shared/features.py` 한 곳에 구현하고, JS에는 동일 로직을 이식한 뒤
**교차 검증 테스트**(같은 입력 → 같은 출력, 허용오차 1e-5)를 둔다.

1. 키포인트 선별 (KSL-Guide 논문의 124포인트 구성을 MediaPipe로 재현):
   - Pose: 상체 12개 — 어깨(11,12), 팔꿈치(13,14), 손목(15,16),
     눈(2,5), 귀(7,8), 입꼬리 대용 mouth(9,10)
     (2026-07-09: 원안의 nose(0)를 제외해 12개·총 124를 맞춤 — 머리 위치
     정보는 눈·귀가 대체)
   - 양손: 21 × 2 = 42개 전부
   - Face Mesh: 눈썹·눈·입 윤곽 중심 70개 서브셋 — 입술 40 + 눈썹 10×2 +
     눈 5×2, `shared/features.py`의 `FACE_SUBSET_INDICES`로 고정
     (비수지 신호: 눈썹 올림=의문문, 입 모양이 핵심. 인덱스는 MediaPipe
     canonical face mesh 기준)
   - 총 124개 포인트 × (x, y) = 248차원/프레임 (z는 1차 버전에서 제외)
2. 정규화: 양어깨(11,12) 중점을 원점으로 평행이동 → 어깨 간 거리로 나눠 스케일
   통일. 손·얼굴 포인트도 같은 기준으로 정규화.
3. 결측 처리: 손이 화면 밖이면 해당 42/84차원을 0으로 채우고 마스크 플래그 추가
   (최종 벡터 250차원 = 248 + 마스크 2).
4. 시퀀스: 윈도우 45프레임(1.5초, 30fps 기준), 짧은 시퀀스는 뒤를 0으로 패딩.
   (2026-07-10: morpheme 구간 트리밍 후 대부분 시퀀스가 60프레임 미만이라
   60→45로 조정. 윈도잉 로직은 `shared/windowing.py`로 학습·추론 공용.)

## 6. 발화 구간 탐지 (서버측)

키포인트 프레임 간 (x,y) 변화량 평균을 계산, 이동평균 임곗값을 넘는 시점을
발화 시작, 다시 내려가는 시점을 발화 종료로 판정 (선행연구 기법).
탐지된 구간만 분류기에 넣는다. 임곗값은 config로 조정 가능하게.
(구현: 팔·손 포인트 유클리드 평균 + EMA + 히스테리시스 —
`server/inference.py`, 파라미터는 `config.yaml`의 `segmentation:`)

**M5 실시간 처리 보강** (2026-07-10, 파리티 버그 수정 — 원본 재현율 96.7%):
- **fps 정규화**: 서버가 클라이언트 타임스탬프로 실제 fps를 추정해
  (`inference.py estimate_fps`, [5,60] 밖이면 생략) 세그먼트를 30fps 등가로
  리샘플 후 패딩 — 학습(30fps)과 클라이언트(~15fps)의 시간 밀도를 정렬.
- **발화 탐지 완화**: `end_patience_frames: 8`, `end_threshold: 0.02` —
  수어 중간의 짧은 멈춤에서 발화가 두 토막 나지 않게 (config로 튜닝).
- **전환 동작 대응**: 모션 에너지 양끝 트리밍(morpheme 트리밍 근사) +
  (시작×길이) 그리드 다중 윈도우 후보를 한 배치로 추론해 최고 confidence 채택.
- **컨텍스트 증강 재학습(v2)**: morpheme 구간 ±0.7초 여유 재추출
  (`extract_keypoints --margin-sec 0.7` → `data/processed_ctx`) + 컨텍스트
  지터/밀도 왕복 증강 — 전환 동작 조각이 분포 밖(OOD)이라 과신 오답을 내던
  문제를 해소. 산출물 `ksl_word_v2.keras`.
- **A안(다화자) 이월과제**: ① max-confidence 후보 선택은 일반화 모델에서
  confidence 보정 필요(temperature scaling/투표), ② 모션 임곗값을
  "프레임당 변위"에서 "초당 변위"로 전환(fps 독립화).

## 7. 모델 — Track 1

- 구조: Conv1D(64, k=3) → MaxPool → Conv1D(128, k=3) → Bi-LSTM(64) →
  Dense(64, relu) → Dense(N_CLASSES, softmax)
- 입력: (45, 250) — 45프레임 윈도우 × (248 좌표 + 손 마스크 2).
  (2026-07-10: §5-4 윈도우 조정에 따라 60→45)
- 저장: `models/ksl_word_v{n}.keras` + 라벨맵 JSON. 서버 기동 시 최신 버전 로드.
  - 현재: v1(트리밍 구간만 학습), **v2(컨텍스트 증강 — 서버가 자동 선택)**.
- 평가: val 정확도, 혼동행렬 이미지 저장 (F1(macro)는 미구현 — A안 재학습
  시 추가 예정). 종단 재현율 벤치마크는 `training/evaluate_e2e.py`.

## 8. 디렉토리 구조

```
earlink-ksl/
├ CLAUDE.md                  # 이 문서
├ config.yaml
├ server/
│  ├ main.py                 # FastAPI 앱, /ws/translate, / (정적 서빙)
│  ├ inference.py            # 모델 로드, 윈도우 버퍼, 발화 탐지
│  └ schemas.py
├ conftest.py                # pytest 임포트 앵커
├ shared/
│  ├ features.py             # 키포인트 선별·정규화 (학습/추론 공용)
│  └ windowing.py            # 시간축 윈도잉·리샘플 (학습/추론 공용)
├ web/
│  ├ index.html
│  ├ app.js                  # 카메라, MediaPipe, WebSocket, TTS
│  ├ features.js             # features.py와 동일 로직
│  └ package.json            # Node가 features.js를 ESM으로 읽게 함 (패리티 테스트)
├ training/
│  ├ earlink_30words_candidates.csv  # 후보 40단어 (utf-8-sig, 라벨·트리밍 정본)
│  ├ earlink_final_30words.csv       # 최종 30단어
│  ├ extract_keypoints.py    # 후보 CSV 영상 → 트리밍 → MediaPipe 재추출 → .npy
│  ├ build_dataset.py        # CSV 파싱·검증, 영상 스캔, label_map 생성
│  ├ analyze_similarity.py   # 단어 선별용 유사도(DTW/코사인) 스크리닝
│  ├ evaluate_e2e.py         # 원본 영상 종단 재현율 벤치마크 (M5 신설)
│  ├ train.py
│  └ evaluate.py
├ tests/
│  ├ test_features_parity.py # Python↔JS 특징 일치 검증 (node 실행 비교)
│  ├ test_segmenter.py       # 발화 탐지 상태기계 단위테스트 (TF 불필요)
│  └ test_ws_smoke.py
└ models/
```

## 9. 구현 순서 — Track 1 마일스톤 (순서대로, 각 단계 동작 확인 후 진행)

1. ✅ **M1 스캐폴드**: FastAPI + 정적 페이지. 브라우저에서 카메라 열고 Holistic
   키포인트를 캔버스에 실시간 시각화. WebSocket 에코까지. (맥/스마트폰 확인)
2. ✅ **M2 특징 파이프라인**: shared/features.py + web/features.js + 일치 테스트
   (패리티 오차 0.0).
3. ✅ **M3 데이터 전처리**: 샘플 3개 ID로 파이프라인 검증 → 후보 40단어
   morpheme 트리밍 추출 → 유사도 스크리닝으로 최종 30단어 확정.
4. ✅ **M4 학습**: train.py — **B안**(REAL01 단일 화자)으로 학습→저장→추론
   완주 검증. 산출물 `ksl_word_v1.keras`/`ksl_word_v2.keras`(컨텍스트 증강).
   과적합은 예상된 결과이며(val은 증강본 홀드아웃 — 누수 있는 참고 지표),
   원안의 "val 정확도 ≥ 80%" 목표는 다화자 재학습 시점에 적용.
5. ✅ **M5 실시간 연결**: 서버 추론 + 발화 탐지 + 프런트 결과 표시 + Web
   Speech TTS. 파리티 버그 수정(§6 보강) 후 **원본 영상 종단 재현율 96.7%**.
6. **M6 마감**: README(실행법, HTTPS 안내), 데모 스크립트.

**현재 상태 (2026-07-10)**: Track 1(단어)은 B안으로 파이프라인 실현성 입증
완료. 다화자(A안) 재학습은 보류. **지문자(Track 2, §11)를 우선 진행.**

## 10. 제약·주의사항

- 모바일 카메라는 HTTPS에서만 열린다. 개발은 localhost(예외 허용),
  스마트폰 테스트는 `uvicorn --ssl-keyfile --ssl-certfile`(mkcert) 또는
  cloudflared/ngrok 터널 사용법을 README에 기재.
- AI Hub 원본 CSV는 cp949. 단, 자체 제작 CSV(`training/earlink_*.csv`)는
  **utf-8-sig**(BOM 포함)로 읽는다. 출력 파일·소스코드는 UTF-8.
- macOS는 한글 파일·폴더명을 NFD(자모 분해형)로 저장한다 — 경로 문자열을
  비교할 때는 반드시 NFC로 정규화할 것 (`build_dataset._nfc` 참조.
  경로 조회 자체는 정규화 무관하지만 `==` 비교는 실패한다).
- MediaPipe legacy solution(`mp.solutions.holistic`) 대신 **Tasks API** 우선.
  단, Python 쪽 재추출 스크립트에서 Tasks API의 holistic이 불안정하면
  pose+hands+face_mesh 조합으로 동일 124포인트를 구성해도 된다 (인덱스 상수 공유).
  (2026-07-10: mediapipe 0.10.35의 Python HolisticLandmarker는 detect 프로브
  후 안정 동작 확인 — 실제 추출은 holistic 모드로 수행됨. combo 폴백은
  extract_keypoints.py에 유지하되, combo의 단독 FaceLandmarker는 전신
  원거리 화면에서 얼굴을 못 찾을 수 있음에 유의.)
- 프레임 이미지 서버 전송 금지. 키포인트 JSON(또는 Float32Array 바이너리)만.
- 커밋은 마일스톤 단위로, 한국어 커밋 메시지.

## 11. Track 2: 지문자·지숫자 (개방 어휘 확장)

### 11.1 목적과 근거

단어 모델(Track 1)은 학습된 어휘만 인식하는 한계가 있다. 지문자(한글 자모를
손으로)와 지숫자를 인식하면, 자모를 이어붙여 **학습하지 않은 임의의
단어·숫자**를 표현할 수 있다(open-vocabulary). 선행사례(Ghoney99, KT AIVLE)가
단일 화자 소량 데이터로도 지문자 ~81%를 달성해 실현성이 확인됨. 또한
지문자는 정지에 가까운 손모양이라 Track 1에서 가장 어려웠던 "연속 동작의
발화 구간 분절" 문제가 크게 완화됨.

### 11.2 대상 클래스 (확정 전 데이터 확인 필요)

- 표준 한글 지문자: 자음 19 + 모음 21 = **40개 자모** (한글 맞춤법 자모 순서 기준)
- 지숫자: 0~9 + 자주 쓰는 수 표현
- 주의: 일부 자모(쌍시옷 등)와 일부 지숫자는 **완전 정지가 아니라 짧은
  손 방향 변화·동작이 있음** — "정지 이미지 1장" 가정 금지. 짧은 시퀀스로 처리.
- AI Hub의 FINSP 데이터는 "자모 40개"가 아니라 자모로 표현한 단어/숫자
  1,000개(지명·이름 등)임. 낱자 단위 라벨 확보는 11.4에서 결정.

### 11.3 데이터

- 원천: AI Hub 수어영상 CROWD(크라우드소싱) 분. 파일명
  `NIA_SL_FINSP####_CROWD##_F`. 정면(F)만 존재 → 각도 필터링 불필요.
- 다운로드 대상(파일키): 원천 `01_crowd_video.zip`(Training),
  라벨 `01_crowd_morpheme.zip`. keypoint json(OpenPose)은 Track 1과 동일하게
  미사용 (학습-추론 추출기 일치 원칙).
- morpheme json의 name이 지문자/지숫자 라벨 소스.

### 11.4 낱자 라벨 확보 전략 (데이터 확인 후 택1)

- **옵션 A (권장 시작점)**: morpheme의 형태소 단위가 이미 자모/숫자 낱자로
  분할돼 있으면 그대로 사용. Cowork로 morpheme 구조부터 확인.
- **옵션 B**: 낱자 분할이 없으면, 지문자 단어 영상을 morpheme 구간으로 나눠
  각 구간을 자모 라벨로 매핑(글자 경계는 morpheme timestamp 활용).
- 데이터 확인 전에는 클래스 수·라벨 방식을 확정하지 않는다.

### 11.5 파이프라인 재사용 (Track 1 자산 최대 활용)

- **그대로 재사용**: MediaPipe Holistic 추출, shared/features 정규화(어깨 기준),
  FastAPI+WebSocket 서버, 웹 카메라/TTS, fps 정규화(§6 보강).
- **수정**:
  - 특징: 지문자는 손 중심 → 손 42점 위주로 쓰되, 손 위치 판별을 위해
    어깨 정규화는 유지(팔·몸 최소 포함). 얼굴은 제외 가능(비수지 거의 없음).
  - 시퀀스 길이: 45 → **짧게(예: 10~15프레임)**. 정지~짧은 동작이라 짧아도 됨.
  - 모델: 동일 Conv1D-LSTM 계열, 입력 차원·시퀀스만 변경. 별도 파일
    `models/ksl_finger_v1.keras` + 라벨맵 (Track 1 모델과 독립 버저닝).

### 11.6 신규 컴포넌트 (Track 2 전용 — 아래 파일들은 예정, §8 트리에는 미기재)

1. **글자 경계 판정**: 연속 지화에서 "한 자모 끝 → 다음 시작"을 끊는 로직.
   손모양이 안정적으로 유지되는 구간 = 한 글자로 판정(정지 기반이라 Track 1
   동작 분절보다 쉬움). config로 임곗값 노출.
2. **한글 조합 오토마타**: 인식된 낱자 시퀀스(ㅎ,ㅏ,ㄱ,ㅛ)를 완성형
   글자(학교)로 조합. 한글 유니코드 조합 공식(초성×588 + 중성×28 + 종성 +
   0xAC00) 사용. 초성/중성/종성 상태머신. 별도 모듈
   `shared/hangul_compose.py`(예정).
3. 웹 UI: 조합 중인 글자를 실시간 표시(예: ㅎ→하→학), 완성 시 확정.

### 11.7 Track 2 마일스톤

- T2-M1: FINSP/CROWD 데이터 확인(Cowork) → 낱자 라벨 전략 확정 (11.4)
- T2-M2: 자모 데이터 추출(짧은 시퀀스) + 소량 검증
- T2-M3: 지문자 분류 모델 학습(단일 화자 우선, 재현율 확인)
- T2-M4: 한글 조합 오토마타 구현 + 단위 테스트
- T2-M5: 실시간 연속 지화 → 글자 경계 판정 → 조합 → 화면/TTS
- T2-M6: Track 1(단어) + Track 2(지문자) 통합 — 아는 단어는 단어로,
  모르면 지화로. 모드 전환 또는 confidence 기반 라우팅.

### 11.8 통합 비전 (최종)

"핵심 어휘 단어 인식 + 지문자 기반 개방 어휘"의 2트랙 구조.
이어링크 연구계획서 서술: 단어 모델로 빈출 표현을 빠르게, 지문자로
미등록 어휘(이름·지명·신조어)를 커버 → 실사용 커버리지 확보.
