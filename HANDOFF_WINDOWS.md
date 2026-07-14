# 윈도우 인수인계 (HANDOFF) — 이어링크(EarLink)

이 문서는 macOS에서 진행하던 작업을 **윈도우에서 이어받는 Claude**를 위한
인계 노트다. 프로젝트의 단일 기준 문서는 여전히 [CLAUDE.md](CLAUDE.md)이며,
이 문서는 환경 차이와 시작 방법만 안내한다.

## 1. 현재 진행 상황 (2026-07-10)

- **Track 1 (단어 인식) — M1~M5 완료.** 카메라 → MediaPipe 키포인트 →
  특징 벡터 → 발화 탐지 → Conv1D-LSTM 추론 → 화면 표시 + 한국어 TTS의
  전 구간 연결. **원본 영상 종단 재현율 96.7%** (`ksl_word_v2.keras`).
  - B안(REAL01 단일 화자)이라 과적합 상태 — 일반 사용자 수어는 잘 안 맞는
    게 정상. 다화자(A안) 재학습은 보류 상태.
- **Track 2 (지문자·지숫자) — 착수.** 데이터 다운로드 중.
  `shared/hangul_compose.py`(자모→완성형 한글 조합 오토마타) 완료·테스트 통과.
  다음은 CLAUDE.md §11.7 T2-M1(FINSP/CROWD 데이터 확인 → 낱자 라벨 전략).

**이 브랜치에 포함된 것**: 전체 소스 + 학습된 모델(`models/ksl_word_v1/v2.keras`
+ labels + history). **제외된 것**(재생성 가능): 전처리 npy(`data/`),
MediaPipe 모델 캐시(`models/mediapipe/` — 첫 실행 시 자동 다운로드),
학습 곡선·혼동행렬 png.

## 2. 윈도우 환경 세팅

**Python 3.11+ 필요** (원 개발은 3.12). PowerShell 기준:

```powershell
cd path\to\EarLink
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- `requirements.txt`의 `tensorflow-metal`은 **주석 처리된 macOS 전용 선택
  항목**이라 윈도우 설치에 영향 없다. `tensorflow==2.21.0`·
  `mediapipe==0.10.35`는 윈도우 파이썬 3.11/3.12 휠이 있다.
- 코드 경로는 전부 `pathlib`이라 경로 구분자(`\` vs `/`)는 신경 쓸 필요 없다.

## 3. 바로 되는 작업 (원본 영상 불필요 — 모델 포함됨)

**서버 + 실시간 추론** (v2 모델 자동 로드):
```powershell
.venv\Scripts\python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```
기동 로그에 `[모델] ksl_word_v2 로드 …`가 뜨면 준비 완료. 브라우저에서
**http://localhost:8000** 접속(카메라는 localhost·HTTPS에서만 열림 —
`0.0.0.0` 주소로 열면 안 됨). 첫 접속 시 `models/mediapipe/`가 없으면
프런트가 CDN에서, 서버는 필요 시 자동 다운로드한다.

**전체 테스트** (52개):
```powershell
.venv\Scripts\python -m pytest tests\ -v
```
- `test_features_parity.py`는 **Node.js 필요**(Python↔JS 특징 일치 검증).
  Node가 없으면 그 테스트만 실패하니, 없으면 `--ignore=tests\test_features_parity.py`.
  나머지(발화 탐지·한글 조합·WS 스모크)는 Node 불필요.

## 4. 원본 영상이 필요한 작업 (데이터셋 별도 복사)

재추출·재학습·종단평가는 AI Hub 원본 영상이 있어야 한다. macOS의
`config.yaml` `data.data_root`는 **macOS 절대경로**(`/Users/ayaan/Desktop/data`)로
되어 있으니, 윈도우에선 아래 중 하나로 바꾼다:
- `config.yaml`의 `data_root`를 윈도우 경로로 수정, 또는
- 실행 시 `--data-root "D:\path\to\data"`, 또는 `$env:DATA_ROOT="D:\..."`.

데이터 루트 하위에 `원천데이터\` 트리가 있어야 스캔된다(라벨링데이터는
읽지 않음). 관련 스크립트: `training\build_dataset.py`,
`training\extract_keypoints.py`, `training\train.py`,
`training\evaluate_e2e.py` (전부 `python -m training.<name>`로 실행).

## 5. 다음 작업 (Track 2)

CLAUDE.md **§11**이 지문자 트랙 전체 명세다. 마일스톤 §11.7:
- **T2-M1**: FINSP/CROWD 데이터 확인 → 낱자 라벨 전략 확정(§11.4 옵션 A/B)
- T2-M2~M6: 자모 추출 → 지문자 모델 학습 → 한글 조합(오토마타 준비됨) →
  실시간 연속 지화 → Track 1 통합

`shared/hangul_compose.py`는 `compose("ㅎㅏㄱㄱㅛ") == "학교"` 식으로 이미
동작하며 `tests/test_hangul_compose.py`로 검증돼 있다(쌍자음·복합모음·
겹받침·받침 이동 포함).

## 6. 작업 규칙 (CLAUDE.md 준수)

- **기준 문서는 CLAUDE.md** — 구현 결정이 충돌하면 CLAUDE.md가 우선.
- 특징 로직은 `shared/features.py` ↔ `web/features.js` **완전 동일**(패리티
  테스트가 강제). 윈도잉은 `shared/windowing.py` 학습·추론 공용.
- 커밋은 **한국어 메시지·마일스톤 단위**. **Claude 공동작업자(Co-Authored-By)
  표시는 넣지 않는다.**
- 프레임 이미지 서버 전송 금지 — 키포인트/특징 JSON만 (개인정보 원칙).
- `claude_md_update_지문자트랙.md`는 Track 2 도입 지시서(반영 완료). 한글
  파일명이지만 윈도우에서 정상 동작한다.
