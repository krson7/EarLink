# Rahites 지문자 모델 체험 데모

[Rahites/Sign_Language_Translation](https://github.com/Rahites/Sign_Language_Translation)
의 학습된 지문자(한글 자모) 인식 모델을 **웹캠/영상으로 실시간 체험**하는 독립 데모.
이어링크 프로젝트와 분리돼 있으며, 자모→완성형 조합에만 `hangul_compose.py`(프로젝트에서
복사)를 사용한다.

## 구성
- `demo.py` — 메인
- `models/multi_hand_gesture_classifier.tflite` — Rahites 학습 가중치(31 자모)
- `hand_landmarker.task` — MediaPipe 손 랜드마커(없으면 첫 실행 시 자동 다운로드)
- `hangul_compose.py` — 자모 조합 오토마타
- `sample_garakro.mp4` — 테스트용 AI Hub 영상(정답 "가락로")

## 실행

이어링크 `.venv`에 의존성이 이미 있으므로 그 파이썬으로 실행하는 게 가장 간단하다:

```powershell
$PY = "C:\Users\IoTLab\Desktop\EarLink\.venv\Scripts\python.exe"

# 웹캠 실시간
& $PY demo.py --webcam

# AI Hub 영상 재생하며 인식 (정답 병기)
& $PY demo.py --video sample_garakro.mp4 --truth "가락로"
```

별도 환경이면: `pip install -r requirements.txt` 후 `python demo.py ...`.

## 화면
좌상단에 **현재 인식 자모 + confidence**(크게), 그 아래 **확정 자모 스트림**과
**조합된 한글**(예: `ㄱㅏㄹㅏㄱ` → `가락`)이 병기된다. 영상 모드에서 `--truth`를 주면
하단에 정답이 표시된다. 한글은 PIL(맑은고딕/AppleGothic)로 렌더한다.

## 키
- `c` — 확정 스트림·조합 리셋
- `q` — 종료
- `space` — 일시정지(영상 모드)

## 옵션
| 인자 | 기본 | 설명 |
|---|---|---|
| `--confirm-frames` | 8 | 같은 자모 연속 N프레임 유지 시 1회 확정(디바운스) |
| `--conf` | 0.8 | 확정 confidence 임곗값 |
| `--flip` | 웹캠 on/영상 off | 좌우반전(오른손 매칭용) |
| `--max-height` | 900 | 표시 리사이즈 최대 높이 |
| `--no-window --save-dir DIR` | — | 헤드리스: 주석 프레임을 DIR에 저장(스크린샷용) |

## 인식 정확도 주의
이 모델은 **국민대 3인 근접 웹캠**으로 학습됐다. AI Hub 크라우드 영상(원거리·세로·다인)
에서는 **실측 정확도 ~5%**(거의 무작위, 검지계열 ㄱ/ㅏ/ㅗ/ㅣ가 ㅜ로 붕괴)로 낮다.
근접 웹캠으로 또박또박 지화하면 더 잘 인식된다. 데모는 "동작 체험"이 목적이다.
전처리는 그들의 `Vector_Normalization`(뼈 단위벡터 20 + 각도 15 = 55, 시퀀스 10)을 그대로 재현했다.
