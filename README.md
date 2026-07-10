# 이어링크 (EarLink)

농인이 브라우저 카메라 앞에서 수어를 하면 실시간으로 한국어 텍스트로 변환해
표시하고 TTS로 읽어주는 웹앱 **연구 시제품**입니다.

키포인트 추출(MediaPipe Tasks, WASM)은 **브라우저에서만** 수행하며, 서버로는
키포인트 JSON만 전송합니다. **영상 프레임은 절대 서버로 전송하지 않습니다**
(개인정보·대역폭 설계 원칙).

```
[브라우저]  카메라 → MediaPipe 키포인트 → 캔버스 시각화
                └─ 키포인트 JSON ─▶ [FastAPI]  /ws/translate (M1: 에코)
```

## 요구사항

- macOS (Apple Silicon 기준), Python 3.11+
- 모던 브라우저 (Chrome / Safari / Edge)

## 설치

```bash
python3 -m venv .venv          # 3.11+ 인터프리터로 생성 (예: python3.12 -m venv .venv)
source .venv/bin/activate
pip install -r requirements.txt
```

## 로컬 실행 (맥 웹캠)

```bash
source .venv/bin/activate
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

브라우저에서 <http://localhost:8000> 접속 → 카메라 허용.

- 포즈(초록)·왼손(파랑)·오른손(주황)·얼굴 윤곽(분홍) 스켈레톤이 실시간으로 그려집니다.
- 우측 상단 HUD: WebSocket 연결 상태, 서버 에코 프레임 카운터(frames), RTT, 포인트 수, 랜드마커 모드.
- 카메라는 HTTPS에서만 열리지만 **localhost는 브라우저 예외**라 그대로 동작합니다.

`python -m server.main`으로 실행하면 `config.yaml`의 host/port를 사용합니다.

## 스마트폰 테스트 (HTTPS 필수)

모바일 브라우저는 HTTPS에서만 카메라를 허용합니다. 두 가지 방법 중 하나를 사용하세요.

### 방법 1 — cloudflared 퀵 터널 (권장)

```bash
brew install cloudflared

# 서버를 먼저 띄운 뒤, 별도 터미널에서:
cloudflared tunnel --url http://localhost:8000
```

출력되는 `https://…trycloudflare.com` URL을 스마트폰 브라우저로 접속합니다.

- WebSocket(wss)도 터널을 그대로 통과하므로 추가 설정이 없습니다.
- URL은 터널을 실행할 때마다 바뀝니다 (계정·설정 불필요한 임시 터널).
- 화면 하단 `카메라 전환` 버튼으로 전/후면 카메라를 전환할 수 있습니다.

### 방법 2 — mkcert 로컬 인증서 (같은 Wi-Fi)

```bash
brew install mkcert
mkcert -install
mkcert <맥의_내부IP>        # 예: mkcert 192.168.0.10

uvicorn server.main:app --host 0.0.0.0 --port 8000 \
  --ssl-keyfile <내부IP>-key.pem --ssl-certfile <내부IP>.pem
```

같은 Wi-Fi의 스마트폰에서 `https://<맥의_내부IP>:8000` 접속.
(스마트폰에 mkcert 루트 인증서를 설치해야 경고 없이 열립니다.)

## 개인정보 원칙

- 서버로 전송되는 것: 키포인트 좌표 JSON(`pose`, `left_hand`, `right_hand`, `face`)뿐.
- 영상 프레임(이미지/blob)을 전송·저장하는 코드는 작성하지 않습니다.

## 마일스톤 현황

| 단계 | 내용 | 상태 |
|------|------|------|
| M1 | 스캐폴드: 정적 서빙 + WS 에코 + 키포인트 시각화 | ✅ |
| M2 | 특징 파이프라인 (`shared/features.py` ↔ `web/features.js` 일치 검증) | ⏳ |
| M3 | 데이터 전처리 (AI Hub KSL-Guide, cp949 CSV) | ⏳ |
| M4 | Conv1D-LSTM 학습 (val 정확도 ≥ 80%) | ⏳ |
| M5 | 실시간 연결 (발화 탐지 + 추론 + TTS) | ⏳ |
| M6 | 마감 (README, 데모 스크립트) | ⏳ |
