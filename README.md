# 이어링크 (EarLink)

농인이 브라우저 카메라 앞에서 한글 지문자(손으로 자모)를 하면, 자모를 실시간
인식·조합해 완성형 한글로 표시하고 TTS로 읽어주는 웹앱 **연구 시제품**입니다.
자모를 이어붙여 학습하지 않은 임의의 단어를 표현합니다.

손 키포인트 추출(MediaPipe Tasks, WASM)은 **브라우저에서만** 수행하며, 서버로는
손 키포인트 JSON만 전송합니다. **영상 프레임은 절대 서버로 전송하지 않습니다**
(개인정보·대역폭 설계 원칙).

```
[브라우저]  카메라 → MediaPipe 손 21점 → 캔버스 시각화
                └─ 손 좌표 JSON ─▶ [FastAPI]  /ws/translate
                                    tflite 자모 분류 → 디바운스 → 한글 조합
   조합 한글 표시 + TTS ◀─ 자모 ─┘
```

## 요구사항

- macOS (Apple Silicon 기준), Python 3.11+
- 모던 브라우저 (Chrome / Safari / Edge)

## 설치

```bash
python3 -m venv .venv          # 3.11+ 인터프리터 (예: python3.12 -m venv .venv)
source .venv/bin/activate
pip install -r requirements.txt
```

## 로컬 실행 (맥 웹캠)

```bash
source .venv/bin/activate
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

브라우저에서 <http://localhost:8000> 접속 → 카메라 허용.

- 손 스켈레톤이 실시간으로 그려집니다.
- 우측 상단 HUD: WebSocket 연결 상태, 프레임 카운터, RTT, 랜드마커 모드.
- 카메라는 HTTPS에서만 열리지만 **localhost는 브라우저 예외**라 그대로 동작합니다.

`python -m server.main`으로 실행하면 `config.yaml`의 host/port를 사용합니다.

## 사용법

1. **오른손**으로 카메라 앞에서 지문자(자모)를 만듭니다.
2. 같은 손모양을 잠깐(~0.6초) 유지하면 자모가 확정되고, 중앙 상단 패널에 조합
   중인 한글이 실시간으로 표시됩니다 (ㅎ → 하 → 학).
3. 완성된 음절은 한국어 TTS로 읽어줍니다. 우하단 🔊 버튼으로 켜고 끕니다.
   **iPhone은 첫 화면 터치 후** TTS가 활성화됩니다 (iOS 정책).
4. **연속 같은 자모**(예: "학교"의 ㄱㄱ)는 사이에 손을 잠깐 내렸다 올리면 재확정됩니다.
5. 확정 대기 시간·민감도는 `config.yaml`의 `fingerspelling` 값으로 조정합니다.

> **주의**: 현재 tflite 모델은 근접 웹캠 소수 인원 데이터로 학습돼, 학습 분포와
> 다른 환경에서는 정확도가 낮을 수 있습니다. 재학습 시 `config.yaml`의
> `model_path`와 `server/fingerspelling.ACTIONS`만 교체하면 됩니다.

## 스마트폰 테스트 (HTTPS 필수)

모바일 브라우저는 HTTPS에서만 카메라를 허용합니다. 두 방법 중 하나를 쓰세요.

### 방법 1 — cloudflared 퀵 터널 (권장)

```bash
brew install cloudflared

# 서버를 먼저 띄운 뒤, 별도 터미널에서:
cloudflared tunnel --url http://localhost:8000
```

출력되는 `https://…trycloudflare.com` URL을 스마트폰 브라우저로 접속합니다.
WebSocket(wss)도 터널을 그대로 통과합니다.

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

- 서버로 전송되는 것: 손 키포인트 좌표 JSON뿐.
- 영상 프레임(이미지/blob)을 전송·저장하는 코드는 작성하지 않습니다.
