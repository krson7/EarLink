"""한글 조합 오토마타 — 자모 낱자 시퀀스를 완성형 한글로 조합 (Track 2, §11.6.2).

지문자 모델은 기본 자모(~40개)만 분류한다. 쌍자음(ㄱ+ㄱ→ㄲ), 복합모음
(ㅗ+ㅏ→ㅘ), 겹받침(ㅂ+ㅅ→ㅄ), 받침 이동은 이 오토마타가 스스로 처리한다.

입력은 Hangul Compatibility Jamo (U+3131~U+3163) — IME·분류기 라벨맵이 쓰는
코드포인트다 (U+1100 Jamo 블록 아님).

받침 이동(핵심 난제): 같은 자음쌍도 위치에 따라 다르게 동작한다.
  - LEAD 위치: ㄱ+ㄱ → ㄲ (쌍자음 초성)
  - TAIL 위치: ㄱ+ㄱ → 잠정 ㄲ 받침 → 다음 모음이 오면 분리(학 + 교)
룩어헤드 없이 "지금 병합, 다음 모음에서 split"으로 처리한다(표준 IME 동작).
예: ㅎㅏㄱㄱㅛ → 학교, ㄱㅏㅂㅅ → 값, ㄱㅏㅂㅅㅣ → 갑시, ㅁㅏㄴㅕ → 마녀.

오토마타는 총함수다 — 어떤 자모 스트림에도 예외를 던지지 않는다(라이브 UI가
잡음 시퀀스를 흘려도 안전). 무두(無頭) 모음은 ㅇ을 자동 삽입하지 않고 원시
자모로 렌더한다.
"""

# 유니코드 조합: 음절 = 0xAC00 + (초성*21 + 중성)*28 + 종성
SBASE = 0xAC00
NUM_JUNG = 21
NUM_JONG = 28

# 초성 19 (쌍자음 ㄸㅃㅉ 포함)
CHOSEONG = ("ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
            "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")
# 중성 21
JUNGSEONG = ("ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
             "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ")
# 종성 28 (인덱스 0 = 받침 없음; ㄸㅃㅉ 없고 ㄳㄵㄶㄺ… 있음 — 초성과 순서 다름)
JONGSEONG = ("", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
             "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
             "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")

LEAD_INDEX = {j: i for i, j in enumerate(CHOSEONG)}
VOWEL_INDEX = {j: i for i, j in enumerate(JUNGSEONG)}
TAIL_INDEX = {j: i for i, j in enumerate(JONGSEONG) if j}  # 빈 문자열 제외

# 쌍자음 (LEAD 위치, 모음 없을 때)
DOUBLE_LEAD = {
    ("ㄱ", "ㄱ"): "ㄲ", ("ㄷ", "ㄷ"): "ㄸ", ("ㅂ", "ㅂ"): "ㅃ",
    ("ㅅ", "ㅅ"): "ㅆ", ("ㅈ", "ㅈ"): "ㅉ",
}
# 복합모음 (VOWEL 위치)
COMPOUND_VOWEL = {
    ("ㅗ", "ㅏ"): "ㅘ", ("ㅗ", "ㅐ"): "ㅙ", ("ㅗ", "ㅣ"): "ㅚ",
    ("ㅜ", "ㅓ"): "ㅝ", ("ㅜ", "ㅔ"): "ㅞ", ("ㅜ", "ㅣ"): "ㅟ",
    ("ㅡ", "ㅣ"): "ㅢ",
}
# 겹받침 + 쌍받침 (TAIL 위치). ㄲ·ㅆ은 유효 종성이라 포함 — 학교 케이스의 핵심.
COMPOUND_TAIL = {
    ("ㄱ", "ㅅ"): "ㄳ", ("ㄴ", "ㅈ"): "ㄵ", ("ㄴ", "ㅎ"): "ㄶ",
    ("ㄹ", "ㄱ"): "ㄺ", ("ㄹ", "ㅁ"): "ㄻ", ("ㄹ", "ㅂ"): "ㄼ",
    ("ㄹ", "ㅅ"): "ㄽ", ("ㄹ", "ㅌ"): "ㄾ", ("ㄹ", "ㅍ"): "ㄿ",
    ("ㄹ", "ㅎ"): "ㅀ", ("ㅂ", "ㅅ"): "ㅄ",
    ("ㄱ", "ㄱ"): "ㄲ", ("ㅅ", "ㅅ"): "ㅆ",
}
# 받침 이동용 분리 맵 (드리프트 방지 위해 COMPOUND_TAIL 역산):
# 복합종성 → (남는 종성, 다음 글자 초성으로 이동할 자음)
TAIL_SPLIT = {v: k for k, v in COMPOUND_TAIL.items()}

# 임포트 시 자기검증 — 인덱스·맵 정합성 (features.py 스타일)
assert len(CHOSEONG) == 19 and len(JUNGSEONG) == 21 and len(JONGSEONG) == 28
assert all(v in LEAD_INDEX for v in DOUBLE_LEAD.values())
assert all(v in VOWEL_INDEX for v in COMPOUND_VOWEL.values())
assert all(v in TAIL_INDEX for v in COMPOUND_TAIL.values())
assert all(stay in TAIL_INDEX and migrate in LEAD_INDEX
           for stay, migrate in TAIL_SPLIT.values())


def _compose_syllable(lead, vowel, tail):
    """초성·중성(·종성) → 완성형 음절 문자."""
    jong = TAIL_INDEX[tail] if tail else 0
    code = SBASE + (LEAD_INDEX[lead] * NUM_JUNG + VOWEL_INDEX[vowel]) * NUM_JONG + jong
    return chr(code)


class HangulComposer:
    """자모 낱자를 하나씩 받아 완성형 한글로 조합하는 증분 오토마타.

    상태는 lead/vowel/tail 필드(자모 문자)로 표현된다 — 값이 어디까지
    채워졌는지가 곧 상태(EMPTY→LEAD→VOWEL→TAIL)다.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._committed = []   # 확정된 글자·조각
        self._lead = None
        self._vowel = None
        self._tail = None

    # ------------------------------------------------ 내부 헬퍼

    def _flush(self):
        """진행 중 음절/조각을 확정 목록으로 옮긴다."""
        if self._lead is not None:
            if self._vowel is not None:
                self._committed.append(_compose_syllable(
                    self._lead, self._vowel, self._tail))
            else:
                self._committed.append(self._lead)  # lead만 → 원시 자모
        elif self._vowel is not None:
            self._committed.append(self._vowel)     # 무두 모음 → 원시 자모
        self._lead = self._vowel = self._tail = None

    def _render_current(self):
        """진행 중 음절의 라이브 렌더 (비파괴). lead만이면 원시 자모."""
        if self._lead is not None:
            if self._vowel is not None:
                return _compose_syllable(self._lead, self._vowel, self._tail)
            return self._lead
        if self._vowel is not None:
            return self._vowel
        return ""

    # ------------------------------------------------ 공개 API

    def feed(self, jamo):
        """자모 낱자 1개를 오토마타에 투입 (Hangul Compatibility Jamo)."""
        is_vowel = jamo in VOWEL_INDEX
        is_lead = jamo in LEAD_INDEX
        is_tail = jamo in TAIL_INDEX

        # 미지 자모 — 진행 중 조각 플러시 후 원시 통과 (방어적)
        if not (is_vowel or is_lead or is_tail):
            self._flush()
            self._committed.append(jamo)
            return

        # ── EMPTY
        if self._lead is None and self._vowel is None:
            if is_vowel:
                self._vowel = jamo   # 무두 모음 (렌더는 원시 자모)
            else:
                self._lead = jamo
            return

        # ── 무두 모음 상태 (lead 없이 vowel만) — 복합모음 병합만 시도
        if self._lead is None:
            if is_vowel and (self._vowel, jamo) in COMPOUND_VOWEL:
                self._vowel = COMPOUND_VOWEL[(self._vowel, jamo)]
            else:
                self._flush()
                self.feed(jamo)  # 새 조각으로 재처리
            return

        # ── LEAD (lead 있고 vowel 없음)
        if self._vowel is None:
            if is_vowel:
                self._vowel = jamo
            elif (self._lead, jamo) in DOUBLE_LEAD:
                self._lead = DOUBLE_LEAD[(self._lead, jamo)]  # 쌍자음
            else:
                self._flush()        # orphan lead → 원시 자모 확정
                self._lead = jamo    # 새 lead 시작
            return

        # ── VOWEL (lead+vowel, tail 없음)
        if self._tail is None:
            if is_vowel:
                if (self._vowel, jamo) in COMPOUND_VOWEL:
                    self._vowel = COMPOUND_VOWEL[(self._vowel, jamo)]
                else:
                    self._flush()   # 병합 불가 모음 → 현재 음절 확정
                    self.feed(jamo)  # 무두 모음으로 재처리
            elif is_tail:
                self._tail = jamo
            else:
                # 종성 불가 자음(ㄸㅃㅉ) → 음절 확정 + 새 lead
                self._flush()
                self._lead = jamo
            return

        # ── TAIL (lead+vowel+tail)
        if is_vowel:
            # 받침 이동: 종성이 다음 글자 초성이었다
            if self._tail in TAIL_SPLIT:
                stay, migrate = TAIL_SPLIT[self._tail]
                self._tail = stay
                self._flush()          # 남는 받침까지 포함해 확정
                self._lead, self._vowel = migrate, jamo
            else:
                migrate = self._tail   # 단일 받침 통째 이동
                self._tail = None
                self._flush()
                self._lead, self._vowel = migrate, jamo
        elif (self._tail, jamo) in COMPOUND_TAIL:
            self._tail = COMPOUND_TAIL[(self._tail, jamo)]  # 잠정 겹/쌍받침
        else:
            # 병합 불가 자음 + 모음 없음 → 음절 확정, 자음은 대기 lead
            self._flush()
            self._lead = jamo

    def commit(self):
        """입력 종료 — 진행 중 조각을 확정한다."""
        self._flush()

    @property
    def text(self):
        """확정 텍스트 + 진행 중 음절의 라이브 렌더 (비파괴)."""
        return "".join(self._committed) + self._render_current()


def compose(jamos):
    """자모 시퀀스(str 또는 iterable) → 완성형 한글 문자열 (배치 편의 함수)."""
    composer = HangulComposer()
    for jamo in jamos:
        composer.feed(jamo)
    composer.commit()
    return composer.text
