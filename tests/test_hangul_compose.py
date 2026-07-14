"""한글 조합 오토마타 단위테스트 — 순수 로직 (TF/numpy 무의존).

자모 낱자 시퀀스 → 완성형 한글. 쌍자음·복합모음·겹받침·받침 이동을
검증한다 (CLAUDE.md §11.6.2). 기대 출력은 전부 유니코드 조합으로 확인됨.
"""

import pytest

from shared.hangul_compose import HangulComposer, compose


# ---------------------------------------------------------------- 참조 예시

@pytest.mark.parametrize("jamos,expected", [
    ("ㅎㅏㄱㄱㅛ", "학교"),   # 받침 뒤 자음+모음 → 쌍받침 잠정병합 후 분리
    ("ㄱㅏㅁㅅㅏ", "감사"),   # 받침 ㅁ + ㅅ(다음 초성) — ㅁㅅ 겹받침 아님
    ("ㄱㅏㅂㅅ", "값"),       # 겹받침 ㅄ (모음 없음 → 분리 안 함)
    ("ㅁㅏㄴㅕ", "마녀"),     # 받침 ㄴ 통째 이동: 만 → 마 + 녀
])
def test_reference_examples(jamos, expected):
    assert compose(jamos) == expected


# ---------------------------------------------------------------- 쌍자음

def test_double_lead_kkum():
    assert compose("ㄱㄱㅜㅁ") == "꿈"   # ㄱ+ㄱ → ㄲ (LEAD 위치)


def test_double_lead_ssang():
    assert compose("ㅅㅅㅏㅇ") == "쌍"   # ㅅ+ㅅ → ㅆ


# ---------------------------------------------------------------- 복합모음

def test_compound_vowel_gwa():
    assert compose("ㄱㅗㅏ") == "과"     # ㅗ+ㅏ → ㅘ


def test_compound_vowel_swi():
    assert compose("ㅅㅜㅣ") == "쉬"     # ㅜ+ㅣ → ㅟ


def test_compound_vowel_mwe():
    assert compose("ㅁㅜㅔ") == "뭬"     # ㅜ+ㅔ → ㅞ


# ---------------------------------------------------------------- 겹받침 (이동 없음)

def test_compound_tail_gaps():
    assert compose("ㄱㅏㅂㅅ") == "값"


def test_compound_tail_dalk():
    assert compose("ㄷㅏㄹㄱ") == "닭"   # ㄹ+ㄱ → ㄺ


# ---------------------------------------------------------------- 받침 이동

def test_tail_migration_single():
    assert compose("ㅁㅏㄴㅕ") == "마녀"   # 단일 받침 ㄴ 이동


def test_tail_migration_double():
    assert compose("ㅎㅏㄱㄱㅛ") == "학교"  # 쌍받침 잠정 ㄲ → split


def test_compound_tail_migration():
    assert compose("ㄱㅏㅂㅅㅣ") == "갑시"  # 겹받침 ㅄ split → 갑 + 시


def test_no_merge_consonant_defers_lead():
    # ㄹ 받침 + ㅇ(겹받침 불가) → 말 확정, ㅇ 대기 lead → +ㅣ → 이
    assert compose("ㅁㅏㄹㅇㅣ") == "말이"


def test_ieung_tail_no_migration():
    assert compose("ㅅㅏㄹㅏㅇ") == "사랑"  # ㅇ 받침은 이동 안 함 (모음 없음)


# ---------------------------------------------------------------- 라이브 렌더

def test_live_render_incremental():
    """웹 UI 계약(§11.6.3): ㅎ→하→학 단계별 라이브 렌더."""
    c = HangulComposer()
    c.feed("ㅎ")
    assert c.text == "ㅎ"      # lead만 → 원시 자모
    c.feed("ㅏ")
    assert c.text == "하"      # lead+vowel
    c.feed("ㄱ")
    assert c.text == "학"      # +tail


def test_live_render_migration_updates():
    """받침 이동이 라이브 텍스트에 반영된다: 마→만→마녀."""
    c = HangulComposer()
    for j in "ㅁㅏㄴ":
        c.feed(j)
    assert c.text == "만"
    c.feed("ㅕ")
    assert c.text == "마녀"


# ---------------------------------------------------------------- 엣지

def test_bare_vowel_start():
    assert compose("ㅏ") == "ㅏ"        # 무두 모음 → 원시 자모 (ㅇ 자동삽입 안 함)


def test_double_vowel_no_compound():
    assert compose("ㅏㅏ") == "ㅏㅏ"     # ㅏ+ㅏ 복합 아님 → 무두 모음 2개


def test_bare_compound_vowel():
    assert compose("ㅗㅏ") == "ㅘ"       # 무두라도 복합모음은 병합 (원시 렌더)


def test_lead_only_render():
    c = HangulComposer()
    c.feed("ㄱ")
    c.feed("ㄱ")
    assert c.text == "ㄲ"       # 모음 없는 쌍자음 → 원시 자모 렌더
    c.commit()
    assert c.text == "ㄲ"


def test_reset():
    c = HangulComposer()
    for j in "ㄱㅏ":
        c.feed(j)
    c.reset()
    assert c.text == ""
    assert compose("ㄴㅏ") == "나"  # 리셋 후 새 조합 정상


def test_commit_flushes_fragment():
    c = HangulComposer()
    c.feed("ㅎ")
    c.commit()
    assert c.text == "ㅎ"


def test_batch_accepts_list():
    assert compose(["ㄱ", "ㅏ"]) == "가"


def test_two_syllables_no_tail():
    assert compose("ㄴㅏㄴㅏ") == "나나"  # 받침 없는 연속 음절
