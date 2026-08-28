"""Characterization tests pinning `cn_en_counts` semantics across the regex
rewrite, plus normalization equivalence for `_util.norm` and
`dedupe_epub._norm`/`_core`.
"""
from __future__ import annotations

import pytest

from book_translate._util import cn_en_counts, norm
from book_translate.dedupe_epub import _core, _norm


def _reference_cn_en_counts(text: str) -> tuple[int, int]:
    """The original per-character implementation, kept as the oracle."""
    cn = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    en = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    return cn, en


# (sample, expected (cn, en)) — expected values were produced by the original
# implementation before the regex rewrite; they must never change.
PINNED = [
    ("", (0, 0)),
    ("Hello 世界 world 你好", (4, 10)),
    ("这是一段纯中文文本，包含标点。", (13, 0)),
    ("The quick brown fox jumps over the lazy dog", (0, 35)),
    ("abc123 !@# 你好42 DEF-ghi", (2, 9)),
    ("@@BT12@@ 前端 Hello @@BT3@@ world", (2, 14)),
    # accented latin lowers outside a-z: not counted as en
    ("café naïve résumé", (0, 11)),
    # U+0130 İ and U+212A KELVIN SIGN lower() into a-z and were counted by
    # the original implementation; ß and the ﬁ ligature were not
    ("İ\u212a straße ﬁ", (0, 7)),
    # CJK range boundaries: U+4E00/U+9FFF inside, U+4DFF/U+A000 outside
    ("\u4e00\u9fff\u4dff\ua000", (2, 0)),
    ("第 3 章  Introduction to Networks（网络导论）", (6, 22)),
    ("  多行\n文本\tmixed   WS  ", (4, 7)),
    # fullwidth digits/letters are neither cn nor en
    ("數字123與ＡＢＣ全形字母", (7, 0)),
]


@pytest.mark.parametrize("text,expected", PINNED)
def test_cn_en_counts_pinned_values(text: str, expected: tuple[int, int]) -> None:
    assert cn_en_counts(text) == expected


@pytest.mark.parametrize("text,_expected", PINNED)
def test_cn_en_counts_matches_reference(text: str, _expected: tuple[int, int]) -> None:
    assert cn_en_counts(text) == _reference_cn_en_counts(text)


def test_cn_en_counts_matches_reference_full_unicode() -> None:
    """Every BMP+ codepoint classifies identically to the old implementation."""
    import sys

    text = "".join(chr(cp) for cp in range(sys.maxunicode + 1) if chr(cp).isprintable())
    assert cn_en_counts(text) == _reference_cn_en_counts(text)


NORM_SAMPLES = [
    "",
    "  a\u00a0b\u200bc  \n d ",
    "“curly” and ‘single’ quotes",
    "第 12 章  作者：某某（注释）(note) ，标点。",
    "  多  空白\t与\n换行  ",
    "@@BT7@@ Mixed 中文 text",
]


@pytest.mark.parametrize("text", NORM_SAMPLES)
def test_norm_unchanged(text: str) -> None:
    """`_util.norm` results are identical to the pre-compiled-regex original."""
    import re

    def reference(s: str) -> str:
        s = (s or "").replace("\xa0", " ").replace("\u200b", "")
        s = re.sub(r"\s+", " ", s).strip()
        return s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')

    assert norm(text) == reference(text)


@pytest.mark.parametrize("text", NORM_SAMPLES)
def test_dedupe_norm_core_unchanged(text: str) -> None:
    """`dedupe_epub._norm`/`_core` results match the original inline-re.sub code."""
    import re

    def ref_norm(t: str) -> str:
        t = t.replace("\xa0", " ")
        t = re.sub(r"\s+", " ", t).strip()
        t = re.sub(r"第\s*(\d+)\s*章", r"第\1章", t)
        t = t.replace("作者：", "").replace("作者:", "")
        t = re.sub(r"[“”\"'《》·，,。．.：:；;、()\s]", "", t)
        return t.lower()

    def ref_core(t: str) -> str:
        t = re.sub(r"\s+", " ", t).strip()
        t = re.sub(r"（[^）]*）", "", t)
        t = re.sub(r"\([^)]*\)", "", t)
        return ref_norm(t)

    assert _norm(text) == ref_norm(text)
    assert _core(text) == ref_core(text)
