import pytest

from book_translate._util import parse_numbered_lines


@pytest.mark.parametrize(
    "line",
    [
        "1. 译文",
        "1、译文",
        "1) 译文",
        "1．译文",
        "**1.** 译文",
        "*1.* 译文",
        "__1.__ 译文",
        "1: 译文",
        "1：译文",
        "- 1. 译文",
        "* 1. 译文",
        "• 1. 译文",
        "- **1.** 译文",
        "第1段：译文",
        "第 1 段：译文",
        "第1段. 译文",
    ],
)
def test_marker_variants_parse(line):
    assert parse_numbered_lines(line) == {1: "译文"}


def test_bold_and_colon_lists_parse_fully():
    assert parse_numbered_lines("**1.** 第一段\n**2.** 第二段") == {1: "第一段", 2: "第二段"}
    assert parse_numbered_lines("1: 第一段\n2: 第二段") == {1: "第一段", 2: "第二段"}
    assert parse_numbered_lines("第1段：第一段\n第2段：第二段") == {1: "第一段", 2: "第二段"}
    assert parse_numbered_lines("- 1. 第一段\n- 2. 第二段") == {1: "第一段", 2: "第二段"}


def test_multiline_answers_still_join():
    assert parse_numbered_lines("1. 第一行\n第二行") == {1: "第一行 第二行"}
    # a bare marker line starts the item; its text arrives on the next line
    assert parse_numbered_lines("1.\n译文正文") == {1: "译文正文"}


def test_embedded_sublist_stays_content():
    """A numbered list *inside* one translation must not become new items."""
    raw = "1. 第一段译文\n2. 步骤如下：\n1. 安装依赖\n2. 运行程序\n3. 检查输出"
    parsed = parse_numbered_lines(raw, expect=2)
    assert set(parsed) == {1, 2}
    assert parsed[1] == "第一段译文"
    assert "安装依赖" in parsed[2]
    assert "检查输出" in parsed[2]


def test_restarted_numbering_stays_content_without_expect():
    raw = "1. 甲\n2. 乙，要点：\n1. 要点一\n2. 要点二"
    parsed = parse_numbered_lines(raw)
    assert set(parsed) == {1, 2}
    assert "要点一" in parsed[2]
    assert "要点二" in parsed[2]


def test_expect_caps_item_index():
    assert parse_numbered_lines("5. 译文", expect=4) == {}
    assert parse_numbered_lines("1. 甲\n2. 乙", expect=2) == {1: "甲", 2: "乙"}


def test_prose_lookalikes_are_not_markers():
    # ratio: bare ":" needs trailing whitespace
    assert parse_numbered_lines("1:2 的比例很常见") == {}
    assert parse_numbered_lines("1. 比例是\n1:2 很常见") == {1: "比例是 1:2 很常见"}
    # years exceed the 3-digit index cap
    assert parse_numbered_lines("2023. 年度回顾") == {}
    # emphasis opener without the number pattern stays content
    assert parse_numbered_lines("1. 甲\n**注意**：请谨慎") == {1: "甲 **注意**：请谨慎"}


def test_bold_content_after_marker_is_not_stripped():
    # the closing-emphasis slot only fires when an opener wrapped the number
    assert parse_numbered_lines("1. **强调**的译文") == {1: "**强调**的译文"}
