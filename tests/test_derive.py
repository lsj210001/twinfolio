from book_translate.derive import _strip_english_html


def test_bilingual_pair_strips_english_keeps_chinese():
    html = (
        "<html><body>"
        "<p>This paragraph was translated properly by the machine.</p>"
        "<p>这一段是机器翻译好的中文对照内容。</p>"
        "</body></html>"
    )
    out, removed = _strip_english_html(html)
    assert removed == 1
    assert "translated properly" not in out
    assert "这一段是机器翻译好的中文对照内容" in out


def test_untranslated_english_is_preserved():
    """H2: English without a Chinese sibling must never be deleted."""
    html = (
        "<html><body>"
        "<p>This paragraph was never translated at all anywhere.</p>"
        "<blockquote>Another untranslated english block quote content.</blockquote>"
        "</body></html>"
    )
    out, removed = _strip_english_html(html)
    assert removed == 0
    assert "never translated at all" in out
    assert "untranslated english block quote" in out


def test_table_cells_and_headings_are_protected():
    """H2: td/th and h1-h6 are never decomposed, even pure-English ones."""
    html = (
        "<html><body>"
        "<h1>Untranslated Chapter Heading Title</h1>"
        "<h2>Another English Heading Here</h2>"
        "<p>这里是标题后面的中文正文内容。</p>"
        "<table><tr>"
        "<th>English header cell text</th>"
        "<td>English body cell text stays</td>"
        "<td>42</td>"
        "</tr></table>"
        "</body></html>"
    )
    out, removed = _strip_english_html(html)
    assert "Untranslated Chapter Heading Title" in out
    assert "Another English Heading Here" in out
    assert "English header cell text" in out
    assert "English body cell text stays" in out
    assert removed == 0


def test_untranslated_english_before_chinese_heading_is_kept():
    html = (
        "<html><body>"
        "<p>This last english paragraph was never translated at all anywhere.</p>"
        "<h2>第二章 训练方法与评估指标</h2>"
        "</body></html>"
    )
    out, removed = _strip_english_html(html)
    assert removed == 0
    assert "never translated at all" in out
    assert "第二章 训练方法与评估指标" in out
