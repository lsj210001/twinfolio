from book_translate.dedupe_epub import collapse, similar


def test_benzhang_paragraphs_are_not_collapsed():
    """M4: two Chinese paragraphs that merely start with 本章 are real content."""
    html = (
        "<html><body>"
        "<p>本章介绍了扩散模型的基本原理和训练方法。</p>"
        "<p>本章小结</p>"
        "</body></html>"
    )
    out, removed = collapse(html)
    assert removed == 0
    assert "本章介绍了扩散模型的基本原理和训练方法" in out
    assert "本章小结" in out


def test_lunwen_prefix_paragraphs_are_not_collapsed():
    html = (
        "<html><body>"
        "<p>论文精选：视频生成方向的十篇代表作。</p>"
        "<p>论文阅读方法概述</p>"
        "</body></html>"
    )
    out, removed = collapse(html)
    assert removed == 0
    assert "论文精选" in out
    assert "论文阅读方法概述" in out


def test_same_language_shared_prefix_is_not_similar():
    """M4: the 18-char prefix rule requires an EN/ZH pair, not two ZH texts."""
    a = "本章介绍了扩散模型的原理与训练方法以及推理加速技巧"
    b = "本章介绍了扩散模型的原理与训练方法以及常用评测指标"
    assert not similar(a, b)


def test_same_language_substring_is_not_similar():
    short = "本章介绍了扩散模型的原理与训练方法。"
    long = "本章介绍了扩散模型的原理与训练方法以及推理加速技巧。"
    assert not similar(short, long)
    html = f"<html><body><p>{short}</p><p>{long}</p></body></html>"
    out, removed = collapse(html)
    assert removed == 0
    assert "推理加速技巧" in out
    assert out.count("本章介绍了扩散模型的原理与训练方法") == 2


def test_same_language_references_sharing_cite_number_are_not_collapsed():
    """Two different references that both start with [3] are real content."""
    zh_a = "[3] 张三：《扩散模型导论》，机械工业出版社，2023。"
    zh_b = "[3] 李四：《生成模型实践》，人民邮电出版社，2022。"
    assert not similar(zh_a, zh_b)
    en_a = "[3] Smith J. Diffusion Models Beat GANs. NeurIPS, 2021."
    en_b = "[3] Jones K. Score-Based Generative Modeling. ICLR, 2021."
    assert not similar(en_a, en_b)
    html = f"<html><body><p>{zh_a}</p><p>{zh_b}</p></body></html>"
    out, removed = collapse(html)
    assert removed == 0
    assert "张三" in out and "李四" in out


def test_cross_language_cite_pair_is_still_similar():
    en = "[3] Smith J. Diffusion Models Beat GANs. NeurIPS, 2021."
    zh = "[3] 史密斯。扩散模型胜过生成对抗网络。NeurIPS，2021。"
    assert similar(en, zh)


def test_same_language_figure_number_paragraphs_are_not_collapsed():
    """A caption and a body paragraph may both start with 图 3.1."""
    cap = "图 3.1 扩散模型的总体架构。"
    body = "图 3.1 展示了训练与推理两个阶段的完整流程。"
    assert not similar(cap, body)
    html = f"<html><body><p>{cap}</p><p>{body}</p></body></html>"
    out, removed = collapse(html)
    assert removed == 0
    assert "总体架构" in out and "完整流程" in out


def test_cross_language_figure_caption_pair_is_still_similar():
    en = "图 3.1 Overall architecture of the diffusion model"
    zh = "图 3.1 扩散模型的总体架构"
    assert similar(en, zh)


def test_volume_markers_in_parentheses_are_not_collapsed():
    """（上）/（下） volumes share the paren-stripped core but differ."""
    up = "扩散模型原理与实践（上）"
    down = "扩散模型原理与实践（下）"
    assert not similar(up, down)
    html = f"<html><body><p>{up}</p><p>{down}</p></body></html>"
    out, removed = collapse(html)
    assert removed == 0
    assert "（上）" in out and "（下）" in out


def test_identical_table_cells_are_never_removed():
    """Dropping a cell shifts every later column in its row."""
    html = (
        "<html><body><table>"
        "<tr><th>特性</th><th>特性</th></tr>"
        "<tr><td>支持中文</td><td>支持中文</td><td>不支持</td></tr>"
        "</table>"
        "<p>第3章 扩散模型</p>"
        "</body></html>"
    )
    out, removed = collapse(html)
    assert removed == 0
    assert out.count("特性") == 2
    assert out.count("支持中文") == 2


def test_cell_text_does_not_chain_into_following_paragraph():
    html = (
        "<html><body><table><tr><td>第3章 扩散模型</td></tr></table>"
        "<p>第3章 扩散模型</p></body></html>"
    )
    out, removed = collapse(html)
    assert removed == 0
    assert out.count("第3章 扩散模型") == 2


def test_removed_text_is_written_to_log():
    html = "<html><body><p>第3章 扩散模型</p><p>第3章 扩散模型</p></body></html>"
    lines: list[str] = []
    out, removed = collapse(html, log=lines.append)
    assert removed == 1
    assert any("第3章 扩散模型" in m for m in lines)


def test_swapping_near_duplicate_zh_variants_keeps_same_content():
    """Normalized equality is transitive, so the run outcome does not depend
    on which of the two near-duplicate variants comes first."""
    en = "Stable Diffusion WebUI Installation and Usage Guide"
    zh = "Stable Diffusion WebUI 安装与使用指南"
    zh2 = "Stable Diffusion WebUI 安装与使用指南。"
    for first, second in ((zh, zh2), (zh2, zh)):
        html = f"<html><body><p>{en}</p><p>{first}</p><p>{second}</p></body></html>"
        out, removed = collapse(html)
        assert removed == 1
        assert "Installation and Usage Guide" in out
        assert out.count("安装与使用指南") == 1
        # prefer() keeps the more complete variant in either order
        assert "安装与使用指南。" in out


def test_real_duplicates_still_collapse():
    """Exact adjacent duplicates and EN+ZH prefix pairs still fold."""
    # exact Chinese duplicate: keep one
    html = "<html><body><p>第3章 扩散模型</p><p>第3章 扩散模型</p></body></html>"
    out, removed = collapse(html)
    assert removed == 1
    assert out.count("第3章 扩散模型") == 1

    # EN + ZH + near-duplicate ZH: keep the English and one Chinese
    html2 = (
        "<html><body>"
        "<p>Stable Diffusion WebUI Installation and Usage Guide</p>"
        "<p>Stable Diffusion WebUI 安装与使用指南</p>"
        "<p>Stable Diffusion WebUI 安装与使用指南。</p>"
        "</body></html>"
    )
    out2, removed2 = collapse(html2)
    assert removed2 == 1
    assert "Installation and Usage Guide" in out2
    assert out2.count("安装与使用指南") == 1
