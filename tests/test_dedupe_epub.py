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
