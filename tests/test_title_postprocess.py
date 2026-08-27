import zipfile
from pathlib import Path

from bs4 import BeautifulSoup

from book_translate.title_postprocess import (
    _apply_mapping_zip,
    _collect_titles_from_epub,
    _force_replace_element,
)


def test_head_title_collected_and_replaced_in_place(tmp_path: Path):
    """A leftover English <head><title> is picked up and replaced in place,
    keeping exactly one <title> (file-resume must no longer duplicate it)."""
    epub = tmp_path / "book.epub"
    chapter = (
        "<html><head><title>My Great Book Chapter</title></head>"
        "<body><p>正文段落已经翻译好了。</p></body></html>"
    )
    with zipfile.ZipFile(epub, "w") as z:
        z.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        z.writestr("EPUB/Text/ch1.xhtml", chapter)

    assert "My Great Book Chapter" in _collect_titles_from_epub(epub)

    out = tmp_path / "out.epub"
    n = _apply_mapping_zip(epub, {"My Great Book Chapter": "我的好书章节"}, out)
    assert n == 1

    with zipfile.ZipFile(out) as z:
        got = z.read("EPUB/Text/ch1.xhtml").decode("utf-8")
    soup = BeautifulSoup(got, "html.parser")
    titles = soup.find_all("title")
    assert len(titles) == 1
    assert titles[0].get_text() == "我的好书章节"


def test_toc_li_replacement_keeps_link(tmp_path: Path):
    """H3: replacing a matched <li> on a visible TOC page must keep <a href>."""
    epub = tmp_path / "book.epub"
    toc_html = (
        "<html><body><ul>"
        '<li><a href="Text/chapter-1.xhtml">Chapter 1. Getting Started</a></li>'
        '<li><a href="Text/chapter-2.xhtml">Chapter 2. Advanced Topics</a></li>'
        "</ul></body></html>"
    )
    with zipfile.ZipFile(epub, "w") as z:
        z.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        z.writestr("EPUB/toc.xhtml", toc_html)

    mapping = {
        "Chapter 1. Getting Started": "第1章 入门",
        "Chapter 2. Advanced Topics": "第2章 进阶主题",
    }
    out = tmp_path / "out.epub"
    n = _apply_mapping_zip(epub, mapping, out)
    assert n >= 2

    with zipfile.ZipFile(out) as z:
        got = z.read("EPUB/toc.xhtml").decode("utf-8")
    soup = BeautifulSoup(got, "html.parser")
    links = soup.find_all("a")
    assert [a.get("href") for a in links] == ["Text/chapter-1.xhtml", "Text/chapter-2.xhtml"]
    assert [a.get_text() for a in links] == ["第1章 入门", "第2章 进阶主题"]


def test_force_replace_without_link_still_replaces():
    soup = BeautifulSoup("<h1>Chapter 1. Getting Started</h1>", "html.parser")
    el = soup.find("h1")
    mapping = {"Chapter 1. Getting Started": "第1章 入门"}
    nmap = dict(mapping)
    assert _force_replace_element(el, mapping, nmap) == 1
    assert el.get_text() == "第1章 入门"


def test_force_replace_li_with_extra_text_keeps_link():
    """A matched <li> whose anchor text is only part of the title must not
    have its structure destroyed."""
    soup = BeautifulSoup(
        '<li>Appendix <a href="app.xhtml">Extra Materials Online</a></li>', "html.parser"
    )
    el = soup.find("li")
    mapping = {
        "Appendix Extra Materials Online": "附录 线上补充材料",
        "Extra Materials Online": "线上补充材料",
    }
    nmap = dict(mapping)
    assert _force_replace_element(el, mapping, nmap) == 1
    a = el.find("a")
    assert a is not None
    assert a["href"] == "app.xhtml"
    assert a.get_text() == "线上补充材料"
