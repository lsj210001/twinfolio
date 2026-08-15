from pathlib import Path
import zipfile

from book_translate.title_postprocess import _write_element_text
from book_translate.toc_repair import ncx_to_nav_xhtml, repair_epub_toc, repair_ncx_xml
from bs4 import BeautifulSoup


BROKEN_NCX = """<?xml version="1.0" encoding="utf-8"?>
<ncx version="2005-1" xmlns="http://www.daisy.org/z3986/2005/ncx/">
<head>
<meta content="uid" name="dtb:uid"/>
<meta content="0" name="dtb:depth"/>
</head>
<docTitle><text/></docTitle>
<navMap>
<navPoint id="navPoint-1">
<navLabel>第1章 伊利亚看到了什么？</navLabel>
<content src="Text/chapter-1.xhtml"/>
<navPoint id="navPoint-2">
<navLabel>1.1 伊利亚的崛起</navLabel>
<content src="Text/chapter-1.xhtml#p14"/>
</navPoint>
</navPoint>
</navMap>
</ncx>
"""


def test_repair_wraps_navlabel_and_playorder():
    out = repair_ncx_xml(BROKEN_NCX, title="Sutskever's List", author="Richard Heimann")
    soup = BeautifulSoup(out, "lxml-xml")
    points = soup.find_all("navPoint")
    assert len(points) == 2
    assert [p.get("playOrder") for p in points] == ["1", "2"]
    for p in points:
        label = p.find("navLabel")
        text = label.find("text")
        assert text is not None
        assert text.get_text(strip=True)
    assert soup.find("docTitle").find("text").get_text(strip=True) == "Sutskever's List"
    assert "dtb:depth" in str(soup)
    assert soup.find("meta", attrs={"name": "dtb:depth"}).get("content") == "2"


def test_nav_xhtml_uses_titles_not_numbers():
    repaired = repair_ncx_xml(BROKEN_NCX, title="目录")
    nav = ncx_to_nav_xhtml(repaired, title="目录")
    assert "第1章 伊利亚看到了什么？" in nav
    assert "1.1 伊利亚的崛起" in nav
    assert 'epub:type="toc"' in nav
    assert "Text/chapter-1.xhtml#p14" in nav


def test_write_navlabel_keeps_text_child():
    soup = BeautifulSoup("<navLabel><text>copyright</text></navLabel>", "lxml-xml")
    el = soup.find("navLabel")
    _write_element_text(el, "版权信息")
    assert el.find("text") is not None
    assert el.find("text").get_text() == "版权信息"
    assert el.find("text").parent.name == "navLabel"


def test_repair_epub_adds_nav(tmp_path: Path):
    src = tmp_path / "book.epub"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr(
            "EPUB/content.opf",
            """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="id" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Demo</dc:title>
    <dc:creator>Author</dc:creator>
    <dc:identifier id="id">urn:demo</dc:identifier>
  </metadata>
  <manifest>
    <item href="toc.ncx" id="ncx" media-type="application/x-dtbncx+xml"/>
    <item href="Text/chapter-1.xhtml" id="c1" media-type="application/xhtml+xml"/>
  </manifest>
  <spine toc="ncx">
    <itemref idref="c1"/>
  </spine>
</package>
""",
        )
        z.writestr("EPUB/toc.ncx", BROKEN_NCX)
        z.writestr("EPUB/Text/chapter-1.xhtml", "<html><body><p>hi</p></body></html>")
    out = repair_epub_toc(src)
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert "EPUB/nav.xhtml" in names
        ncx = z.read("EPUB/toc.ncx").decode("utf-8")
        opf = z.read("EPUB/content.opf").decode("utf-8")
        nav = z.read("EPUB/nav.xhtml").decode("utf-8")
    assert "<text>第1章 伊利亚看到了什么？</text>" in ncx
    assert 'playOrder="1"' in ncx
    assert 'properties="nav"' in opf
    assert "第1章 伊利亚看到了什么？" in nav
