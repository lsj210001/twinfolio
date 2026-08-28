import logging
import zipfile
from pathlib import Path

from book_translate import epub_io

CONTAINER = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <!-- <rootfile full-path="old-backup.opf" media-type="application/oebps-package+xml"/> -->
    <rootfile media-type='application/oebps-package+xml'
              full-path='real/package.opf'/>
  </rootfiles>
</container>
"""

OPF_WITH_NCX = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="id">
  <manifest>
    <item href="wrong.ncx" id="not-the-toc" media-type="application/x-dtbncx+xml"/>
    <item href="toc.ncx" id="ncx" media-type="application/x-dtbncx+xml"/>
    <item href="Text/c1.xhtml" id="c1" media-type="application/xhtml+xml"/>
  </manifest>
  <spine toc="ncx">
    <itemref idref="c1"/>
  </spine>
</package>
"""


def test_container_opf_path_uses_xml_parser_not_regex():
    # a regex scan would match the commented-out backup rootfile first
    assert epub_io.container_opf_path(CONTAINER) == "real/package.opf"


def test_find_opf_in_zip_ignores_decoy(tmp_path: Path, caplog):
    epub = tmp_path / "book.epub"
    with zipfile.ZipFile(epub, "w") as z:
        z.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", CONTAINER)
        z.writestr("a-decoy.opf", "<package/>")
        z.writestr("real/package.opf", "<package/>")
    with caplog.at_level(logging.WARNING, logger="book_translate.epub_io"):
        with zipfile.ZipFile(epub) as z:
            assert epub_io.find_opf_in_zip(z) == "real/package.opf"
    assert not caplog.records


def test_find_opf_in_tree_ignores_decoy(tmp_path: Path):
    (tmp_path / "META-INF").mkdir()
    (tmp_path / "META-INF/container.xml").write_text(CONTAINER, encoding="utf-8")
    (tmp_path / "a-decoy.opf").write_text("<package/>", encoding="utf-8")
    (tmp_path / "real").mkdir()
    (tmp_path / "real/package.opf").write_text("<package/>", encoding="utf-8")
    assert epub_io.find_opf_in_tree(tmp_path) == tmp_path / "real/package.opf"


def test_missing_declared_opf_warns_before_fallback(tmp_path: Path, caplog):
    (tmp_path / "META-INF").mkdir()
    (tmp_path / "META-INF/container.xml").write_text(CONTAINER, encoding="utf-8")
    (tmp_path / "a-decoy.opf").write_text("<package/>", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="book_translate.epub_io"):
        assert epub_io.find_opf_in_tree(tmp_path) == tmp_path / "a-decoy.opf"
    assert any("missing OPF" in r.message for r in caplog.records)


def test_no_container_warns_before_fallback(tmp_path: Path, caplog):
    (tmp_path / "b.opf").write_text("<package/>", encoding="utf-8")
    (tmp_path / "a.opf").write_text("<package/>", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="book_translate.epub_io"):
        assert epub_io.find_opf_in_tree(tmp_path) == tmp_path / "a.opf"
    assert any("container.xml" in r.message for r in caplog.records)


def test_ncx_href_prefers_spine_toc_then_media_type():
    assert epub_io.ncx_href_from_opf(OPF_WITH_NCX) == "toc.ncx"
    no_spine_toc = OPF_WITH_NCX.replace(' toc="ncx"', "")
    assert epub_io.ncx_href_from_opf(no_spine_toc) == "wrong.ncx"
    assert epub_io.ncx_href_from_opf("<package><manifest/></package>") is None


def test_find_ncx_in_tree_resolves_from_selected_opf(tmp_path: Path):
    (tmp_path / "a-decoy.ncx").write_text("<ncx/>", encoding="utf-8")
    (tmp_path / "real").mkdir()
    (tmp_path / "real/package.opf").write_text(OPF_WITH_NCX, encoding="utf-8")
    (tmp_path / "real/toc.ncx").write_text("<ncx/>", encoding="utf-8")
    got = epub_io.find_ncx_in_tree(tmp_path, tmp_path / "real/package.opf")
    assert got == tmp_path / "real/toc.ncx"


def test_find_ncx_in_tree_without_any_opf_warns_and_scans(tmp_path: Path, caplog):
    (tmp_path / "only.ncx").write_text("<ncx/>", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="book_translate.epub_io"):
        assert epub_io.find_ncx_in_tree(tmp_path, None) == tmp_path / "only.ncx"
    assert any("no OPF" in r.message for r in caplog.records)
