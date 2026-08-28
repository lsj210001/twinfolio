"""Regression tests for encoding sniffing gaps and invalid-XML OPF patching.

Covers: (a) HTML meta charset, (b) latin-1 declarations on real UTF-8 bytes,
(c) BOM-less UTF-16, (d) gb2312-labelled GBK — plus self-closing manifest/spine
in OPF patching, and a GBK NCX surviving TOC repair without mojibake.
"""

import codecs
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from book_translate import file_resume, toc_repair
from book_translate._util import decode_bytes, ensure_utf8_declaration

# ---------------------------------------------------------------------------
# decode_bytes


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # (a) HTML meta charset: legacy cp1252 book without an XML declaration
        pytest.param(
            b'<html><head><meta charset="windows-1252"></head>'
            b"<body>caf\xe9 \x93quoted\x94</body></html>",
            "caf\xe9 \u201cquoted\u201d",
            id="html-meta-charset-cp1252",
        ),
        pytest.param(
            b'<html><head><meta http-equiv="Content-Type" '
            b'content="text/html; charset=iso-8859-1"></head>'
            b"<body>na\xefve</body></html>",
            "na\xefve",
            id="html-meta-http-equiv-latin1",
        ),
        # (b) declared latin-1/iso-8859-1 but the bytes are valid UTF-8:
        # strict UTF-8 must win instead of silently producing mojibake
        pytest.param(
            '<?xml version="1.0" encoding="iso-8859-1"?><p>café 中文</p>'.encode("utf-8"),
            "café 中文",
            id="mislabelled-latin1-actually-utf8",
        ),
        # genuine cp1252 bytes under a latin-1 label still decode as cp1252
        pytest.param(
            b'<?xml version="1.0" encoding="iso-8859-1"?><p>\x93smart\x94</p>',
            "\u201csmart\u201d",
            id="genuine-cp1252-smart-quotes",
        ),
        # (c) UTF-16 without BOM, both endiannesses
        pytest.param(
            '<?xml version="1.0" encoding="utf-16"?><p>你好世界</p>'.encode("utf-16-le"),
            "你好世界",
            id="utf16-le-no-bom",
        ),
        pytest.param(
            '<?xml version="1.0" encoding="utf-16"?><p>你好世界</p>'.encode("utf-16-be"),
            "你好世界",
            id="utf16-be-no-bom",
        ),
        # (d) gb2312 label on GBK content (U+9555 镕 exists in GBK, not GB2312)
        pytest.param(
            '<?xml version="1.0" encoding="gb2312"?><p>朱镕基传</p>'.encode("gbk"),
            "朱镕基传",
            id="gb2312-alias-decodes-as-gbk",
        ),
    ],
)
def test_decode_bytes_handles_legacy_encodings(raw: bytes, expected: str):
    assert expected in decode_bytes(raw, log=lambda m: None)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(codecs.BOM_UTF8 + "中文".encode("utf-8"), "中文", id="utf8-bom"),
        pytest.param("plain utf-8 中文".encode("utf-8"), "plain utf-8 中文", id="utf8-plain"),
        pytest.param(
            '<?xml version="1.0" encoding="utf-8"?><p>中文</p>'.encode("utf-8"),
            "中文",
            id="utf8-declared",
        ),
        pytest.param(
            '<?xml version="1.0" encoding="utf-16"?><p>中文</p>'.encode("utf-16"),
            "中文",
            id="utf16-with-bom",
        ),
        pytest.param(
            '<?xml version="1.0" encoding="gbk"?><p>中文</p>'.encode("gbk"),
            "中文",
            id="gbk-declared",
        ),
    ],
)
def test_decode_bytes_existing_paths_unchanged(raw: bytes, expected: str):
    assert expected in decode_bytes(raw, log=lambda m: None)


def test_decode_bytes_warns_on_heavy_replacement_damage():
    messages: list[str] = []
    out = decode_bytes(b"\xff" * 64, log=messages.append)
    assert "\ufffd" in out
    assert messages and "warning" in messages[0]


def test_decode_bytes_no_warning_on_clean_text():
    messages: list[str] = []
    decode_bytes("clean 中文 text".encode("utf-8"), log=messages.append)
    assert messages == []


# ---------------------------------------------------------------------------
# ensure_utf8_declaration


def test_ensure_utf8_declaration_rewrites_meta_charset_too():
    text = (
        '<?xml version="1.0" encoding="gbk"?><html><head>'
        '<meta charset="gbk"/>'
        '<meta http-equiv="Content-Type" content="text/html; charset=gb2312"/>'
        "</head><body>ok</body></html>"
    )
    out = ensure_utf8_declaration(text)
    assert 'encoding="utf-8"' in out
    assert '<meta charset="utf-8"/>' in out
    assert "charset=utf-8" in out
    assert "gbk" not in out.lower()
    assert "gb2312" not in out.lower()


def test_ensure_utf8_declaration_leaves_utf8_untouched():
    text = (
        '<?xml version="1.0" encoding="utf-8"?><html><head>'
        '<meta charset="utf-8"/></head><body>ok</body></html>'
    )
    assert ensure_utf8_declaration(text) == text


# ---------------------------------------------------------------------------
# self-closing <manifest/> / <spine/> in OPF patching


@pytest.mark.parametrize(
    ("manifest", "spine"),
    [
        pytest.param("<manifest/>", "<spine/>", id="bare-self-closing"),
        pytest.param("<manifest />", "<spine />", id="space-before-slash"),
        pytest.param("<manifest/>", '<spine toc="ncx"/>', id="spine-with-attrs"),
    ],
)
def test_patch_opf_self_closing_manifest_spine_stays_valid_xml(manifest: str, spine: str):
    opf = (
        '<?xml version="1.0"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">\n'
        f"  <metadata/>\n  {manifest}\n  {spine}\n</package>\n"
    )
    out = file_resume._patch_opf(opf, ["EPUB/Text/alpha.xhtml"], opf_zip_path="EPUB/content.opf")
    root = ET.fromstring(out)  # must parse: no content after </package>
    ns = "{http://www.idpf.org/2007/opf}"
    items = root.findall(f"{ns}manifest/{ns}item")
    itemrefs = root.findall(f"{ns}spine/{ns}itemref")
    assert [i.get("href") for i in items] == ["Text/alpha.xhtml"]
    assert len(itemrefs) == 1
    assert itemrefs[0].get("idref") == items[0].get("id")


def test_insert_before_close_never_appends_after_package():
    text = '<?xml version="1.0"?><package><metadata/></package>'
    messages: list[str] = []
    out = file_resume._insert_before_close(text, "manifest", "<item/>", log=messages.append)
    assert out == text
    assert messages and "manifest" in messages[0]


# ---------------------------------------------------------------------------
# TOC repair must not corrupt non-UTF-8 files

GBK_NCX = (
    '<?xml version="1.0" encoding="gb2312"?>\n'
    '<ncx version="2005-1" xmlns="http://www.daisy.org/z3986/2005/ncx/">\n'
    "<head/>\n"
    "<docTitle><text/></docTitle>\n"
    "<navMap>\n"
    '<navPoint id="n1"><navLabel><text>第1章 朱镕基年代</text></navLabel>'
    '<content src="Text/c1.xhtml"/></navPoint>\n'
    "</navMap>\n"
    "</ncx>\n"
)

CP1252_OPF = (
    '<?xml version="1.0" encoding="iso-8859-1"?>\n'
    '<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="id" version="3.0">\n'
    '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
    "    <dc:title>Caf\xe9 Memoir</dc:title>\n"
    "    <dc:creator>Ren\xe9e</dc:creator>\n"
    '    <dc:identifier id="id">urn:demo</dc:identifier>\n'
    "  </metadata>\n"
    "  <manifest>\n"
    '    <item href="toc.ncx" id="ncx" media-type="application/x-dtbncx+xml"/>\n'
    '    <item href="Text/c1.xhtml" id="c1" media-type="application/xhtml+xml"/>\n'
    "  </manifest>\n"
    '  <spine toc="ncx">\n'
    '    <itemref idref="c1"/>\n'
    "  </spine>\n"
    "</package>\n"
)


def test_repair_extracted_gbk_ncx_and_cp1252_opf_no_mojibake(tmp_path: Path):
    root = tmp_path / "book"
    (root / "EPUB" / "Text").mkdir(parents=True)
    (root / "EPUB" / "toc.ncx").write_bytes(GBK_NCX.encode("gbk"))
    (root / "EPUB" / "content.opf").write_bytes(CP1252_OPF.encode("cp1252"))
    (root / "EPUB" / "Text" / "c1.xhtml").write_text(
        "<html><body><p>hi</p></body></html>", encoding="utf-8"
    )

    stats = toc_repair.repair_extracted(root)
    assert stats["wrote_ncx"] == 1
    assert stats["wrote_nav"] == 1

    ncx = (root / "EPUB" / "toc.ncx").read_text(encoding="utf-8")
    nav = (root / "EPUB" / "nav.xhtml").read_text(encoding="utf-8")
    opf = (root / "EPUB" / "content.opf").read_text(encoding="utf-8")
    # GBK chapter label survives (镕 is GBK-only; the old utf-8/ignore read
    # dropped these bytes entirely)
    assert "第1章 朱镕基年代" in ncx
    assert "第1章 朱镕基年代" in nav
    # cp1252 title from the OPF is used and not mangled
    assert "Caf\xe9 Memoir" in ncx
    assert "Caf\xe9 Memoir" in opf
    assert 'properties="nav"' in opf
    ET.fromstring(opf)  # rewritten OPF stays well-formed
