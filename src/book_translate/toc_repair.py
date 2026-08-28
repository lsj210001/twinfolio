"""Repair EPUB TOC so readers show chapter titles instead of 1, 2, 3.

ebooklib rewrites EPUB 2 as EPUB 3 and drops playOrder / navLabel/<text>.
title_postprocess used to flatten navLabel to a bare string. Many readers then
fall back to spine order numbers.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from ._util import read_text, rezip


def _local(name: str | None) -> str:
    if not name:
        return ""
    return name.split("}")[-1]


def _children(el: Tag, local: str) -> list[Tag]:
    out: list[Tag] = []
    for child in el.children:
        if isinstance(child, Tag) and _local(child.name) == local:
            out.append(child)
    return out


def _first(el: Tag, local: str) -> Tag | None:
    kids = _children(el, local)
    return kids[0] if kids else None


def _find_one(root: Path, suffix: str) -> Path | None:
    hits = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == suffix)
    return hits[0] if hits else None


def _dc_text(opf_soup: BeautifulSoup, local: str) -> str:
    for el in opf_soup.find_all(True):
        if _local(el.name) == local:
            text = " ".join(el.get_text(" ", strip=True).split())
            if text:
                return text
    return ""


def _set_meta(head: Tag, soup: BeautifulSoup, name: str, content: str) -> None:
    for meta in head.find_all("meta"):
        if meta.get("name") == name:
            meta["content"] = content
            return
    meta = soup.new_tag("meta")
    meta["name"] = name
    meta["content"] = content
    head.append(meta)


def _ensure_named(parent: Tag, soup: BeautifulSoup, local: str) -> Tag:
    el = _first(parent, local)
    if el is not None:
        return el
    el = soup.new_tag(local)
    parent.append(el)
    return el


def _set_ncx_text(parent: Tag, soup: BeautifulSoup, value: str) -> None:
    text_el = _first(parent, "text")
    if text_el is None:
        parent.clear()
        text_el = soup.new_tag("text")
        parent.append(text_el)
    text_el.clear()
    if value:
        text_el.append(value)


def repair_ncx_xml(raw: str, *, title: str = "", author: str = "") -> str:
    soup = BeautifulSoup(raw, "lxml-xml")
    ncx = soup.find("ncx")
    if ncx is None:
        return raw

    if not ncx.get("xmlns"):
        ncx["xmlns"] = "http://www.daisy.org/z3986/2005/ncx/"
    ncx["version"] = ncx.get("version") or "2005-1"

    head = _ensure_named(ncx, soup, "head")
    doc_title = _ensure_named(ncx, soup, "docTitle")
    if title:
        _set_ncx_text(doc_title, soup, title)
    elif not doc_title.get_text(strip=True):
        _set_ncx_text(doc_title, soup, "目录")

    nav_map = _first(ncx, "navMap")
    if author:
        doc_author = _first(ncx, "docAuthor")
        if doc_author is None:
            doc_author = soup.new_tag("docAuthor")
            if nav_map is not None:
                nav_map.insert_before(doc_author)
            else:
                ncx.append(doc_author)
        if not doc_author.get_text(strip=True):
            _set_ncx_text(doc_author, soup, author)

    if nav_map is None:
        return str(soup)

    points = list(nav_map.find_all("navPoint"))
    for i, np in enumerate(points, 1):
        np["playOrder"] = str(i)
        label = _first(np, "navLabel")
        if label is None:
            continue
        _set_ncx_text(label, soup, " ".join(label.get_text(" ", strip=True).split()))

    depth = 0
    top = _children(nav_map, "navPoint")

    def walk(nodes: list[Tag], level: int) -> None:
        nonlocal depth
        if not nodes:
            return
        depth = max(depth, level)
        for node in nodes:
            walk(_children(node, "navPoint"), level + 1)

    walk(top, 1)
    _set_meta(head, soup, "dtb:depth", str(depth))
    xml = soup.encode(encoding="utf-8", formatter="minimal").decode("utf-8")
    if not xml.lstrip().startswith("<?xml"):
        xml = '<?xml version="1.0" encoding="utf-8"?>\n' + xml
    return xml


def ncx_to_nav_xhtml(ncx_raw: str, *, title: str = "目录") -> str:
    ncx_soup = BeautifulSoup(ncx_raw, "lxml-xml")
    ncx = ncx_soup.find("ncx")
    html_soup = BeautifulSoup("", "lxml-xml")
    html = html_soup.new_tag("html")
    html["xmlns"] = "http://www.w3.org/1999/xhtml"
    html["xmlns:epub"] = "http://www.idpf.org/2007/ops"
    html_soup.append(html)

    head = html_soup.new_tag("head")
    title_el = html_soup.new_tag("title")
    title_el.string = title or "目录"
    head.append(title_el)
    html.append(head)

    body = html_soup.new_tag("body")
    nav = html_soup.new_tag("nav")
    nav["epub:type"] = "toc"
    nav["id"] = "toc"
    h1 = html_soup.new_tag("h1")
    h1.string = title or "目录"
    nav.append(h1)

    def ol_from(parent: Tag) -> Tag:
        ol = html_soup.new_tag("ol")
        for np in _children(parent, "navPoint"):
            li = html_soup.new_tag("li")
            href = "#"
            content = _first(np, "content")
            if content is not None and content.get("src"):
                href = content.get("src")
            label = _first(np, "navLabel")
            text = " ".join((label.get_text(" ", strip=True) if label else "").split()) or "未命名"
            a = html_soup.new_tag("a", href=href)
            a.string = text
            li.append(a)
            kids = _children(np, "navPoint")
            if kids:
                li.append(ol_from(np))
            ol.append(li)
        return ol

    nav_map = _first(ncx, "navMap") if ncx is not None else None
    if nav_map is not None:
        nav.append(ol_from(nav_map))
    body.append(nav)
    html.append(body)
    xml = html_soup.encode(encoding="utf-8", formatter="minimal").decode("utf-8")
    if not xml.lstrip().startswith("<?xml"):
        xml = '<?xml version="1.0" encoding="utf-8"?>\n' + xml
    return xml


def _patch_opf(opf_path: Path, nav_href: str) -> None:
    soup = BeautifulSoup(read_text(opf_path), "lxml-xml")
    manifest = soup.find("manifest")
    if manifest is None:
        return
    existing = None
    for item in manifest.find_all("item"):
        props = (item.get("properties") or "").split()
        href = item.get("href") or ""
        if "nav" in props or href.endswith("nav.xhtml"):
            existing = item
            break
    if existing is None:
        item = soup.new_tag("item")
        item["id"] = "nav"
        item["href"] = nav_href
        item["media-type"] = "application/xhtml+xml"
        item["properties"] = "nav"
        manifest.append(item)
    else:
        existing["href"] = nav_href
        existing["media-type"] = "application/xhtml+xml"
        props = set((existing.get("properties") or "").split())
        props.add("nav")
        existing["properties"] = " ".join(sorted(props))
    xml = soup.encode(encoding="utf-8", formatter="minimal").decode("utf-8")
    if not xml.lstrip().startswith("<?xml"):
        xml = '<?xml version="1.0" encoding="utf-8"?>\n' + xml
    opf_path.write_text(xml, encoding="utf-8")


def repair_extracted(root: Path) -> dict[str, int]:
    """Repair TOC files already extracted from an EPUB. Returns counts."""
    ncx_path = _find_one(root, ".ncx")
    opf_path = _find_one(root, ".opf")
    title = ""
    author = ""
    if opf_path is not None:
        opf_soup = BeautifulSoup(read_text(opf_path), "lxml-xml")
        title = _dc_text(opf_soup, "title")
        author = _dc_text(opf_soup, "creator")

    stats = {"navpoints": 0, "wrote_ncx": 0, "wrote_nav": 0}
    if ncx_path is None:
        return stats

    repaired = repair_ncx_xml(
        read_text(ncx_path),
        title=title,
        author=author,
    )
    ncx_path.write_text(repaired, encoding="utf-8")
    stats["wrote_ncx"] = 1
    stats["navpoints"] = repaired.count("<navPoint")

    nav_dir = ncx_path.parent
    nav_path = nav_dir / "nav.xhtml"
    nav_path.write_text(ncx_to_nav_xhtml(repaired, title=title or "目录"), encoding="utf-8")
    stats["wrote_nav"] = 1

    if opf_path is not None:
        rel = str(nav_path.relative_to(opf_path.parent)).replace("\\", "/")
        _patch_opf(opf_path, rel)
    return stats


def repair_epub_toc(epub_path: Path, out_path: Path | None = None) -> Path:
    """Rewrite EPUB so NCX is valid and an EPUB3 nav document exists."""
    epub_path = Path(epub_path)
    if out_path is None:
        out_path = epub_path
    else:
        out_path = Path(out_path)

    tmp = Path(tempfile.mkdtemp(prefix="epub-toc-repair-"))
    try:
        with zipfile.ZipFile(epub_path) as zin:
            zin.extractall(tmp)
        repair_extracted(tmp)
        rezip(tmp, out_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_path
