"""Derive a Chinese-only EPUB from a bilingual EPUB by rewriting zip HTML."""
from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup

from ._util import cn_en_counts, has_zh_follow, read_text, rezip, write_text_utf8

# Never removed: dropping cells shifts table columns; dropping headings can
# break TOC anchors. Untranslated English there is preferable to lost content.
_PROTECTED = {"td", "th", "h1", "h2", "h3", "h4", "h5", "h6"}


def _sanitize(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", name).strip().strip(".")
    return name[:120] or "book"


def _strip_english_html(html: str) -> tuple[str, int]:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.body or soup
    tags = ["p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "td", "th", "small", "div"]
    nodes = list(body.find_all(tags))
    nodes.sort(key=lambda n: len(list(n.parents)), reverse=True)
    removed = 0
    for node in nodes:
        if node.name == "div" and node.find(tags):
            continue
        if node.name in _PROTECTED:
            continue
        text = " ".join(node.get_text(" ", strip=True).split())
        if not text:
            continue
        cn, en = cn_en_counts(text)
        english = cn == 0 and en >= 8
        mostly_en = cn > 0 and en > 0 and (cn / max(cn + en, 1)) < 0.25 and en >= 20
        if (english or mostly_en) and has_zh_follow(node):
            node.decompose()
            removed += 1
    return str(soup), removed


def derive_zh(bilingual_epub: Path, out_dir: Path) -> Path:
    stem = bilingual_epub.stem
    for token in ("_zh-bilingual", "-bilingual", "_bilingual", "-中英双语"):
        stem = stem.replace(token, "")
    out = out_dir / f"{_sanitize(stem)}-中文.epub"
    tmp = Path(tempfile.mkdtemp(prefix="derive-zh-"))
    try:
        with zipfile.ZipFile(bilingual_epub) as zin:
            zin.extractall(tmp)
        for f in tmp.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in {".html", ".xhtml", ".htm"}:
                continue
            html = read_text(f)
            new_html, n = _strip_english_html(html)
            if n:
                write_text_utf8(f, new_html)
        rezip(tmp, out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not out.exists() or out.stat().st_size < 1000:
        raise RuntimeError(f"derive zh produced empty/small file: {out}")
    return out
