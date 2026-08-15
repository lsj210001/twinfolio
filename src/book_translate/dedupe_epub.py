#!/usr/bin/env python3
"""Collapse consecutive duplicate / near-duplicate blocks in a translated EPUB."""
from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from ._util import cn_en_counts, read_text, rezip, write_text_utf8

BLOCK = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "blockquote", "small"]


def _norm(text: str) -> str:
    """Aggressive normalization for duplicate detection only."""
    t = text.replace("\xa0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"第\s*(\d+)\s*章", r"第\1章", t)
    t = t.replace("作者：", "").replace("作者:", "")
    t = re.sub(r"[“”\"'《》·，,。．.：:；;、()\s]", "", t)
    return t.lower()


def _cite_no(text: str) -> str | None:
    m = re.match(r"\s*\[(\d+)\]", text)
    return m.group(1) if m else None


def _core(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r"（[^）]*）", "", t)
    t = re.sub(r"\([^)]*\)", "", t)
    return _norm(t)


def _zh_dominant(text: str) -> bool:
    cn, en = cn_en_counts(text)
    return cn >= en


def similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    na, nb = _norm(a), _norm(b)
    if na and na == nb:
        return True
    ca, cb = _cite_no(a), _cite_no(b)
    if ca and cb and ca == cb:
        return True
    fa = re.match(r"^\s*图\s*([\d.]+)", a)
    fb = re.match(r"^\s*图\s*([\d.]+)", b)
    if fa and fb and fa.group(1) == fb.group(1):
        return True
    oa, ob = _core(a), _core(b)
    if oa and oa == ob and len(oa) >= 8:
        return True
    # shared-prefix duplicates only count when one side is English and the
    # other Chinese (a translation pair); two same-language paragraphs that
    # merely start alike ("本章介绍…" vs "本章小结") are legitimate content
    if (
        _zh_dominant(a) != _zh_dominant(b)
        and min(len(na), len(nb)) >= 18
        and (na.startswith(nb[:18]) or nb.startswith(na[:18]))
    ):
        return True
    if (
        _zh_dominant(a) != _zh_dominant(b)
        and na
        and nb
        and (na in nb or nb in na)
        and min(len(na), len(nb)) >= 16
    ):
        return True
    return False


def prefer(keep: str, cand: str) -> str:
    """Prefer the more complete Chinese version, else longer."""
    c1, _ = cn_en_counts(keep)
    c2, _ = cn_en_counts(cand)
    if c2 > c1 + 2:
        return cand
    if c1 > c2 + 2:
        return keep
    return cand if len(cand) > len(keep) else keep


def collapse(html: str) -> tuple[str, int]:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.body or soup
    nodes: list[Tag] = []
    for el in body.find_all(BLOCK):
        if el.find(BLOCK):
            continue
        t = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if len(t) < 2:
            continue
        nodes.append(el)

    removed = 0
    i = 0
    while i < len(nodes):
        texts = [re.sub(r"\s+", " ", nodes[i].get_text(" ", strip=True))]
        j = i + 1
        while j < len(nodes) and similar(texts[0], re.sub(r"\s+", " ", nodes[j].get_text(" ", strip=True))):
            texts.append(re.sub(r"\s+", " ", nodes[j].get_text(" ", strip=True)))
            j += 1
        if j - i >= 2:
            # Keep English (if any) + one Chinese, or a single best Chinese.
            en_idx = None
            zh_best = None
            zh_text = ""
            for k, node in enumerate(nodes[i:j]):
                t = texts[k]
                cn, en = cn_en_counts(t)
                if cn == 0 and en >= 6:
                    if en_idx is None:
                        en_idx = i + k
                    continue
                if zh_best is None:
                    zh_best = i + k
                    zh_text = t
                else:
                    better = prefer(zh_text, t)
                    if better == t:
                        zh_best = i + k
                        zh_text = t
            keep = set()
            if en_idx is not None:
                keep.add(en_idx)
            if zh_best is not None:
                keep.add(zh_best)
            elif en_idx is None:
                keep.add(i)
            for k in range(i, j):
                if k not in keep:
                    nodes[k].decompose()
                    removed += 1
        i = j
    return str(soup), removed


def rewrite(src: Path, dest: Path) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="dedupe-"))
    total = 0
    try:
        with zipfile.ZipFile(src) as zin:
            zin.extractall(tmp)
        for f in tmp.rglob("*"):
            # .ncx stays untouched: html.parser would lowercase navMap/navPoint
            if not f.is_file() or f.suffix.lower() not in {".html", ".xhtml", ".htm"}:
                continue
            html = read_text(f)
            new, n = collapse(html)
            if n:
                write_text_utf8(f, new)
                total += n
        rezip(tmp, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return total


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    n = rewrite(Path(args.src), Path(args.output))
    print(f"removed {n} -> {args.output}")


if __name__ == "__main__":
    main()
