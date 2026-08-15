#!/usr/bin/env python3
"""Fixed post-process: translate leftover English titles in EPUB headings/TOC.

Body paragraphs are handled by file-resume; TOC/nav/ncx and nested heading
markup often still remain English. This module batch-translates remaining
English titles and rewrites the EPUB via zip (avoids ebooklib NCX issues).
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, NavigableString

from ._util import LlmCache, chat, cn_en_counts, decode_bytes, has_zh_follow, norm as _norm, parse_numbered_lines, read_text, rezip, write_text_utf8
from .glossary import Term, apply_glossary, restore_text

_has_zh_follow = has_zh_follow


def _is_en_title(text: str) -> bool:
    t = _norm(text)
    if not t or len(t) < 2:
        return False
    cn, en = cn_en_counts(t)
    if cn > 0:
        return False
    if en < 2:
        return False
    if re.fullmatch(r"[\d\.\-\s]+", t):
        return False
    # skip pure filenames / urls
    if t.startswith("http") or t.endswith(".html") or t.endswith(".xhtml"):
        return False
    return True


def _collect_titles_from_epub(epub_path: Path) -> list[str]:
    titles: "OrderedDict[str, None]" = OrderedDict()

    def consider(text: str) -> None:
        t = _norm(text)
        if _is_en_title(t) and t not in titles:
            titles[t] = None

    with zipfile.ZipFile(epub_path, "r") as zin:
        for name in zin.namelist():
            low = name.lower()
            if not low.endswith((".html", ".xhtml", ".htm", ".ncx", ".xml")):
                continue
            try:
                text = decode_bytes(zin.read(name))
            except Exception:
                continue
            try:
                soup = BeautifulSoup(
                    text,
                    "lxml-xml"
                    if text.lstrip().startswith("<?xml") or low.endswith(".ncx")
                    else "html.parser",
                )
            except Exception:
                soup = BeautifulSoup(text, "html.parser")

            is_toc = ("toc" in low) or ("nav" in low) or low.endswith(".ncx")
            tags = ["h1", "h2", "h3", "h4", "h5", "h6", "title", "navLabel", "text"]
            if is_toc:
                tags += ["a", "span", "li"]
            for node in soup.find_all(tags):
                t = _norm(node.get_text(" ", strip=True))
                if not t:
                    continue
                if node.name in ("li", "div") and node.find(["h1", "h2", "h3", "h4", "a", "p"]):
                    if len(t) > 80:
                        continue
                if node.name in ("a", "text", "span", "li") and len(t) > 120:
                    continue
                # Body headings already have a Chinese sibling from file-resume.
                # Replacing them in place would yield 中文+中文.
                if not is_toc and _has_zh_follow(node):
                    continue
                consider(t)
    return list(titles.keys())


def _translate_titles(
    titles: list[str],
    *,
    base_url: str,
    api_key: str,
    model: str,
    reasoning_effort: str = "minimal",
    batch_size: int = 40,
    log_path: Optional[Path] = None,
    glossary: list[Term] | None = None,
    cache: LlmCache | None = None,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not titles:
        return mapping

    def log(msg: str) -> None:
        line = f"[title_postprocess] {msg}"
        print(line, flush=True)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    api = {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "reasoning_effort": reasoning_effort,
    }
    terms = glossary or []
    for i in range(0, len(titles), batch_size):
        batch = titles[i : i + batch_size]
        log(f"translate titles {i+1}-{i+len(batch)} / {len(titles)}")
        protected, maps, hint, appearing = apply_glossary(batch, terms)
        prefix = (hint + "\n\n") if hint else ""
        payload_lines = [f"{idx}. {t}" for idx, t in enumerate(protected, 1)]
        prompt = (
            prefix
            + "你是专业中文图书编辑。把下面英文标题译成简洁自然的简体中文标题。\n"
            "要求：\n"
            "1) 保持原编号，每行一条：`<编号>. <中文标题>`\n"
            "2) 只输出译文列表，不要解释\n"
            "3) 专有名词可保留英文或中英并用（如 DiT、CogVideoX、GANs、Sharpe）\n"
            "4) Chapter N. xxx → 第N章 xxx\n"
            "5) 尽量短，像目录标题\n"
            "6) 术语表与 @@BTn@@ 占位符必须遵守，不得改写\n\n"
            + "\n".join(payload_lines)
        )
        part: dict[str, str] = {}
        try:
            raw = chat(api, prompt, max_tokens=4000, log=log, cache=cache, source="\n".join(batch))
            for idx, zh in parse_numbered_lines(raw).items():
                if 1 <= idx <= len(batch) and zh:
                    part[batch[idx - 1]] = restore_text(zh, maps[idx - 1], appearing)
        except Exception as e:
            log(f"batch failed: {e}")
        mapping.update(part)
        missing = [t for t in batch if t not in part]
        for t in missing:
            try:
                j = batch.index(t)
                zh = chat(
                    api,
                    prefix + f"把这个英文标题译成简洁自然的简体中文标题，只输出中文标题本身：\n{protected[j]}",
                    max_tokens=max(1000, 3 * len(t)),
                    log=log,
                    cache=cache,
                    source=t,
                )
                zh = " ".join(zh.split()).strip().strip("\"“”")
                zh = restore_text(zh, maps[j], appearing)
                if zh:
                    mapping[t] = zh
                    log(f"single: {t} => {zh}")
            except Exception as e:
                log(f"single fail for {t!r}: {e}")
        time.sleep(0.2)
    return mapping


def _write_element_text(el, zh: str) -> None:
    """Replace visible text without flattening NCX <navLabel><text>."""
    name = (el.name or "").split("}")[-1].lower()
    if name == "navlabel":
        inner = None
        for child in el.children:
            if getattr(child, "name", None) and (child.name or "").split("}")[-1].lower() == "text":
                inner = child
                break
        if inner is None:
            el.clear()
            inner = BeautifulSoup("", "lxml-xml").new_tag("text")
            el.append(inner)
        inner.clear()
        inner.append(zh)
        return
    el.clear()
    el.append(zh)


def _force_replace_element(el, mapping: dict[str, str], nmap: dict[str, str]) -> int:
    full = _norm(el.get_text(" ", strip=True))
    rawfull = " ".join(el.get_text(" ", strip=True).split())
    zh = nmap.get(full)
    if zh is None:
        zh = mapping.get(rawfull)
    if zh is not None:
        if el.name == "a" or el.find("a") is None:
            _write_element_text(el, zh)
            return 1
        # clearing this element would destroy <a href> links (e.g. visible
        # TOC pages); translate the anchors themselves and keep structure
        replaced = 0
        for a in el.find_all("a"):
            a_text = _norm(a.get_text(" ", strip=True))
            target = nmap.get(a_text)
            if target is None and a_text == full:
                target = zh
            if target is not None:
                _write_element_text(a, target)
                replaced += 1
        if replaced:
            return 1
        # fall through to per-text-node replacement, links stay intact
    for child in list(el.children):
        if isinstance(child, NavigableString):
            s = str(child)
            key = _norm(s)
            if key in nmap:
                lead = s[: len(s) - len(s.lstrip())]
                trail = s[len(s.rstrip()) :]
                child.replace_with(lead + nmap[key] + trail)
                return 1
    return 0


def _apply_mapping_zip(epub_path: Path, mapping: dict[str, str], out_path: Path) -> int:
    if not mapping:
        shutil.copy2(epub_path, out_path)
        return 0
    nmap = {_norm(k): v for k, v in mapping.items()}
    tmpdir = Path(tempfile.mkdtemp(prefix="epub-title-fix-"))
    total = 0
    try:
        with zipfile.ZipFile(epub_path, "r") as zin:
            zin.extractall(tmpdir)
        for f in tmpdir.rglob("*"):
            if not f.is_file():
                continue
            if f.suffix.lower() not in {".html", ".xhtml", ".htm", ".ncx", ".xml"}:
                continue
            text = read_text(f)
            name = str(f.relative_to(tmpdir)).lower()
            is_toc = ("toc" in name) or ("nav" in name) or name.endswith(".ncx")
            try:
                soup = BeautifulSoup(
                    text,
                    "lxml-xml"
                    if text.lstrip().startswith("<?xml") or name.endswith(".ncx")
                    else "html.parser",
                )
            except Exception:
                soup = BeautifulSoup(text, "html.parser")
            local = 0
            tags = ["h1", "h2", "h3", "h4", "h5", "h6", "title", "navLabel", "text"]
            if is_toc:
                tags += ["a", "span", "li"]
            for node in soup.find_all(tags):
                t = _norm(node.get_text(" ", strip=True))
                cn, en = cn_en_counts(t)
                if cn == 0 and en >= 2 and (t in nmap or " ".join(node.get_text(" ", strip=True).split()) in mapping):
                    if not is_toc and _has_zh_follow(node):
                        continue
                    local += _force_replace_element(node, mapping, nmap)
            if local:
                write_text_utf8(f, str(soup))
                total += local
        rezip(tmpdir, out_path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return total


def fix_english_titles(
    epub_path: Path,
    *,
    out_path: Optional[Path] = None,
    base_url: str,
    api_key: str,
    model: str,
    reasoning_effort: str = "minimal",
    log_path: Optional[Path] = None,
    map_path: Optional[Path] = None,
    glossary: list[Term] | None = None,
    cache: LlmCache | None = None,
) -> Path:
    """Translate leftover English titles and write a fixed EPUB.

    Returns output path (may equal input copy if nothing to fix).
    """
    epub_path = Path(epub_path)
    if out_path is None:
        out_path = epub_path.with_name(epub_path.stem + "-titles-zh" + epub_path.suffix)
    out_path = Path(out_path)

    def log(msg: str) -> None:
        line = f"[title_postprocess] {msg}"
        print(line, flush=True)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    titles = _collect_titles_from_epub(epub_path)
    log(f"{epub_path.name}: english titles found = {len(titles)}")
    if not titles:
        shutil.copy2(epub_path, out_path)
        log(f"{epub_path.name}: nothing to fix, copied")
        return out_path

    mapping = _translate_titles(
        titles,
        base_url=base_url,
        api_key=api_key,
        model=model,
        reasoning_effort=reasoning_effort,
        log_path=log_path,
        glossary=glossary,
        cache=cache,
    )
    log(f"{epub_path.name}: mapped {len(mapping)}/{len(titles)}")
    if map_path:
        Path(map_path).write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    replaced = _apply_mapping_zip(epub_path, mapping, out_path)
    from .toc_repair import repair_epub_toc

    repair_epub_toc(out_path)
    log(f"{epub_path.name}: replacements={replaced}, out={out_path.name}, size={out_path.stat().st_size}")
    if not out_path.exists() or out_path.stat().st_size < 1000:
        raise RuntimeError(f"title postprocess produced empty/small file: {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Fix English titles in an EPUB")
    ap.add_argument("epub")
    ap.add_argument("-o", "--output", default="")
    ap.add_argument("--map", default="")
    args = ap.parse_args()

    out = fix_english_titles(
        Path(args.epub),
        out_path=Path(args.output) if args.output else None,
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        model=os.environ.get("BOOK_TRANSLATE_MODEL") or os.environ.get("MODEL", "gpt-4o-mini"),
        reasoning_effort=os.environ.get("BOOK_TRANSLATE_REASONING_EFFORT")
        or os.environ.get("BBM_REASONING_EFFORT")
        or "low",
        map_path=Path(args.map) if args.map else None,
    )
    print(out)
