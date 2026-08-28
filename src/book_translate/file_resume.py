#!/usr/bin/env python3
"""Resume EPUB translation for HTML files missing from a partial bilingual book."""
from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import shutil
import tempfile
import threading
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import copy
from pathlib import Path
from typing import Callable

from bs4 import BeautifulSoup, Tag

from ._util import (
    cn_en_counts,
    decode_bytes,
    has_zh_follow,
    norm,
    parse_numbered_lines,
    read_text,
    rezip,
    write_text_utf8,
)
from .epub_io import find_opf_in_tree, find_opf_in_zip
from .glossary import Term, apply_glossary, restore_text, visible_text
from .llm import LlmClient

# <title> is deliberately excluded: translate_html inserts translations as
# siblings, which would leave two <title> elements in <head> (invalid XHTML,
# epubcheck error). Title translation is title_postprocess's job.
TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "small", "blockquote"]
EXCLUDE = {"code", "pre", "sup"}
SHORT_NAME_RE = re.compile(
    r"(references?|bibliograph(?:y|ies)|index|endnotes?|footnotes?|notes|toc|nav|contents)\.(xhtml|html|htm)$",
    re.I,
)
DEFAULT_BATCH = 12
SHORT_BATCH = 8
DEFAULT_BATCH_TOKENS = 1600
SHORT_BATCH_TOKENS = 600
DEFAULT_CONTEXT_PARAGRAPHS = 8
# A real translation carries at least this many CJK chars (same bar as
# has_zh_follow); anything below is an English paraphrase/refusal, not Chinese.
MIN_ZH_CHARS = 2
# Give up on a file after this many passes that inserted no Chinese at all,
# so a stubborn file cannot burn tokens forever across reruns.
MAX_FILE_ATTEMPTS = 3
RESUME_WORKERS_MAX = 4
_RESUME_CHAT_GAP_SEC = 0.08
_CLOSE_TAG_RE = {
    "manifest": re.compile(r"</(?:[\w.-]+:)?manifest\s*>", re.I),
    "spine": re.compile(r"</(?:[\w.-]+:)?spine\s*>", re.I),
}
_SELF_CLOSING_TAG_RE = {
    "manifest": re.compile(r"<((?:[\w.-]+:)?manifest)(\b[^<>]*?)/\s*>", re.I),
    "spine": re.compile(r"<((?:[\w.-]+:)?spine)(\b[^<>]*?)/\s*>", re.I),
}


def _skip_text(text: str) -> bool:
    t = norm(text)
    if len(t) < 2:
        return True
    if t.startswith("http") or t.startswith("www.") or t.startswith("doi:"):
        return True
    if re.fullmatch(r"[\d\W_]+", t):
        return True
    cn, en = cn_en_counts(t)
    if cn >= 2:
        return True
    if en < 3:
        return True
    return False


def is_short_batch_name(name: str) -> bool:
    return bool(SHORT_NAME_RE.search(Path(name).name))


def is_short_batch_html(name: str, html: str = "") -> bool:
    """Name-only. Do not classify by paragraph median; Packt chapters look like references."""
    return is_short_batch_name(name)


def short_batch_basenames(src: Path) -> set[str]:
    found: set[str] = set()
    with zipfile.ZipFile(src) as z:
        for name in z.namelist():
            if not name.lower().endswith((".xhtml", ".html", ".htm")):
                continue
            if is_short_batch_name(name):
                found.add(Path(name).name)
    return found


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: ~1 per CJK char, ~1 per 4 Latin chars."""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = max(0, len(text) - cjk)
    return max(1, cjk + (other + 3) // 4)


def pack_batches(
    texts: list[str], *, token_budget: int, max_items: int
) -> list[tuple[int, int]]:
    """Split texts into (start, end) slices by token budget, at least one item each."""
    if token_budget < 1:
        token_budget = DEFAULT_BATCH_TOKENS
    if max_items < 1:
        max_items = 1
    out: list[tuple[int, int]] = []
    i = 0
    n = len(texts)
    while i < n:
        used = 0
        j = i
        while j < n and (j - i) < max_items:
            cost = estimate_tokens(texts[j])
            if j > i and used + cost > token_budget:
                break
            used += cost
            j += 1
        if j == i:
            j = i + 1
        out.append((i, j))
        i = j
    return out


def parse_name_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    parts = value.split(",") if isinstance(value, str) else list(value)
    return [p.strip().replace("\\", "/") for p in parts if p and str(p).strip()]


def job_matches(src_path: str, base: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    src_n = src_path.replace("\\", "/")
    for pat in patterns:
        if pat == base or pat == src_n:
            return True
        if src_n.endswith("/" + pat) or src_n.endswith(pat):
            return True
        if Path(pat).name == base:
            return True
    return False


def html_paths(epub: Path) -> list[str]:
    """Zip-internal paths of all HTML documents, in archive order."""
    with zipfile.ZipFile(epub) as z:
        return [n for n in z.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))]


def resume_worker_count(value: int | str | None = None) -> int:
    """1 = serial (default). Env BOOK_TRANSLATE_RESUME_WORKERS; cap at 4."""
    if value is None:
        value = os.environ.get("BOOK_TRANSLATE_RESUME_WORKERS", "1")
    try:
        n = int(str(value).strip() or "1")
    except (TypeError, ValueError):
        n = 1
    return max(1, min(n, RESUME_WORKERS_MAX))


def _local(name: str | None) -> str:
    if not name:
        return ""
    return name.split("}")[-1]


def _norm_href(href: str) -> str:
    href = (href or "").split("#", 1)[0].replace("\\", "/").strip()
    if not href:
        return ""
    return posixpath.normpath(href).replace("\\", "/")


def _resolve_zip_path(opf_zip: str, href: str) -> str:
    href = _norm_href(href)
    if not href:
        return ""
    opf_dir = posixpath.dirname(opf_zip.replace("\\", "/"))
    if href.startswith("/"):
        return href.lstrip("/")
    joined = posixpath.normpath(posixpath.join(opf_dir, href) if opf_dir else href)
    return joined.replace("\\", "/")


def href_for_dest(opf_zip: str, dest: str) -> str:
    """OPF href for a zip member: path relative to the OPF, never basename-only."""
    dest = dest.replace("\\", "/")
    opf_zip = (opf_zip or "").replace("\\", "/")
    opf_dir = posixpath.dirname(opf_zip) if opf_zip else ""
    return _norm_href(posixpath.relpath(dest, opf_dir or "."))


def _xml_attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def _find_local(root, local: str):
    for el in root.find_all(True):
        if _local(el.name) == local:
            return el
    return None


def _manifest_id_hrefs(soup: BeautifulSoup) -> dict[str, str]:
    manifest = _find_local(soup, "manifest")
    if manifest is None:
        return {}
    out: dict[str, str] = {}
    for item in manifest.find_all(True):
        if _local(item.name) != "item":
            continue
        ident = item.get("id")
        href = _norm_href(item.get("href") or "")
        if ident and href:
            out[str(ident)] = href
    return out


def _spine_idrefs(soup: BeautifulSoup) -> list[str]:
    spine = _find_local(soup, "spine")
    if spine is None:
        return []
    out: list[str] = []
    for ref in spine.find_all(True):
        if _local(ref.name) != "itemref":
            continue
        idref = ref.get("idref")
        if idref:
            out.append(str(idref))
    return out


def spine_hrefs(opf_text: str) -> list[str]:
    soup = BeautifulSoup(opf_text, "lxml-xml")
    id_to_href = _manifest_id_hrefs(soup)
    return [id_to_href[i] for i in _spine_idrefs(soup) if i in id_to_href]


def manifest_hrefs(opf_text: str) -> list[str]:
    soup = BeautifulSoup(opf_text, "lxml-xml")
    id_to_href = _manifest_id_hrefs(soup)
    manifest = _find_local(soup, "manifest")
    if manifest is None:
        return list(id_to_href.values())
    out: list[str] = []
    for item in manifest.find_all(True):
        if _local(item.name) != "item":
            continue
        href = _norm_href(item.get("href") or "")
        if href:
            out.append(href)
    return out


def spine_zip_paths(epub: Path) -> list[str]:
    """Document zip paths in OPF spine order (full paths, not basenames)."""
    with zipfile.ZipFile(epub) as z:
        opf = find_opf_in_zip(z)
        if not opf:
            return []
        soup = BeautifulSoup(decode_bytes(z.read(opf)), "lxml-xml")
        id_to_href = _manifest_id_hrefs(soup)
        out: list[str] = []
        for idref in _spine_idrefs(soup):
            href = id_to_href.get(idref)
            if href:
                out.append(_resolve_zip_path(opf, href))
        return out


def _guess_existing_dest(
    src_path: str,
    bi_set: set[str],
    bi_by_base: dict[str, list[str]],
    src_by_base: dict[str, list[str]],
    claimed: set[str],
) -> str | None:
    dest = None
    if src_path in bi_set:
        dest = src_path
    else:
        base = Path(src_path).name
        cands = bi_by_base.get(base, [])
        src_same = src_by_base.get(base, [])
        if len(src_same) == 1 and len(cands) == 1:
            dest = cands[0]
        elif len(src_same) == 1 and len(cands) > 1:
            dest = max(cands, key=lambda c: len(os.path.commonprefix([c[::-1], src_path[::-1]])))
    if dest is not None and dest in claimed:
        return None
    return dest


def dest_spine_order(
    src: Path,
    bilingual: Path,
    missing: list[tuple[str, str, str]] | None = None,
) -> list[str]:
    """Dest zip paths in source spine order, then leftover source HTML order."""
    src_paths = html_paths(src)
    bi_paths = html_paths(bilingual)
    bi_set = set(bi_paths)
    bi_by_base: dict[str, list[str]] = {}
    for p in bi_paths:
        bi_by_base.setdefault(Path(p).name, []).append(p)
    src_by_base: dict[str, list[str]] = {}
    for p in src_paths:
        src_by_base.setdefault(Path(p).name, []).append(p)
    claimed: set[str] = set()
    mapping: dict[str, str] = {}
    for src_path in src_paths:
        dest = _guess_existing_dest(src_path, bi_set, bi_by_base, src_by_base, claimed)
        if dest:
            claimed.add(dest)
            mapping[src_path] = dest
    if missing:
        for _base, src_path, dest in missing:
            mapping[src_path] = dest
    order: list[str] = []
    seen: set[str] = set()
    for src_path in spine_zip_paths(src) or src_paths:
        dest = mapping.get(src_path)
        if dest and dest not in seen:
            seen.add(dest)
            order.append(dest)
    for src_path in src_paths:
        dest = mapping.get(src_path)
        if dest and dest not in seen:
            seen.add(dest)
            order.append(dest)
    if missing:
        for _base, _src_path, dest in missing:
            if dest not in seen:
                seen.add(dest)
                order.append(dest)
    return order


def _en_block_count(html: str) -> int:
    soup = BeautifulSoup(html, "html.parser")
    n = 0
    for el in soup.find_all(TAGS):
        if el.find(TAGS):
            continue
        t = norm(el.get_text(" ", strip=True))
        if _skip_text(t) or has_zh_follow(el):
            continue
        n += 1
    return n


def _zh_block_count(html: str) -> int:
    soup = BeautifulSoup(html, "html.parser")
    n = 0
    for el in soup.find_all(TAGS):
        if el.find(TAGS):
            continue
        cn, _en = cn_en_counts(norm(el.get_text(" ", strip=True)))
        if cn >= 2:
            n += 1
    return n


def html_jobs(src: Path, bilingual: Path) -> list[tuple[str, str, str]]:
    """All English HTML files as (basename, src_zip_path, dest_zip_path)."""
    src_paths = html_paths(src)
    bi_paths = html_paths(bilingual)
    bi_set = set(bi_paths)
    bi_by_base: dict[str, list[str]] = {}
    for p in bi_paths:
        bi_by_base.setdefault(Path(p).name, []).append(p)
    src_by_base: dict[str, list[str]] = {}
    for p in src_paths:
        src_by_base.setdefault(Path(p).name, []).append(p)

    dest_prefix = "EPUB/Text/"
    if bi_paths:
        dest_prefix = str(Path(bi_paths[0]).parent).replace("\\", "/") + "/"
        if dest_prefix == "./":
            dest_prefix = ""

    jobs: list[tuple[str, str, str]] = []
    used_dests = set(bi_set)
    claimed: set[str] = set()
    with zipfile.ZipFile(src) as zs:
        for src_path in src_paths:
            base = Path(src_path).name
            src_html = decode_bytes(zs.read(src_path))
            if _en_block_count(src_html) == 0:
                continue
            dest = _guess_existing_dest(src_path, bi_set, bi_by_base, src_by_base, claimed)
            if dest is None:
                dest = dest_prefix + base
                if dest in used_dests or dest in claimed:
                    dest = dest_prefix + src_path.replace("/", "_")
                used_dests.add(dest)
            claimed.add(dest)
            jobs.append((base, src_path, dest))
    return jobs


def missing_files(src: Path, bilingual: Path) -> list[tuple[str, str, str]]:
    """Return (basename, src_zip_path, dest_zip_path) still needing translation.

    Files are keyed by their full zip-internal path so same-named HTML files in
    different directories do not shadow each other.
    """
    jobs = html_jobs(src, bilingual)
    bi_set = set(html_paths(bilingual))
    missing: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(bilingual) as zb:
        for job in jobs:
            _base, _src_path, dest = job
            if dest not in bi_set:
                missing.append(job)
                continue
            bi_html = decode_bytes(zb.read(dest))
            if _zh_block_count(bi_html) == 0:
                missing.append(job)
    return missing


def select_jobs(
    src: Path,
    bilingual: Path,
    *,
    only_files: str | list[str] | None = None,
    retranslate: str | list[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Jobs to translate: missing files, plus any --retranslate matches.

    `--only` limits the set to matching names (basename or zip path).
    """
    jobs = html_jobs(src, bilingual)
    only = parse_name_list(only_files)
    redo = parse_name_list(retranslate)
    miss = {(b, s, d) for b, s, d in missing_files(src, bilingual)}
    selected: list[tuple[str, str, str]] = []
    for job in jobs:
        base, src_path, _dest = job
        if only and not job_matches(src_path, base, only):
            continue
        if job in miss or job_matches(src_path, base, redo):
            selected.append(job)
    return selected


def _format_context(pairs: list[tuple[str, str]]) -> str:
    if not pairs:
        return ""
    lines = ["【上文对照，只供语气与用词参考，不要翻译或重复】"]
    for i, (en, zh) in enumerate(pairs, 1):
        lines.append(f"{i}. EN: {en}")
        lines.append(f"   ZH: {zh}")
    return "\n".join(lines) + "\n\n"


def _translate_batch(
    client: LlmClient,
    items: list[str],
    *,
    glossary: list[Term] | None = None,
    match_texts: list[str] | None = None,
    context: list[tuple[str, str]] | None = None,
) -> list[str]:
    terms = glossary or []
    protected, maps, hint, appearing = apply_glossary(items, terms, match_texts=match_texts)
    prefix = (hint + "\n\n") if hint else ""
    prefix += _format_context(context or [])
    lines = [f"{i}. {t}" for i, t in enumerate(protected, 1)]
    prompt = (
        prefix
        + f"把下面 {len(items)} 段英文译成简体中文。严格按这个格式输出：\n"
        "1. 译文\n2. 译文\n"
        "只输出编号列表，不要解释，不要合并或拆分段落。\n\n" + "\n".join(lines)
    )
    raw = client.chat(
        prompt,
        max_tokens=min(8000, 180 * len(items) + 400),
        timeout=90,
        source="\n".join(items),
    )
    parsed = parse_numbered_lines(raw)
    out: list[str] = []
    for i, src in enumerate(items, 1):
        zh = parsed.get(i, "")
        if zh:
            out.append(restore_text(zh, maps[i - 1], appearing))
            continue
        single = client.chat(
            prefix + "把这段英文译成简体中文，只输出译文：\n" + protected[i - 1],
            max_tokens=max(1000, 3 * len(src)),
            timeout=90,
            source=src,
        )
        single = " ".join(single.split()).strip().strip("\"“”")
        restored = restore_text(single, maps[i - 1], appearing) if single else ""
        out.append(restored if restored else src)
    return out


def batch_size_for(fname: str, short_names: set[str] | None = None, short_n: int = SHORT_BATCH) -> int:
    if is_short_batch_name(fname) or (short_names and Path(fname).name in short_names):
        return short_n
    return DEFAULT_BATCH


def batch_token_budget(
    fname: str,
    short_names: set[str] | None = None,
    *,
    default_tokens: int = DEFAULT_BATCH_TOKENS,
    short_tokens: int = SHORT_BATCH_TOKENS,
) -> int:
    if is_short_batch_name(fname) or (short_names and Path(fname).name in short_names):
        return short_tokens
    return default_tokens


def translate_html(
    html: str,
    *,
    client: LlmClient,
    log: Callable[[str], None],
    fname: str,
    batch_size: int | None = None,
    short_names: set[str] | None = None,
    short_n: int = SHORT_BATCH,
    glossary: list[Term] | None = None,
    max_blocks: int | None = None,
    use_context: bool = False,
    context_paragraphs: int = DEFAULT_CONTEXT_PARAGRAPHS,
    batch_tokens: int | None = None,
) -> str:
    soup = BeautifulSoup(html, "html.parser")
    nodes: list[Tag] = []
    texts: list[str] = []
    visibles: list[str] = []
    for el in soup.find_all(TAGS):
        if el.find(TAGS):
            continue
        text = norm(el.get_text(" ", strip=True))
        if _skip_text(text) or has_zh_follow(el):
            continue
        nodes.append(el)
        texts.append(text)
        visibles.append(visible_text(el, EXCLUDE))
    if max_blocks is not None:
        texts = texts[: max(0, max_blocks)]
        nodes = nodes[: len(texts)]
        visibles = visibles[: len(texts)]
    if batch_size is None:
        batch_size = batch_size_for(fname, short_names, short_n)
    if batch_tokens is None:
        env_tok = os.environ.get("BOOK_TRANSLATE_BATCH_TOKENS", "").strip()
        try:
            batch_tokens = int(env_tok) if env_tok else 0
        except ValueError:
            batch_tokens = 0
        if batch_tokens < 1:
            batch_tokens = batch_token_budget(fname, short_names)
    ranges = pack_batches(texts, token_budget=batch_tokens, max_items=batch_size)
    log(
        f"resume {fname}: {len(texts)} blocks batch_tokens={batch_tokens} "
        f"max_items={batch_size} batches={len(ranges)}"
    )
    if not texts:
        return str(soup)
    ctx: list[tuple[str, str]] = []
    ctx_limit = context_paragraphs if context_paragraphs > 0 else DEFAULT_CONTEXT_PARAGRAPHS
    done = 0
    for start, end in ranges:
        chunk = texts[start:end]
        log(f"resume {fname}: {start + 1}-{end}/{len(texts)}")
        zhs = _translate_batch(
            client,
            chunk,
            glossary=glossary,
            match_texts=visibles[start:end],
            context=ctx if use_context else None,
        )
        for el, src_text, zh in zip(nodes[start:end], chunk, zhs):
            if not zh or norm(zh) == norm(el.get_text(" ", strip=True)):
                continue
            cn, _en = cn_en_counts(zh)
            if cn < MIN_ZH_CHARS:
                # the model answered in English; treat as untranslated so the
                # EN+EN pair never lands in the book or the rolling context
                continue
            if use_context:
                ctx.append((src_text, zh))
                if len(ctx) > ctx_limit:
                    ctx = ctx[-ctx_limit:]
            if has_zh_follow(el):
                continue
            in_cell = el.name in ("td", "th")
            in_ol_li = el.name == "li" and el.parent is not None and el.parent.name == "ol"
            if in_cell or in_ol_li:
                # a sibling would break table layout / renumber the list
                el.append(soup.new_tag("br"))
                el.append(zh)
            else:
                new_el = copy(el)
                new_el.clear()
                new_el.attrs.pop("id", None)
                new_el.append(zh)
                el.insert_after(new_el)
            done += 1
        time.sleep(0.08)
    log(f"resume {fname}: inserted {done}")
    return str(soup)


def _start_tag_re(tag: str, attr: str, value: str) -> re.Pattern[str]:
    return re.compile(
        rf"<(?:[\w.-]+:)?{tag}\b(?=[^>]*\b{attr}=[\"']{re.escape(value)}[\"'])[^>]*>",
        re.I,
    )


def _insert_before_close(
    text: str, tag: str, snippet: str, log: Callable[[str], None] = print
) -> str:
    m = _CLOSE_TAG_RE[tag].search(text)
    if m is None:
        sc = _SELF_CLOSING_TAG_RE[tag].search(text)
        if sc is not None:
            # expand <manifest/> into <manifest></manifest> so items go inside
            expanded = f"<{sc.group(1)}{sc.group(2)}></{sc.group(1)}>"
            text = text[: sc.start()] + expanded + text[sc.end() :]
            m = _CLOSE_TAG_RE[tag].search(text)
    if m is None:
        # blindly appending would land the snippet after </package>: invalid XML
        log(f"warning: OPF has no <{tag}> element; skipped inserting {snippet.strip()!r}")
        return text
    return text[: m.start()] + snippet + text[m.start() :]


def _insert_before_attr(text: str, tag: str, attr: str, value: str, snippet: str) -> str | None:
    m = _start_tag_re(tag, attr, value).search(text)
    if not m:
        return None
    return text[: m.start()] + snippet + text[m.start() :]


def _unique_item_id(opf_text: str, dest: str) -> str:
    ident = "bt-" + re.sub(r"[^A-Za-z0-9_.-]", "-", dest.replace("\\", "/"))
    base = ident
    n = 2
    while re.search(rf'\bid="{re.escape(ident)}"', opf_text):
        ident = f"{base}-{n}"
        n += 1
    return ident


def _order_hrefs(order_dests: list[str], opf_zip: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for dest in order_dests:
        href = href_for_dest(opf_zip, dest)
        if href and href not in seen:
            seen.add(href)
            out.append(href)
    return out


def _neighbor_href(my_href: str, order_hrefs: list[str], present: set[str]) -> str | None:
    """First desired-order successor that is already present."""
    try:
        my_i = order_hrefs.index(my_href)
    except ValueError:
        return None
    nxt = None
    best_j: int | None = None
    for href in present:
        try:
            j = order_hrefs.index(href)
        except ValueError:
            continue
        if j > my_i and (best_j is None or j < best_j):
            best_j = j
            nxt = href
    return nxt


def _id_for_href(id_to_href: dict[str, str], href: str) -> str | None:
    want = _norm_href(href)
    for ident, h in id_to_href.items():
        if _norm_href(h) == want:
            return ident
    return None


def _infer_opf_zip(dest_paths: list[str]) -> str:
    if not dest_paths:
        return "content.opf"
    dest = dest_paths[0].replace("\\", "/")
    if dest.startswith("EPUB/"):
        return "EPUB/content.opf"
    if dest.startswith("OEBPS/"):
        return "OEBPS/content.opf"
    return "content.opf"


def _ordered_dests(dest_paths: list[str], order_dests: list[str] | None) -> list[str]:
    if not order_dests:
        return list(dest_paths)
    rank = {d.replace("\\", "/"): i for i, d in enumerate(order_dests)}
    return sorted(dest_paths, key=lambda d: rank.get(d.replace("\\", "/"), len(rank)))


def _patch_opf(
    opf_text: str,
    dest_paths: list[str],
    *,
    opf_zip_path: str = "",
    order_dests: list[str] | None = None,
) -> str:
    """Insert missing manifest/spine entries next to their source-order neighbors."""
    if not dest_paths:
        return opf_text
    opf_zip = opf_zip_path.replace("\\", "/") if opf_zip_path else _infer_opf_zip(dest_paths)
    order_hrefs = _order_hrefs(order_dests or [], opf_zip)
    soup0 = BeautifulSoup(opf_text, "lxml-xml")
    id_to_href0 = _manifest_id_hrefs(soup0)
    present_spine = {id_to_href0[i] for i in _spine_idrefs(soup0) if i in id_to_href0}

    for dest in _ordered_dests(dest_paths, order_dests):
        dest = dest.replace("\\", "/")
        opf_href = href_for_dest(opf_zip, dest)
        soup = BeautifulSoup(opf_text, "lxml-xml")
        id_to_href = _manifest_id_hrefs(soup)
        spine_ids = set(_spine_idrefs(soup))
        existing_id = _id_for_href(id_to_href, opf_href)
        nxt = _neighbor_href(opf_href, order_hrefs, present_spine)

        if existing_id is None:
            ident = _unique_item_id(opf_text, dest)
            item = (
                f'    <item href="{_xml_attr(opf_href)}" id="{_xml_attr(ident)}" '
                f'media-type="application/xhtml+xml"/>\n'
            )
            inserted = None
            if nxt:
                nid = _id_for_href(id_to_href, nxt)
                if nid:
                    inserted = _insert_before_attr(opf_text, "item", "id", nid, item)
                if inserted is None:
                    inserted = _insert_before_attr(opf_text, "item", "href", nxt, item)
            opf_text = inserted if inserted is not None else _insert_before_close(opf_text, "manifest", item)
        else:
            ident = existing_id

        if ident not in spine_ids:
            itemref = f'    <itemref idref="{_xml_attr(ident)}"/>\n'
            inserted = None
            if nxt:
                nid = _id_for_href(id_to_href, nxt)
                if nid:
                    inserted = _insert_before_attr(opf_text, "itemref", "idref", nid, itemref)
            opf_text = inserted if inserted is not None else _insert_before_close(opf_text, "spine", itemref)
        present_spine.add(opf_href)
    return opf_text


def splice(
    bilingual: Path,
    updates: dict[str, str],
    out: Path,
    *,
    order_dests: list[str] | None = None,
) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="file-resume-"))
    try:
        with zipfile.ZipFile(bilingual) as zin:
            zin.extractall(tmp)
        for rel, html in updates.items():
            dest = tmp / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            write_text_utf8(dest, html)
        opf = find_opf_in_tree(tmp)
        if opf is not None:
            opf_rel = opf.relative_to(tmp).as_posix()
            write_text_utf8(
                opf,
                _patch_opf(
                    read_text(opf),
                    list(updates),
                    opf_zip_path=opf_rel,
                    order_dests=order_dests,
                ),
            )
        rezip(tmp, out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _append_members(prev: Path, out: Path, entries: list[tuple[str, bytes]]) -> None:
    """Cheap per-file checkpoint: copy the previous EPUB and append the changed
    members. Copying compressed bytes is far cheaper than splice()'s
    extractall+rezip, which inflates and re-deflates every member of the book
    for each finished file (O(files^2) over a whole resume).

    Appended names shadow earlier duplicates for zip readers, so the
    checkpoint stays a valid resume base; the single clean rezip at the end of
    resume_missing removes the shadowed duplicates. Writes go to a temp file
    that atomically replaces `out`, so a crash mid-checkpoint never corrupts
    the previous checkpoint.
    """
    tmp = out.with_name(out.name + ".ckpt")
    try:
        shutil.copy2(prev, tmp)
        with warnings.catch_warnings():
            # shadowing an existing member is the whole point here
            warnings.filterwarnings("ignore", message="Duplicate name:", category=UserWarning)
            with zipfile.ZipFile(tmp, "a", zipfile.ZIP_DEFLATED) as z:
                for rel, data in entries:
                    z.writestr(rel, data)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _fingerprint(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_state(
    state_path: Path,
    base: Path,
    done: list[str],
    src: Path | None = None,
    *,
    src_sha1: str | None = None,
    attempts: dict[str, int] | None = None,
) -> None:
    """`src_sha1` may be passed pre-computed; src never changes during a run,
    so callers in per-file loops should hash it once instead of every file."""
    payload: dict = {"base_sha1": _fingerprint(base), "done": done}
    if src_sha1 is None and src is not None:
        src_sha1 = _fingerprint(src)
    if src_sha1 is not None:
        payload["src_sha1"] = src_sha1
    if attempts:
        payload["attempts"] = attempts
    tmp = state_path.with_name(state_path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, state_path)


def checkpoint_matches_src(state_path: Path, src: Path) -> bool:
    if not state_path.is_file():
        return False
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(data, dict) and data.get("src_sha1") == _fingerprint(src)


def resume_missing(
    src: Path,
    bilingual: Path,
    out: Path,
    *,
    client: LlmClient,
    log: Callable[[str], None],
    progress: Callable[[str], None] | None = None,
    state_path: Path | None = None,
    short_names: set[str] | None = None,
    short_n: int = SHORT_BATCH,
    glossary: list[Term] | None = None,
    workers: int | None = None,
    only_files: str | list[str] | None = None,
    retranslate: str | list[str] | None = None,
    test_num: int | None = None,
    use_context: bool = False,
    context_paragraphs: int = DEFAULT_CONTEXT_PARAGRAPHS,
    batch_tokens: int | None = None,
) -> tuple[Path, list[str]]:
    """Translate leftover source HTML files and splice them into bilingual EPUB.

    Completion is judged from content (missing_files); the done list only
    records files whose pass actually landed Chinese in this exact base
    (bound via base_sha1). A pass that inserted no Chinese at all (e.g. the
    model kept answering in English) leaves the file off the done list so a
    later run retries it; per-file attempts are tracked in the state and after
    MAX_FILE_ATTEMPTS futile passes the file is marked done anyway so it
    cannot burn tokens forever.

    `workers` defaults to BOOK_TRANSLATE_RESUME_WORKERS (1 = serial). Each
    file is still checkpointed into `out` on disk before it is marked done;
    the book is extracted once into a long-lived working tree and rezipped
    cleanly once at the end instead of being fully re-zipped per file.
    `--test` / `test_num` does not write resume-done (partial files stay retryable).
    """
    if short_names is None:
        short_names = short_batch_basenames(src)
    if test_num is not None and test_num < 1:
        test_num = None
    workers = 1 if test_num is not None else resume_worker_count(workers)
    write_state = state_path is not None and test_num is None
    todo = select_jobs(src, bilingual, only_files=only_files, retranslate=retranslate)
    done: list[str] = []
    attempts: dict[str, int] = {}
    if write_state and state_path and state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            data = None
        if (
            isinstance(data, dict)
            and data.get("base_sha1") == _fingerprint(bilingual)
            and (not data.get("src_sha1") or data.get("src_sha1") == _fingerprint(src))
        ):
            done = [str(x) for x in data.get("done") or []]
            raw_attempts = data.get("attempts")
            if isinstance(raw_attempts, dict):
                attempts = {str(k): int(v) for k, v in raw_attempts.items() if isinstance(v, int)}
        elif data is not None:
            log("resume state stale or legacy format; ignoring its done list")
    redo = parse_name_list(retranslate)
    done_set = {p for p in done if not job_matches(p, Path(p).name, redo)}
    done = [p for p in done if p in done_set]
    attempts = {p: n for p, n in attempts.items() if not job_matches(p, Path(p).name, redo)}
    todo = [t for t in todo if t[1] not in done_set]
    log(
        f"file-resume missing={len(todo)} already_done={len(done)} "
        f"workers={workers}"
        + (f" test_num={test_num}" if test_num is not None else "")
    )
    if not todo:
        if bilingual.resolve() != out.resolve():
            shutil.copy2(bilingual, out)
        return out, done

    order_dests = dest_spine_order(src, bilingual, todo)
    remaining = test_num
    src_sha1 = _fingerprint(src) if write_state else None

    def _translate_one(
        base: str, src_html: str, log_fn: Callable[[str], None], max_blocks: int | None
    ) -> str:
        return translate_html(
            src_html,
            client=client,
            log=log_fn,
            fname=base,
            short_names=short_names,
            short_n=short_n,
            glossary=glossary,
            max_blocks=max_blocks,
            use_context=use_context,
            context_paragraphs=context_paragraphs,
            batch_tokens=batch_tokens,
        )

    # One long-lived working tree for the whole run: extract the book once,
    # write each finished file into it, and rezip once at the end. Per-file
    # checkpoints append the changed members to a copy of the previous EPUB
    # (_append_members) instead of re-deflating the whole book per file.
    workdir = Path(tempfile.mkdtemp(prefix="file-resume-"))
    committed = 0
    current = bilingual

    def _record_result(
        base: str, src_path: str, new_html: str, log_fn: Callable[[str], None]
    ) -> None:
        """Mark done only when Chinese actually landed; count futile passes."""
        if _zh_block_count(new_html) > 0:
            done.append(src_path)
            attempts.pop(src_path, None)
            return
        n = attempts.get(src_path, 0) + 1
        attempts[src_path] = n
        if n >= MAX_FILE_ATTEMPTS:
            log_fn(f"resume {base}: no Chinese inserted after {n} attempts; giving up")
            done.append(src_path)
        else:
            log_fn(f"resume {base}: no Chinese inserted; will retry ({n}/{MAX_FILE_ATTEMPTS})")

    try:
        with zipfile.ZipFile(bilingual) as zin:
            zin.extractall(workdir)
        opf_path = find_opf_in_tree(workdir)
        opf_rel = str(opf_path.relative_to(workdir)).replace("\\", "/") if opf_path else ""

        def _commit(dest: str, html: str) -> None:
            """Write into the working tree, then checkpoint `out` on disk."""
            nonlocal committed, current
            target = workdir / dest
            target.parent.mkdir(parents=True, exist_ok=True)
            write_text_utf8(target, html)
            entries = [(dest, target.read_bytes())]
            if opf_path is not None:
                write_text_utf8(
                    opf_path,
                    _patch_opf(
                        read_text(opf_path),
                        [dest],
                        opf_zip_path=opf_rel,
                        order_dests=order_dests,
                    ),
                )
                entries.append((opf_rel, opf_path.read_bytes()))
            _append_members(current, out, entries)
            current = out
            committed += 1

        if workers <= 1:
            with zipfile.ZipFile(src) as zs:
                for i, (base, src_path, dest) in enumerate(todo, 1):
                    if remaining is not None and remaining <= 0:
                        break
                    if progress:
                        progress(f"resume:{base} {i}/{len(todo)}")
                    src_html = decode_bytes(zs.read(src_path))
                    n_src = _en_block_count(src_html)
                    cap = remaining
                    new_html = _translate_one(base, src_html, log, cap)
                    # checkpoint first, then record state, so an interruption can
                    # never mark a file done whose translation is not in the output
                    _commit(dest, new_html)
                    if remaining is not None:
                        remaining -= n_src if cap is None else min(n_src, cap)
                    else:
                        _record_result(base, src_path, new_html, log)
                        if write_state and state_path:
                            _write_state(
                                state_path, out, done, src, src_sha1=src_sha1, attempts=attempts
                            )
            return out, done

        htmls: dict[str, str] = {}
        with zipfile.ZipFile(src) as zs:
            for _base, src_path, _dest in todo:
                htmls[src_path] = decode_bytes(zs.read(src_path))
        log_lock = threading.Lock()

        def tlog(msg: str) -> None:
            with log_lock:
                log(msg)

        # Raise the gap only on this client's throttle for the parallel section;
        # other clients in the process are unaffected.
        throttle = client.throttle
        prev_gap = throttle.min_interval
        throttle.min_interval = max(prev_gap, _RESUME_CHAT_GAP_SEC)
        errors: list[BaseException] = []
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [
                    pool.submit(_translate_one, base, htmls[src_path], tlog, None)
                    for base, src_path, _dest in todo
                ]
                pending = {fut: item for fut, item in zip(futs, todo)}
                # workers only translate; this thread is the single consumer
                # that commits results, so no splice lock is needed
                for fut in as_completed(pending):
                    base, src_path, dest = pending[fut]
                    try:
                        new_html = fut.result()
                    except BaseException as e:  # noqa: BLE001 - surface after other commits
                        errors.append(e)
                        tlog(f"resume {base} failed: {e}")
                        continue
                    if progress:
                        progress(f"resume:{base} {len(done) + 1}/{len(todo)}")
                    _commit(dest, new_html)
                    _record_result(base, src_path, new_html, tlog)
                    if write_state and state_path:
                        _write_state(
                            state_path, out, done, src, src_sha1=src_sha1, attempts=attempts
                        )
        finally:
            throttle.min_interval = prev_gap
        if errors:
            raise errors[0]
        return out, done
    finally:
        try:
            if committed:
                # one clean rezip: the deliverable keeps mimetype first/stored
                # and has none of the checkpoint's shadowed duplicate members
                rezip(workdir, out)
                if write_state and state_path and (done or attempts):
                    _write_state(
                        state_path, out, done, src, src_sha1=src_sha1, attempts=attempts
                    )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
