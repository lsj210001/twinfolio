"""Shared helpers: text normalization, CN/EN counting, encoding-aware
read/write, and atomic EPUB re-zipping. LLM chat lives in `llm.py`.
"""
from __future__ import annotations

import codecs
import os
import re
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

from bs4 import Tag

# ---------------------------------------------------------------------------
# text helpers


_WS_RE = re.compile(r"\s+")
_CJK = re.compile(r"[\u4e00-\u9fff]+")
# U+0130 (İ) and U+212A (KELVIN SIGN) are the only non-ASCII codepoints whose
# str.lower() lands in a-z, so the previous per-character
# `"a" <= ch.lower() <= "z"` test counted them; keep exact parity.
_LATIN = re.compile(r"[A-Za-z\u0130\u212a]+")


def norm(s: str) -> str:
    """Collapse whitespace/nbsp/zwsp and fold curly quotes to straight ones."""
    s = (s or "").replace("\xa0", " ").replace("\u200b", "")
    s = _WS_RE.sub(" ", s).strip()
    return s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def cn_en_counts(text: str) -> tuple[int, int]:
    cn = sum(map(len, _CJK.findall(text)))
    en = sum(map(len, _LATIN.findall(text)))
    return cn, en


def next_tag(el) -> Tag | None:
    cur = el.next_sibling
    while cur is not None and not isinstance(cur, Tag):
        cur = cur.next_sibling
    return cur if isinstance(cur, Tag) else None


def has_zh_follow(el) -> bool:
    """True if the next sibling of the same tag is Chinese-dominant."""
    nxt = next_tag(el)
    if nxt is None or nxt.name != el.name:
        return False
    cn, en = cn_en_counts(norm(nxt.get_text(" ", strip=True)))
    return cn >= 2 and cn >= en


_NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)\s*[.、)）．]\s*(.+)$")


def parse_numbered_lines(raw: str) -> dict[int, str]:
    """Parse `1. text` numbered lists, accepting "1、" / "1)" / "1．" variants.

    Un-numbered lines are joined onto the previous item (multi-line answers),
    so long translations are not silently truncated to their first line.
    """
    found: dict[int, list[str]] = {}
    last_idx: int | None = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _NUMBERED_LINE_RE.match(line)
        if m:
            last_idx = int(m.group(1))
            found[last_idx] = [m.group(2).strip()]
        elif last_idx is not None:
            found[last_idx].append(line)
    return {i: " ".join(parts).strip().strip("\"“”") for i, parts in found.items()}


# ---------------------------------------------------------------------------
# encoding-aware IO

_XML_ENC_BYTES_RE = re.compile(rb'<\?xml[^>]*?encoding=["\']([A-Za-z0-9_.\-]+)["\']', re.I)
_XML_ENC_TEXT_RE = re.compile(r'^(\s*<\?xml[^>]*?encoding=["\'])([^"\']+)(["\'])', re.I)
# matches both <meta charset="..."> and
# <meta http-equiv="Content-Type" content="text/html; charset=...">
_META_CHARSET_BYTES_RE = re.compile(rb'<meta\b[^>]*?charset\s*=\s*["\']?\s*([A-Za-z0-9_.\-]+)', re.I)
_META_CHARSET_TEXT_RE = re.compile(r'(<meta\b[^>]*?charset\s*=\s*["\']?\s*)([A-Za-z0-9_.\-]+)', re.I)

# Legacy books routinely declare a subset of the encoding actually used;
# decode with the superset so those extra characters survive.
_ENCODING_ALIASES = {
    "gb2312": "gbk",
    "gb-2312": "gbk",
    "iso-8859-1": "cp1252",
    "iso8859-1": "cp1252",
    "latin-1": "cp1252",
    "latin1": "cp1252",
    "windows-1252": "cp1252",
    "us-ascii": "ascii",
    "ucs-2": "utf-16",
    "ucs2": "utf-16",
}

# Single-byte codecs that decode almost any byte sequence without error, so a
# mislabelled UTF-8 file would silently turn into mojibake. For these we try
# strict UTF-8 first and only fall back to the declared codec when that fails.
_LATIN_LIKE = {"cp1252", "ascii"}


def _declared_encoding(raw: bytes) -> str | None:
    """Sniff the declared encoding: XML declaration first, then HTML meta."""
    m = _XML_ENC_BYTES_RE.search(raw[:1024])
    if m is None:
        m = _META_CHARSET_BYTES_RE.search(raw[:2048])
    if m is None:
        return None
    name = m.group(1).decode("ascii", errors="replace").strip().lower()
    return _ENCODING_ALIASES.get(name, name)


def _decode_utf16_no_bom(raw: bytes) -> str | None:
    """Pick utf-16-le/be for BOM-less data by which half holds the NUL bytes."""
    sample = raw[:2048]
    even_nuls = sample[0::2].count(0)
    odd_nuls = sample[1::2].count(0)
    if odd_nuls > even_nuls * 2:
        return raw.decode("utf-16-le", errors="replace")
    if even_nuls > odd_nuls * 2:
        return raw.decode("utf-16-be", errors="replace")
    return None


def _decode_bytes_impl(raw: bytes) -> str:
    if raw.startswith(codecs.BOM_UTF8):
        return raw.decode("utf-8-sig", errors="replace")
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        return raw.decode("utf-16", errors="replace")
    # BOM-less UTF-16 markup: '<' interleaved with a NUL byte
    if raw.startswith(b"<\x00"):
        return raw.decode("utf-16-le", errors="replace")
    if raw.startswith(b"\x00<"):
        return raw.decode("utf-16-be", errors="replace")
    enc = _declared_encoding(raw)
    if enc:
        if enc == "utf-16":
            text = _decode_utf16_no_bom(raw)
            if text is not None:
                return text
        else:
            if enc in _LATIN_LIKE:
                try:
                    return raw.decode("utf-8")
                except UnicodeDecodeError:
                    pass
            try:
                return raw.decode(enc)
            except (LookupError, UnicodeDecodeError, ValueError):
                pass
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def decode_bytes(raw: bytes, log: Callable[[str], None] = print) -> str:
    """Decode HTML/XML bytes: BOM, then the declared encoding (XML declaration
    or HTML meta charset), then UTF-8.

    Never raises; the last resort is utf-8 with errors="replace" so damage
    stays visible instead of being silently dropped. An unusual share of
    U+FFFD/NUL in the result triggers a warning.
    """
    text = _decode_bytes_impl(raw)
    if text:
        bad = text.count("\ufffd") + text.count("\x00")
        if bad / len(text) > 0.01:
            log(
                f"warning: decoded text has {bad}/{len(text)} replacement/NUL "
                "characters; the source encoding is likely misdeclared"
            )
    return text


def read_text(path: Path) -> str:
    return decode_bytes(Path(path).read_bytes())


def ensure_utf8_declaration(text: str) -> str:
    """Rewrite non-UTF-8 XML/meta charset declarations since we always write
    UTF-8 back."""
    m = _XML_ENC_TEXT_RE.match(text)
    if m and m.group(2).lower() not in {"utf-8", "utf8"}:
        text = text[: m.start(2)] + "utf-8" + text[m.end(2) :]

    def _meta_sub(mm: re.Match[str]) -> str:
        if mm.group(2).lower() in {"utf-8", "utf8"}:
            return mm.group(0)
        return mm.group(1) + "utf-8"

    return _META_CHARSET_TEXT_RE.sub(_meta_sub, text)


def write_text_utf8(path: Path, text: str) -> None:
    Path(path).write_text(ensure_utf8_declaration(text), encoding="utf-8")


# ---------------------------------------------------------------------------
# exclusive lock (Windows msvcrt / POSIX fcntl)


@contextmanager
def _exclusive_lock(lock_path: Path) -> Iterator[None]:
    """Serialize writers. Released when the process dies."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                fh.write(b"\0")
                fh.flush()
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        if locked:
            try:
                if os.name == "nt":
                    import msvcrt

                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        fh.close()


# ---------------------------------------------------------------------------
# zip helpers


def rezip(src_dir: Path, out: Path) -> None:
    """Zip an extracted EPUB tree (mimetype first, stored) and atomically
    replace `out`, so an interrupted run never leaves a corrupt half-file."""
    src_dir = Path(src_dir)
    out = Path(out)
    tmp_zip = out.with_name(out.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp_zip, "w") as zout:
            mt = src_dir / "mimetype"
            if mt.exists():
                zout.write(mt, "mimetype", compress_type=zipfile.ZIP_STORED)
            for f in sorted(src_dir.rglob("*")):
                if not f.is_file():
                    continue
                rel = f.relative_to(src_dir).as_posix()
                if rel == "mimetype":
                    continue
                zout.write(f, rel, compress_type=zipfile.ZIP_DEFLATED)
        os.replace(tmp_zip, out)
    finally:
        if tmp_zip.exists():
            try:
                tmp_zip.unlink()
            except OSError:
                pass


_WIN_BAD_CHARS = re.compile(r'[:*?"<>|]')


def warn_zip_member_names(path: Path, log: Callable[[str], None] = print) -> None:
    """Warn about zip member names that are unsafe on Windows (illegal chars,
    or names differing only by case). Warning only; extraction proceeds."""
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
    except Exception:
        return
    seen: dict[str, str] = {}
    for n in names:
        if _WIN_BAD_CHARS.search(n):
            log(f"warning: zip member has Windows-unsafe characters: {n!r}")
        low = n.lower()
        if low in seen and seen[low] != n:
            log(f"warning: zip members differ only by case: {seen[low]!r} vs {n!r}")
        else:
            seen.setdefault(low, n)
