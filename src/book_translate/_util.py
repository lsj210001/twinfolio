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


def decode_bytes(raw: bytes) -> str:
    """Decode HTML/XML bytes: BOM first, then the XML declaration, then UTF-8.

    Never raises; the last resort is utf-8 with errors="replace" so damage
    stays visible instead of being silently dropped.
    """
    if raw.startswith(codecs.BOM_UTF8):
        return raw.decode("utf-8-sig", errors="replace")
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        return raw.decode("utf-16", errors="replace")
    m = _XML_ENC_BYTES_RE.search(raw[:512])
    if m:
        enc = m.group(1).decode("ascii", errors="replace")
        try:
            return raw.decode(enc)
        except (LookupError, UnicodeDecodeError, ValueError):
            pass
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def read_text(path: Path) -> str:
    return decode_bytes(Path(path).read_bytes())


def ensure_utf8_declaration(text: str) -> str:
    """Rewrite a non-UTF-8 XML declaration since we always write UTF-8 back."""
    m = _XML_ENC_TEXT_RE.match(text)
    if m and m.group(2).lower() not in {"utf-8", "utf8"}:
        text = text[: m.start(2)] + "utf-8" + text[m.end(2) :]
    return text


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
