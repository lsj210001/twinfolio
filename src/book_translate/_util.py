"""Shared helpers: text normalization, CN/EN counting, LLM chat with retry,
encoding-aware read/write, and atomic EPUB re-zipping.
"""
from __future__ import annotations

import codecs
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

from bs4 import Tag

# ---------------------------------------------------------------------------
# text helpers


def norm(s: str) -> str:
    """Collapse whitespace/nbsp/zwsp and fold curly quotes to straight ones."""
    s = (s or "").replace("\xa0", " ").replace("\u200b", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def cn_en_counts(text: str) -> tuple[int, int]:
    cn = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    en = sum(1 for ch in text if "a" <= ch.lower() <= "z")
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
# paragraph-level LLM cache + chat with retry


class TokenUsage:
    """Process-wide totals from chat/completions calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.calls = 0

    def reset(self) -> None:
        with self._lock:
            self.input_tokens = 0
            self.output_tokens = 0
            self.total_tokens = 0
            self.calls = 0

    def add_from_response(self, data: object) -> None:
        if not isinstance(data, dict):
            return
        usage = data.get("usage")
        if not isinstance(usage, dict):
            return
        inp = _usage_int(usage.get("prompt_tokens"), usage.get("input_tokens"))
        out = _usage_int(usage.get("completion_tokens"), usage.get("output_tokens"))
        tot = _usage_int(usage.get("total_tokens"))
        if tot <= 0:
            tot = inp + out
        if inp == 0 and out == 0 and tot == 0:
            return
        with self._lock:
            self.input_tokens += inp
            self.output_tokens += out
            self.total_tokens += tot
            self.calls += 1

    def format_line(self) -> str:
        with self._lock:
            return (
                f"llm tokens input={self.input_tokens} "
                f"output={self.output_tokens} total={self.total_tokens} "
                f"calls={self.calls}"
            )


class ChatThrottle:
    """Serialize outbound chat/completions so resume workers cannot stampede the API."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.min_interval = 0.0
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            gap = self._last + self.min_interval - now
            if gap > 0:
                time.sleep(gap)
            self._last = time.monotonic()


CHAT_THROTTLE = ChatThrottle()


def _usage_int(*values: object) -> int:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


TOKEN_USAGE = TokenUsage()


class LlmCache:
    """jsonl cache. Empty or half-written lines are ignored.

    Writes replace the whole file via a temp path so a crash cannot leave a
    torn jsonl. A sibling `.lock` file serializes writers.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._mem: dict[str, str] = {}
        # In-process lock for _mem: resume workers share one instance, and a
        # put() must not mutate _mem while another thread iterates it. The
        # `.lock` file below only serializes writers across processes.
        self._mem_lock = threading.Lock()
        self._load()

    @staticmethod
    def make_key(model: str, prompt: str, source: str = "") -> str:
        blob = f"{model}\n{prompt}\n{norm(source)}"
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            if self.path.stat().st_size == 0:
                return
        except OSError:
            return
        try:
            raw = self.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return
        if not raw.strip():
            return
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            key, response = obj.get("key"), obj.get("response")
            if isinstance(key, str) and isinstance(response, str) and key and response:
                self._mem[key] = response

    def get(self, model: str, prompt: str, source: str = "") -> str | None:
        key = self.make_key(model, prompt, source)
        with self._mem_lock:
            return self._mem.get(key)

    def put(self, model: str, prompt: str, response: str, source: str = "") -> None:
        if not response:
            return
        key = self.make_key(model, prompt, source)
        with self._mem_lock:
            self._mem[key] = response
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with _exclusive_lock(self.path.with_name(self.path.name + ".lock")):
            disk = LlmCache(self.path) if self.path.is_file() else None
            # Merge and snapshot under the thread lock so no other thread can
            # resize _mem mid-iteration; serialize the snapshot outside it.
            with self._mem_lock:
                if disk is not None:
                    disk._mem.update(self._mem)
                    self._mem = disk._mem
                snapshot = dict(self._mem)
            payload = "".join(
                json.dumps({"key": k, "response": v}, ensure_ascii=False) + "\n"
                for k, v in snapshot.items()
            )
            try:
                tmp.write_text(payload, encoding="utf-8")
                os.replace(tmp, self.path)
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass


def _chat_once(api: dict[str, str], body: dict, timeout: int) -> tuple[str, str]:
    req = urllib.request.Request(
        api["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": "Bearer " + api["api_key"],
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {raw[:240]}") from e
    choice = (data.get("choices") or [{}])[0]
    content = ((choice.get("message") or {}).get("content") or "").strip()
    finish = choice.get("finish_reason") or ""
    TOKEN_USAGE.add_from_response(data)
    return content, finish


def chat(
    api: dict[str, str],
    prompt: str,
    *,
    max_tokens: int = 4000,
    temperature: float = 0.2,
    timeout: int = 180,
    retries: int = 3,
    log: Callable[[str], None] | None = None,
    cache: LlmCache | None = None,
    source: str = "",
) -> str:
    """One-shot chat completion with exponential-backoff retries.

    If the reply was cut off (finish_reason == "length"), retry once with a
    doubled token budget and keep the longer answer.

    `cache` is a paragraph-level store keyed by sha1(model + prompt +
    normalized source). Changing the model or prompt is a miss.
    """
    model = api.get("model") or ""
    if cache is not None:
        hit = cache.get(model, prompt, source)
        if hit is not None:
            return hit
    body = {
        "model": api["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if api.get("reasoning_effort"):
        body["reasoning_effort"] = api["reasoning_effort"]
    last: Exception | None = None
    for attempt in range(retries):
        try:
            CHAT_THROTTLE.wait()
            content, finish = _chat_once(api, body, timeout)
            if finish == "length":
                bumped = dict(body, max_tokens=min(max_tokens * 2, 16000))
                try:
                    CHAT_THROTTLE.wait()
                    content2, _ = _chat_once(api, bumped, timeout)
                    if len(content2) > len(content):
                        content = content2
                except Exception:
                    pass
            if cache is not None and content:
                cache.put(model, prompt, content, source)
            return content
        except Exception as e:  # noqa: BLE001 - network layer, retry everything
            last = e
            if log:
                log(f"chat attempt {attempt + 1}/{retries} failed: {e}")
            if attempt + 1 < retries:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"chat failed after {retries} attempts: {last}") from last


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
