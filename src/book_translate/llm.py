"""LLM chat client: immutable config, per-instance throttle/usage/cache,
and layered network error handling.

One `LlmClient` is built per `pipeline.translate` run, so two concurrent
runs in the same process cannot reset each other's token totals or rewrite
each other's throttle interval (which module-level singletons used to allow).

Errors are typed so the retry loop can tell transient failures (429, 5xx,
timeouts, connection errors) from permanent ones (other 4xx), which are
raised immediately instead of burning retries against a rejected request.
"""
from __future__ import annotations

import email.utils
import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ._util import _exclusive_lock, norm


@dataclass(frozen=True)
class LlmConfig:
    """Immutable endpoint configuration for one OpenAI-compatible API."""

    base_url: str
    api_key: str
    model: str
    reasoning_effort: str = "low"


class LlmError(RuntimeError):
    """Base for LLM transport/protocol failures.

    Subclasses RuntimeError so callers that caught the old opaque
    RuntimeError keep working.
    """


class LlmHttpError(LlmError):
    """HTTP-level failure with its status code preserved.

    Also used when a 200 response carries an error payload that names an
    HTTP-like status code (common with gateways).
    """

    def __init__(self, code: int, body: str = "", retry_after: float | None = None):
        super().__init__(f"HTTP {code}: {body[:240]}")
        self.code = code
        self.body = body
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.code == 429 or self.code >= 500


class LlmProtocolError(LlmError):
    """Malformed or error-carrying response without a usable status code
    (non-JSON body, or a 200 whose error payload names no HTTP code).
    Treated as transient: gateways emit these for upstream hiccups."""


def _is_retryable(exc: BaseException) -> bool:
    """Transient failures only: 429/5xx, gateway protocol noise, timeouts,
    and connection-level errors. Everything else (400/401/403/404, coding
    bugs) fails fast."""
    if isinstance(exc, LlmHttpError):
        return exc.retryable
    if isinstance(exc, LlmProtocolError):
        return True
    if isinstance(exc, urllib.error.HTTPError):  # pragma: no cover - normally wrapped
        return exc.code == 429 or exc.code >= 500
    # URLError covers DNS/connection failures; TimeoutError and other
    # OSErrors can escape response.read() unwrapped.
    return isinstance(exc, (urllib.error.URLError, TimeoutError, OSError))


def _parse_retry_after(headers: object) -> float | None:
    """Seconds to wait from a Retry-After header: delta-seconds or HTTP-date."""
    get = getattr(headers, "get", None)
    value = get("Retry-After") if callable(get) else None
    if not value:
        return None
    value = str(value).strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            dt = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        seconds = (dt - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, seconds) if seconds == seconds else None  # reject NaN


def _error_code(err: dict) -> int | None:
    """HTTP-like status code from an error payload, if it names one."""
    for key in ("code", "status", "status_code"):
        try:
            code = int(err.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if 100 <= code <= 599:
            return code
    return None


def _backoff_seconds(attempt: int, retry_after: float | None = None) -> float:
    """Exponential backoff with jitter; honors Retry-After when it is longer."""
    base = 2.0 * (attempt + 1)
    if retry_after is not None and retry_after > 0:
        base = max(base, retry_after)
    return base + random.uniform(0.0, base / 4)


class TokenUsage:
    """Token totals from chat/completions calls, scoped to one client."""

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

    def __init__(self, min_interval: float = 0.0) -> None:
        self._lock = threading.Lock()
        self.min_interval = min_interval
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


def _usage_int(*values: object) -> int:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


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


class LlmClient:
    """One LLM endpoint plus per-instance cache, throttle, and token usage.

    Nothing is shared between clients unless explicitly injected, so the
    library is reentrant: concurrent translate runs each own their state.
    """

    def __init__(
        self,
        config: LlmConfig,
        *,
        cache: LlmCache | None = None,
        throttle: ChatThrottle | None = None,
        usage: TokenUsage | None = None,
    ) -> None:
        self.config = config
        self.cache = cache
        self.throttle = throttle if throttle is not None else ChatThrottle()
        self._usage = usage if usage is not None else TokenUsage()

    @property
    def usage(self) -> TokenUsage:
        return self._usage

    def _chat_once(self, body: dict, timeout: int) -> tuple[str, str]:
        req = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": "Bearer " + self.config.api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise LlmHttpError(
                e.code, detail, retry_after=_parse_retry_after(e.headers)
            ) from e
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise LlmProtocolError(f"non-JSON response body: {raw[:240]}") from e
        if not isinstance(data, dict):
            raise LlmProtocolError(f"unexpected response shape: {raw[:240]}")
        err = data.get("error")
        if isinstance(err, dict):
            # gateways report upstream failures as HTTP 200 + {"error": ...};
            # an empty content here is a failure, not a legitimate answer
            message = str(err.get("message") or err)[:240]
            code = _error_code(err)
            if code is not None:
                raise LlmHttpError(code, message)
            raise LlmProtocolError(f"gateway error: {message}")
        choice = (data.get("choices") or [{}])[0]
        content = ((choice.get("message") or {}).get("content") or "").strip()
        finish = choice.get("finish_reason") or ""
        self._usage.add_from_response(data)
        return content, finish

    def chat(
        self,
        prompt: str,
        *,
        max_tokens: int = 4000,
        temperature: float = 0.2,
        timeout: int = 180,
        retries: int = 3,
        log: Callable[[str], None] | None = None,
        source: str = "",
    ) -> str:
        """One-shot chat completion with jittered exponential-backoff retries.

        Only transient failures (429, 5xx, timeouts, connection errors) are
        retried; other 4xx raise immediately. A 429's Retry-After header
        stretches the backoff when it asks for more than the schedule.

        If the reply was cut off (finish_reason == "length"), retry once with
        a doubled token budget and keep the longer answer. A reply that is
        still truncated after that is returned but never cached, so a later
        run can retry instead of hitting the truncation forever.

        `self.cache` is a paragraph-level store keyed by sha1(model + prompt +
        normalized source). Changing the model or prompt is a miss.
        """
        model = self.config.model
        if self.cache is not None:
            hit = self.cache.get(model, prompt, source)
            if hit is not None:
                return hit
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.config.reasoning_effort:
            body["reasoning_effort"] = self.config.reasoning_effort
        last: Exception | None = None
        for attempt in range(retries):
            try:
                self.throttle.wait()
                content, finish = self._chat_once(body, timeout)
                if finish == "length":
                    bumped = dict(body, max_tokens=min(max_tokens * 2, 16000))
                    try:
                        self.throttle.wait()
                        content2, finish2 = self._chat_once(bumped, timeout)
                        if len(content2) > len(content):
                            content, finish = content2, finish2
                    except Exception as e:  # noqa: BLE001 - keep the first answer
                        if log:
                            log(f"chat length-bump retry failed, keeping truncated answer: {e}")
                if self.cache is not None and content and finish != "length":
                    self.cache.put(model, prompt, content, source)
                return content
            except Exception as e:  # noqa: BLE001 - classified below
                if not _is_retryable(e):
                    raise
                last = e
                if log:
                    log(f"chat attempt {attempt + 1}/{retries} failed: {e}")
                if attempt + 1 < retries:
                    time.sleep(_backoff_seconds(attempt, getattr(e, "retry_after", None)))
        raise LlmError(f"chat failed after {retries} attempts: {last}") from last
