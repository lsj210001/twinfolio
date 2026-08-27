"""LLM chat client: immutable config, per-instance throttle/usage/cache.

One `LlmClient` is built per `pipeline.translate` run, so two concurrent
runs in the same process cannot reset each other's token totals or rewrite
each other's throttle interval (which module-level singletons used to allow).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
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
                data = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code}: {raw[:240]}") from e
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
        """One-shot chat completion with exponential-backoff retries.

        If the reply was cut off (finish_reason == "length"), retry once with
        a doubled token budget and keep the longer answer.

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
                        content2, _ = self._chat_once(bumped, timeout)
                        if len(content2) > len(content):
                            content = content2
                    except Exception:
                        pass
                if self.cache is not None and content:
                    self.cache.put(model, prompt, content, source)
                return content
            except Exception as e:  # noqa: BLE001 - network layer, retry everything
                last = e
                if log:
                    log(f"chat attempt {attempt + 1}/{retries} failed: {e}")
                if attempt + 1 < retries:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"chat failed after {retries} attempts: {last}") from last
