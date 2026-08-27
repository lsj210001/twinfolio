"""Network-layer behavior of LlmClient: typed errors, selective retry,
Retry-After-aware backoff, truncation handling, and per-instance state."""
import email.message
import io
import json
import urllib.error

import pytest

from book_translate.llm import (
    LlmCache,
    LlmClient,
    LlmConfig,
    LlmError,
    LlmHttpError,
    LlmProtocolError,
    _parse_retry_after,
)

CFG = LlmConfig(base_url="http://unit.test/v1", api_key="k", model="m", reasoning_effort="")


def _no_jitter_sleeps(monkeypatch):
    """Record backoff sleeps without waiting for them."""
    sleeps: list[float] = []
    monkeypatch.setattr("book_translate.llm.time.sleep", lambda s: sleeps.append(s))
    return sleeps


def test_5xx_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_once(self, body, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise LlmHttpError(500, "upstream exploded")
        return "译文", "stop"

    monkeypatch.setattr(LlmClient, "_chat_once", fake_once)
    sleeps = _no_jitter_sleeps(monkeypatch)
    client = LlmClient(CFG)
    assert client.chat("p") == "译文"
    assert calls["n"] == 3
    # exponential backoff with jitter: 2*(attempt+1) plus up to 25%
    assert len(sleeps) == 2
    assert 2.0 <= sleeps[0] <= 2.5
    assert 4.0 <= sleeps[1] <= 5.0


def test_timeout_and_connection_errors_are_retried(monkeypatch):
    errors = iter([TimeoutError("read timed out"), urllib.error.URLError("refused")])
    calls = {"n": 0}

    def fake_once(self, body, timeout):
        calls["n"] += 1
        try:
            raise next(errors)
        except StopIteration:
            return "通了", "stop"

    monkeypatch.setattr(LlmClient, "_chat_once", fake_once)
    _no_jitter_sleeps(monkeypatch)
    assert LlmClient(CFG).chat("p") == "通了"
    assert calls["n"] == 3


def test_all_retries_fail_raises_with_cause(monkeypatch):
    def fake_once(self, body, timeout):
        raise LlmHttpError(503, "still down")

    monkeypatch.setattr(LlmClient, "_chat_once", fake_once)
    _no_jitter_sleeps(monkeypatch)
    with pytest.raises(LlmError) as exc:
        LlmClient(CFG).chat("p", retries=3)
    assert "after 3 attempts" in str(exc.value)
    assert isinstance(exc.value.__cause__, LlmHttpError)
    assert exc.value.__cause__.code == 503
    # callers that caught the old opaque RuntimeError keep working
    assert isinstance(exc.value, RuntimeError)


def test_4xx_fails_fast_without_retry(monkeypatch):
    calls = {"n": 0}

    def fake_once(self, body, timeout):
        calls["n"] += 1
        raise LlmHttpError(401, "bad api key")

    monkeypatch.setattr(LlmClient, "_chat_once", fake_once)
    sleeps = _no_jitter_sleeps(monkeypatch)
    with pytest.raises(LlmHttpError) as exc:
        LlmClient(CFG).chat("p")
    assert exc.value.code == 401
    assert calls["n"] == 1, "a permanent 4xx must not be retried"
    assert sleeps == [], "no backoff for a permanent failure"


def test_http_error_keeps_status_and_retry_after(monkeypatch):
    headers = email.message.Message()
    headers["Retry-After"] = "7"

    def raise_429(*a, **k):
        raise urllib.error.HTTPError(
            "http://unit.test/v1/chat/completions",
            429,
            "Too Many Requests",
            headers,
            io.BytesIO(b'{"error": {"message": "slow down"}}'),
        )

    monkeypatch.setattr("book_translate.llm.urllib.request.urlopen", raise_429)
    with pytest.raises(LlmHttpError) as exc:
        LlmClient(CFG)._chat_once({"model": "m"}, 10)
    assert exc.value.code == 429
    assert exc.value.retry_after == 7.0
    assert "slow down" in exc.value.body
    assert isinstance(exc.value.__cause__, urllib.error.HTTPError)


def test_429_backoff_honors_retry_after(monkeypatch):
    calls = {"n": 0}

    def fake_once(self, body, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LlmHttpError(429, "rate limited", retry_after=9.0)
        return "过了", "stop"

    monkeypatch.setattr(LlmClient, "_chat_once", fake_once)
    sleeps = _no_jitter_sleeps(monkeypatch)
    assert LlmClient(CFG).chat("p") == "过了"
    assert len(sleeps) == 1
    assert 9.0 <= sleeps[0] <= 9.0 * 1.25, "Retry-After must stretch the backoff"


def test_parse_retry_after_variants():
    numeric = email.message.Message()
    numeric["Retry-After"] = "12"
    assert _parse_retry_after(numeric) == 12.0

    dated = email.message.Message()
    dated["Retry-After"] = "Wed, 21 Oct 2015 07:28:00 GMT"  # long past -> clamp to 0
    assert _parse_retry_after(dated) == 0.0

    junk = email.message.Message()
    junk["Retry-After"] = "soonish"
    assert _parse_retry_after(junk) is None
    assert _parse_retry_after(email.message.Message()) is None
    assert _parse_retry_after(None) is None


def test_truncated_reply_is_returned_but_never_cached(tmp_path, monkeypatch):
    cache = LlmCache(tmp_path / "llm-cache.jsonl")
    bodies: list[dict] = []

    def fake_once(self, body, timeout):
        bodies.append(dict(body))
        return "截断的译文", "length"

    monkeypatch.setattr(LlmClient, "_chat_once", fake_once)
    client = LlmClient(CFG, cache=cache)
    assert client.chat("p", max_tokens=4000, source="s") == "截断的译文"
    # first attempt plus the doubled-budget attempt, both truncated
    assert [b["max_tokens"] for b in bodies] == [4000, 8000]
    assert cache.get("m", "p", "s") is None, "a truncated reply must not poison the cache"
    # next call misses the cache and hits the network again (self-healing)
    client.chat("p", max_tokens=4000, source="s")
    assert len(bodies) == 4


def test_length_bump_success_is_cached(tmp_path, monkeypatch):
    cache = LlmCache(tmp_path / "llm-cache.jsonl")
    replies = iter([("短", "length"), ("加倍之后拿到的完整译文", "stop")])

    def fake_once(self, body, timeout):
        return next(replies)

    monkeypatch.setattr(LlmClient, "_chat_once", fake_once)
    client = LlmClient(CFG, cache=cache)
    assert client.chat("p", source="s") == "加倍之后拿到的完整译文"
    assert cache.get("m", "p", "s") == "加倍之后拿到的完整译文"


def test_length_bump_failure_is_logged_and_keeps_first_answer(tmp_path, monkeypatch):
    cache = LlmCache(tmp_path / "llm-cache.jsonl")
    calls = {"n": 0}

    def fake_once(self, body, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return "第一次的截断答案", "length"
        raise LlmHttpError(500, "bump blew up")

    monkeypatch.setattr(LlmClient, "_chat_once", fake_once)
    logs: list[str] = []
    client = LlmClient(CFG, cache=cache)
    assert client.chat("p", log=logs.append, source="s") == "第一次的截断答案"
    assert any("length-bump" in line and "bump blew up" in line for line in logs), logs
    assert calls["n"] == 2, "the failed bump must not restart the outer retry loop"
    assert cache.get("m", "p", "s") is None, "still truncated, so still uncached"


def test_gateway_error_in_200_body_raises_http_error(monkeypatch):
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                {"error": {"message": "upstream 502 Bad Gateway", "code": 502}}
            ).encode()

    monkeypatch.setattr("book_translate.llm.urllib.request.urlopen", lambda *a, **k: Resp())
    client = LlmClient(CFG)
    with pytest.raises(LlmHttpError) as exc:
        client._chat_once({"model": "m"}, 10)
    assert exc.value.code == 502
    assert "upstream 502" in str(exc.value)
    assert client.usage.calls == 0, "an error body must not count as usage"


def test_gateway_error_without_code_is_protocol_error(monkeypatch):
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"error": {"message": "something odd", "code": "oops"}}).encode()

    monkeypatch.setattr("book_translate.llm.urllib.request.urlopen", lambda *a, **k: Resp())
    with pytest.raises(LlmProtocolError):
        LlmClient(CFG)._chat_once({"model": "m"}, 10)


def test_non_json_body_is_protocol_error(monkeypatch):
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"<html>502 Bad Gateway</html>"

    monkeypatch.setattr("book_translate.llm.urllib.request.urlopen", lambda *a, **k: Resp())
    with pytest.raises(LlmProtocolError):
        LlmClient(CFG)._chat_once({"model": "m"}, 10)


def test_clients_do_not_share_usage_or_throttle():
    a, b = LlmClient(CFG), LlmClient(CFG)
    a.usage.add_from_response({"usage": {"prompt_tokens": 5, "completion_tokens": 5}})
    a.throttle.min_interval = 0.5
    assert b.usage.calls == 0
    assert b.usage.total_tokens == 0
    assert b.throttle.min_interval == 0.0
