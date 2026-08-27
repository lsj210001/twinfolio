import json
import random
import threading
import time

from book_translate._util import LlmCache, TOKEN_USAGE, _chat_once, chat


API = {"base_url": "http://unit.test/v1", "api_key": "k", "model": "m", "reasoning_effort": ""}


def test_cache_hit_skips_chat(tmp_path, monkeypatch):
    cache = LlmCache(tmp_path / "llm-cache.jsonl")
    calls = {"n": 0}

    def fake_once(api, body, timeout):
        calls["n"] += 1
        return "译文甲", "stop"

    monkeypatch.setattr("book_translate._util._chat_once", fake_once)
    first = chat(API, "把这段译成中文：hello world", cache=cache, source="hello world")
    second = chat(API, "把这段译成中文：hello world", cache=cache, source="hello world")
    assert first == second == "译文甲"
    assert calls["n"] == 1


def test_prompt_or_model_change_is_a_miss(tmp_path, monkeypatch):
    cache = LlmCache(tmp_path / "llm-cache.jsonl")
    replies = iter(["一", "二", "三"])
    calls = {"n": 0}

    def fake_once(api, body, timeout):
        calls["n"] += 1
        return next(replies), "stop"

    monkeypatch.setattr("book_translate._util._chat_once", fake_once)
    chat(API, "prompt-A\nhello", cache=cache, source="hello")
    other_model = dict(API, model="other-model")
    chat(other_model, "prompt-A\nhello", cache=cache, source="hello")
    chat(API, "prompt-B\nhello", cache=cache, source="hello")
    assert calls["n"] == 3


def test_empty_and_half_written_cache_are_ignored(tmp_path, monkeypatch):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert LlmCache(empty).get("m", "p", "s") is None

    half = tmp_path / "half.jsonl"
    good_key = LlmCache.make_key("m", "ok-prompt", "src")
    half.write_text(
        json.dumps({"key": good_key, "response": "命中"}, ensure_ascii=False)
        + "\n"
        + '{"key": "truncated"',
        encoding="utf-8",
    )
    cache = LlmCache(half)
    assert cache.get("m", "ok-prompt", "src") == "命中"

    calls = {"n": 0}

    def fake_once(api, body, timeout):
        calls["n"] += 1
        return "新译文", "stop"

    monkeypatch.setattr("book_translate._util._chat_once", fake_once)
    assert chat(API, "missing-prompt", cache=cache, source="src") == "新译文"
    assert calls["n"] == 1


def test_cache_put_leaves_valid_jsonl(tmp_path):
    path = tmp_path / "llm-cache.jsonl"
    cache = LlmCache(path)
    cache.put("m", "p1", "r1", "s1")
    cache.put("m", "p2", "r2", "s2")
    assert not path.with_name(path.name + ".tmp").exists()
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    for line in lines:
        json.loads(line)
    loaded = LlmCache(path)
    assert loaded.get("m", "p1", "s1") == "r1"
    assert loaded.get("m", "p2", "s2") == "r2"


def test_cache_put_serializes_writers(tmp_path):
    path = tmp_path / "llm-cache.jsonl"
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            LlmCache(path).put("m", f"p{i}", f"r{i}", f"s{i}")
        except BaseException as e:  # noqa: BLE001 - collect for the parent thread
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 8
    for line in lines:
        json.loads(line)
    assert len(LlmCache(path)._mem) == 8


def test_cache_put_shared_instance_is_thread_safe(tmp_path):
    """Resume workers share one LlmCache; concurrent put() on that instance
    must not crash (dict mutated while another thread iterates it) and must
    not lose entries. The random sleeps scatter each thread's put across the
    other threads' payload-building windows to make the race reproducible."""
    path = tmp_path / "llm-cache.jsonl"
    cache = LlmCache(path)
    prefill = 3000
    for i in range(prefill):
        cache._mem[f"prefill-{i:05d}"] = "值" * 20
    puts_per_thread = 25
    errors: list[BaseException] = []
    start = threading.Barrier(4)

    def worker(tid: int) -> None:
        rnd = random.Random(tid)
        try:
            start.wait()
            for j in range(puts_per_thread):
                time.sleep(rnd.random() * 0.004)
                cache.put("m", f"p{tid}-{j}", f"r{tid}-{j}", f"s{tid}-{j}")
        except BaseException as e:  # noqa: BLE001 - collect for the parent thread
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"concurrent put raised: {errors[:3]}"
    loaded = LlmCache(path)
    assert len(loaded._mem) == prefill + 4 * puts_per_thread
    for tid in range(4):
        for j in range(puts_per_thread):
            assert loaded.get("m", f"p{tid}-{j}", f"s{tid}-{j}") == f"r{tid}-{j}"


def test_chat_once_records_usage(monkeypatch):
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": {"content": "  你好  "}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                }
            ).encode()

    monkeypatch.setattr("book_translate._util.urllib.request.urlopen", lambda *a, **k: Resp())
    TOKEN_USAGE.reset()
    content, finish = _chat_once(API, {"model": "m"}, 10)
    assert content == "你好"
    assert finish == "stop"
    assert TOKEN_USAGE.input_tokens == 11
    assert TOKEN_USAGE.output_tokens == 7
    assert TOKEN_USAGE.total_tokens == 18
    assert TOKEN_USAGE.calls == 1
    assert "input=11" in TOKEN_USAGE.format_line()


def test_token_usage_accepts_input_output_aliases():
    TOKEN_USAGE.reset()
    TOKEN_USAGE.add_from_response({"usage": {"input_tokens": 2, "output_tokens": 3}})
    assert TOKEN_USAGE.input_tokens == 2
    assert TOKEN_USAGE.output_tokens == 3
    assert TOKEN_USAGE.total_tokens == 5
    assert TOKEN_USAGE.calls == 1
    TOKEN_USAGE.add_from_response({"choices": []})
    assert TOKEN_USAGE.calls == 1
