import builtins
import io
import json
import random
import threading
import time
from dataclasses import replace

from book_translate.llm import LlmCache, LlmClient, LlmConfig, TokenUsage


CFG = LlmConfig(base_url="http://unit.test/v1", api_key="k", model="m", reasoning_effort="")


def test_cache_hit_skips_chat(tmp_path, monkeypatch):
    cache = LlmCache(tmp_path / "llm-cache.jsonl")
    calls = {"n": 0}

    def fake_once(self, body, timeout):
        calls["n"] += 1
        return "译文甲", "stop"

    monkeypatch.setattr(LlmClient, "_chat_once", fake_once)
    client = LlmClient(CFG, cache=cache)
    first = client.chat("把这段译成中文：hello world", source="hello world")
    second = client.chat("把这段译成中文：hello world", source="hello world")
    assert first == second == "译文甲"
    assert calls["n"] == 1


def test_prompt_or_model_change_is_a_miss(tmp_path, monkeypatch):
    cache = LlmCache(tmp_path / "llm-cache.jsonl")
    replies = iter(["一", "二", "三"])
    calls = {"n": 0}

    def fake_once(self, body, timeout):
        calls["n"] += 1
        return next(replies), "stop"

    monkeypatch.setattr(LlmClient, "_chat_once", fake_once)
    client = LlmClient(CFG, cache=cache)
    other_model = LlmClient(replace(CFG, model="other-model"), cache=cache)
    client.chat("prompt-A\nhello", source="hello")
    other_model.chat("prompt-A\nhello", source="hello")
    client.chat("prompt-B\nhello", source="hello")
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

    def fake_once(self, body, timeout):
        calls["n"] += 1
        return "新译文", "stop"

    monkeypatch.setattr(LlmClient, "_chat_once", fake_once)
    client = LlmClient(CFG, cache=cache)
    assert client.chat("missing-prompt", source="src") == "新译文"
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


def _prefill_jsonl(path, n: int, response: str = "旧值") -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(
                json.dumps({"key": f"prefill-{i:05d}", "response": response}, ensure_ascii=False)
                + "\n"
            )


def test_cache_put_shared_instance_is_thread_safe(tmp_path):
    """Resume workers share one LlmCache; concurrent put() on that instance
    (plus a compact() racing them) must not crash and must not lose entries.
    The random sleeps scatter each thread's put across the other threads'
    append/compact windows to make interleavings reproducible."""
    path = tmp_path / "llm-cache.jsonl"
    prefill = 3000
    _prefill_jsonl(path, prefill)
    cache = LlmCache(path)
    puts_per_thread = 25
    errors: list[BaseException] = []
    start = threading.Barrier(5)

    def worker(tid: int) -> None:
        rnd = random.Random(tid)
        try:
            start.wait()
            for j in range(puts_per_thread):
                time.sleep(rnd.random() * 0.004)
                cache.put("m", f"p{tid}-{j}", f"r{tid}-{j}", f"s{tid}-{j}")
        except BaseException as e:  # noqa: BLE001 - collect for the parent thread
            errors.append(e)

    def compactor() -> None:
        try:
            start.wait()
            for _ in range(5):
                time.sleep(0.01)
                cache.compact()
        except BaseException as e:  # noqa: BLE001 - collect for the parent thread
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    threads.append(threading.Thread(target=compactor))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"concurrent put/compact raised: {errors[:3]}"
    loaded = LlmCache(path)
    assert len(loaded._mem) == prefill + 4 * puts_per_thread
    assert loaded._mem["prefill-00000"] == "旧值"
    for tid in range(4):
        for j in range(puts_per_thread):
            assert loaded.get("m", f"p{tid}-{j}", f"s{tid}-{j}") == f"r{tid}-{j}"


def test_put_io_is_constant_per_entry(tmp_path, monkeypatch):
    """I/O-conservation regression guard: with n entries already on disk,
    k puts must re-read nothing (zero _load calls) and write only the k new
    lines -- a byte volume independent of n. The old read-everything /
    rewrite-everything put() trips both assertions (it called _load once per
    put and wrote ~n lines per put), regardless of machine speed."""
    path = tmp_path / "llm-cache.jsonl"
    prefill = 2000
    _prefill_jsonl(path, prefill)
    cache = LlmCache(path)
    assert len(cache._mem) == prefill

    load_calls = {"n": 0}
    real_load = LlmCache._load

    def counting_load(self):
        load_calls["n"] += 1
        return real_load(self)

    monkeypatch.setattr(LlmCache, "_load", counting_load)

    # Count every unit written to the cache file or its .tmp sibling, no
    # matter which code path writes it (direct open() or Path.write_text,
    # which goes through io.open in pathlib).
    written = {"units": 0}
    tracked = {str(path), str(path) + ".tmp"}
    real_open = builtins.open

    class CountingFile:
        def __init__(self, fh):
            self._fh = fh

        def write(self, data):
            written["units"] += len(data)
            return self._fh.write(data)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self._fh.__exit__(*exc)

        def __iter__(self):
            return iter(self._fh)

        def __getattr__(self, name):
            return getattr(self._fh, name)

    def counting_open(file, mode="r", *args, **kwargs):
        fh = real_open(file, mode, *args, **kwargs)
        if str(file) in tracked and any(c in mode for c in "wax+"):
            return CountingFile(fh)
        return fh

    monkeypatch.setattr(builtins, "open", counting_open)
    monkeypatch.setattr(io, "open", counting_open)

    size_before = path.stat().st_size
    puts = 20
    expected = 0  # note: independent of `prefill`
    for i in range(puts):
        line = (
            json.dumps(
                {"key": LlmCache.make_key("m", f"p{i}", "s"), "response": f"resp-{i}"},
                ensure_ascii=False,
            )
            + "\n"
        )
        expected += len(line.encode("utf-8"))
        cache.put("m", f"p{i}", f"resp-{i}", "s")

    assert load_calls["n"] == 0, "put must never re-read the file"
    assert written["units"] == expected, "put must write exactly the appended lines"
    assert path.stat().st_size - size_before == expected
    loaded = LlmCache(path)
    assert len(loaded._mem) == prefill + puts


def test_put_repairs_torn_tail_so_no_entry_is_lost(tmp_path):
    """Crash semantics: a put after a crash left a torn (newline-less) last
    line must terminate that line before appending, or the new record would
    be glued onto the fragment and both lines lost on reload."""
    path = tmp_path / "llm-cache.jsonl"
    cache = LlmCache(path)
    for i in range(5):
        cache.put("m", f"p{i}", f"r{i}", "s")
    with open(path, "ab") as fh:
        fh.write(b'{"key": "torn')  # crash mid-append: no trailing newline

    survivor = LlmCache(path)
    for i in range(5):
        assert survivor.get("m", f"p{i}", "s") == f"r{i}"
    survivor.put("m", "p-new", "r-new", "s")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[5] == '{"key": "torn', "the torn fragment must stay on its own line"
    json.loads(lines[6])  # the new record is a complete line of valid JSON
    loaded = LlmCache(path)
    for i in range(5):
        assert loaded.get("m", f"p{i}", "s") == f"r{i}"
    assert loaded.get("m", "p-new", "s") == "r-new"


def test_compact_collapses_duplicates_last_wins(tmp_path):
    path = tmp_path / "llm-cache.jsonl"
    cache = LlmCache(path)
    for rnd in range(4):
        for i in range(6):
            cache.put("m", f"p{i}", f"r{i}-round{rnd}", "s")
    raw_lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(raw_lines) == 24, "puts only append; superseded lines stay until compact"

    cache.compact()
    assert not path.with_name(path.name + ".tmp").exists()
    compacted = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(compacted) == 6
    loaded = LlmCache(path)
    for i in range(6):
        assert loaded.get("m", f"p{i}", "s") == f"r{i}-round3"


def test_load_autocompacts_only_past_slack(tmp_path, monkeypatch):
    monkeypatch.setattr(LlmCache, "_compact_slack", 8)
    path = tmp_path / "llm-cache.jsonl"
    cache = LlmCache(path)
    for rnd in range(5):
        for i in range(3):
            cache.put("m", f"p{i}", f"r{i}-round{rnd}", "s")
    # 15 lines, 3 unique keys -> 12 dead lines > 8: reload compacts
    reloaded = LlmCache(path)
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 3
    for i in range(3):
        assert reloaded.get("m", f"p{i}", "s") == f"r{i}-round4"

    # 3 unique lines, 0 dead: reload leaves the file byte-for-byte untouched
    before = path.read_bytes()
    LlmCache(path)
    assert path.read_bytes() == before


def test_chat_once_records_usage_on_client(monkeypatch):
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

    monkeypatch.setattr("book_translate.llm.urllib.request.urlopen", lambda *a, **k: Resp())
    client = LlmClient(CFG)
    content, finish = client._chat_once({"model": "m"}, 10)
    assert content == "你好"
    assert finish == "stop"
    assert client.usage.input_tokens == 11
    assert client.usage.output_tokens == 7
    assert client.usage.total_tokens == 18
    assert client.usage.calls == 1
    assert "input=11" in client.usage.format_line()


def test_token_usage_accepts_input_output_aliases():
    usage = TokenUsage()
    usage.add_from_response({"usage": {"input_tokens": 2, "output_tokens": 3}})
    assert usage.input_tokens == 2
    assert usage.output_tokens == 3
    assert usage.total_tokens == 5
    assert usage.calls == 1
    usage.add_from_response({"choices": []})
    assert usage.calls == 1
