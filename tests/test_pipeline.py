import shutil
import zipfile
from pathlib import Path

from book_translate import file_resume, pipeline
from book_translate.llm import LlmClient


def _tiny_epub(path: Path, payload: bytes = b"x" * 1200) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        z.writestr("payload.bin", payload)
    return path


def test_reuse_checkpoint_requires_src_sha1(tmp_path):
    src = _tiny_epub(tmp_path / "book.epub")
    resumed = _tiny_epub(tmp_path / "book_bilingual-resumed.epub")
    state = tmp_path / "resume-done.json"
    assert not pipeline._reuse_resume_checkpoint(resumed, state, src)

    file_resume._write_state(state, resumed, [], src)
    assert pipeline._reuse_resume_checkpoint(resumed, state, src)

    other = _tiny_epub(tmp_path / "other.epub", b"y" * 1200)
    assert not pipeline._reuse_resume_checkpoint(resumed, state, other)


def test_seed_bilingual_copies_source(tmp_path):
    src = _tiny_epub(tmp_path / "book.epub")
    work = tmp_path / "work"
    work.mkdir()
    dest = pipeline._seed_bilingual(src, work)
    assert dest == work / "book_bilingual.epub"
    assert dest.is_file()
    assert dest.stat().st_size == src.stat().st_size


def test_translate_uses_file_resume_not_bbook(tmp_path, monkeypatch):
    src = _tiny_epub(tmp_path / "book.epub")
    seen: dict[str, object] = {}

    def fake_resume(src_epub, bilingual, resumed, **kwargs):
        seen["bilingual"] = Path(bilingual)
        seen["client"] = kwargs.get("client")
        shutil.copy2(bilingual, resumed)
        return resumed, []

    monkeypatch.setattr(file_resume, "resume_missing", fake_resume)
    monkeypatch.setattr(
        pipeline, "fix_english_titles", lambda epub, out_path, **k: shutil.copy2(epub, out_path)
    )
    monkeypatch.setattr(
        pipeline, "dedupe_epub", lambda src_epub, dest: shutil.copy2(src_epub, dest) or 0
    )
    monkeypatch.setattr(pipeline, "repair_epub_toc", lambda p: None)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("BBM_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BOOK_TRANSLATE_MODEL", raising=False)
    monkeypatch.delenv("MODEL", raising=False)

    out = tmp_path / "out"
    result = pipeline.translate(src, out, bilingual_only=True)
    assert result["bilingual"].is_file()
    assert seen["bilingual"].name == "book_bilingual.epub"
    assert zipfile.is_zipfile(seen["bilingual"])
    client = seen["client"]
    assert isinstance(client, LlmClient)
    assert client.config.model == "gpt-4o-mini"
    assert client.cache is not None
    assert client.usage.calls == 0, "a fresh client must start with zero usage"
    assert not hasattr(pipeline, "_run_bbook")
    assert not hasattr(pipeline, "_bbm_bin")


def test_translate_test_mode_skips_titles_and_writes_sample(tmp_path, monkeypatch):
    src = _tiny_epub(tmp_path / "book.epub")
    called = {"titles": 0}

    def fake_resume(src_epub, bilingual, resumed, **kwargs):
        assert kwargs.get("test_num") == 10
        shutil.copy2(bilingual, resumed)
        return resumed, []

    def fake_titles(*_a, **_k):
        called["titles"] += 1

    monkeypatch.setattr(file_resume, "resume_missing", fake_resume)
    monkeypatch.setattr(pipeline, "fix_english_titles", fake_titles)
    monkeypatch.setattr(
        pipeline, "dedupe_epub", lambda src_epub, dest: shutil.copy2(src_epub, dest) or 0
    )
    monkeypatch.setattr(pipeline, "repair_epub_toc", lambda p: None)
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    result = pipeline.translate(src, tmp_path / "out", test_num=10)
    assert "test" in result
    assert result["test"].name.endswith("试译.epub")
    assert result["test"].is_file()
    assert called["titles"] == 0
    assert "bilingual" not in result
    assert "zh" not in result
