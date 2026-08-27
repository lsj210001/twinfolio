"""Orchestrate bilingual translation + Chinese-only derive. CLI only, no HTTP sidecar."""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

from . import file_resume
from ._util import warn_zip_member_names
from .dedupe_epub import rewrite as dedupe_epub
from .derive import derive_zh
from .glossary import load_glossary
from .llm import LlmCache, LlmClient, LlmConfig
from .title_postprocess import fix_english_titles
from .toc_repair import repair_epub_toc


def _log(msg: str) -> None:
    print(msg, flush=True)


def _api() -> LlmConfig:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("BBM_OPENAI_API_KEY") or ""
    if not key:
        raise RuntimeError("set OPENAI_API_KEY (OpenAI-compatible)")
    model = os.environ.get("BOOK_TRANSLATE_MODEL") or os.environ.get("MODEL") or "gpt-4o-mini"
    effort = (
        os.environ.get("BOOK_TRANSLATE_REASONING_EFFORT")
        or os.environ.get("BBM_REASONING_EFFORT")
        or "low"
    ).strip()
    if "3.7" in model and effort.lower() in {"minimal", "none", "min"}:
        effort = "low"
    return LlmConfig(
        base_url=(os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/"),
        api_key=key,
        model=model,
        reasoning_effort=effort,
    )


def _reuse_resume_checkpoint(resumed: Path, state_path: Path, src: Path) -> bool:
    try:
        if not resumed.is_file() or not zipfile.is_zipfile(resumed):
            return False
    except OSError:
        return False
    return file_resume.checkpoint_matches_src(state_path, src)


def _seed_bilingual(src: Path, work: Path) -> Path:
    dest = work / f"{src.stem}_bilingual.epub"
    shutil.copy2(src, dest)
    return dest


def translate(
    src: Path,
    out_dir: Path,
    *,
    bilingual_only: bool = False,
    test_num: int | None = None,
    only_files: str | list[str] | None = None,
    retranslate: str | list[str] | None = None,
    use_context: bool = False,
    context_paragraphs: int = 0,
    batch_tokens: int | None = None,
) -> dict[str, Path]:
    src = Path(src).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / f".work-{src.stem}"
    work.mkdir(parents=True, exist_ok=True)
    glossary = load_glossary(work / "glossary.json")
    cache = LlmCache(work / "llm-cache.jsonl")
    client = LlmClient(_api(), cache=cache)
    warn_zip_member_names(src, _log)
    short_names = file_resume.short_batch_basenames(src)
    if context_paragraphs < 1:
        context_paragraphs = file_resume.DEFAULT_CONTEXT_PARAGRAPHS
    if not use_context:
        use_context = os.environ.get("BOOK_TRANSLATE_USE_CONTEXT", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    resumed = work / f"{src.stem}_bilingual-resumed.epub"
    state_path = work / "resume-done.json"
    testing = test_num is not None
    if testing:
        bilingual = _seed_bilingual(src, work)
        _log(f"test mode: first {test_num} paragraphs, skip titles/zh")
    elif _reuse_resume_checkpoint(resumed, state_path, src):
        _log(f"reusing resume checkpoint as base: {resumed.name}")
        bilingual = resumed
    else:
        bilingual = _seed_bilingual(src, work)
        _log("translating all HTML via file-resume")

    short_n = int(os.environ.get("SHORT_BATCH_SIZE", "8"))
    bilingual, _done = file_resume.resume_missing(
        src,
        bilingual,
        resumed if not testing else work / f"{src.stem}_bilingual-test.epub",
        client=client,
        log=_log,
        state_path=None if testing else state_path,
        short_names=short_names,
        short_n=short_n,
        glossary=glossary,
        only_files=only_files,
        retranslate=retranslate,
        test_num=test_num,
        use_context=use_context,
        context_paragraphs=context_paragraphs,
        batch_tokens=batch_tokens,
    )

    if testing:
        sample = out_dir / f"{src.stem}-试译.epub"
        removed = dedupe_epub(bilingual, sample)
        repair_epub_toc(sample)
        _log(f"dedupe test removed={removed}")
        result = {"test": sample}
        _log(client.usage.format_line())
        _log("done " + " ".join(f"{k}={v}" for k, v in result.items()))
        return result

    titled = work / f"{bilingual.stem}-titles.epub"
    fix_english_titles(
        bilingual,
        out_path=titled,
        client=client,
        log_path=work / "title.log",
        glossary=glossary,
    )
    deduped = out_dir / f"{src.stem}-中英双语.epub"
    removed = dedupe_epub(titled, deduped)
    repair_epub_toc(deduped)
    _log(f"dedupe bilingual removed={removed}")

    result = {"bilingual": deduped}
    if not bilingual_only:
        zh_raw = derive_zh(deduped, work)
        zh_titled = work / f"{zh_raw.stem}-titles.epub"
        fix_english_titles(
            zh_raw,
            out_path=zh_titled,
            client=client,
            log_path=work / "title-zh.log",
            glossary=glossary,
        )
        zh_out = out_dir / f"{src.stem}-中文.epub"
        removed_zh = dedupe_epub(zh_titled, zh_out)
        repair_epub_toc(zh_out)
        _log(f"dedupe zh removed={removed_zh}")
        result["zh"] = zh_out
    _log(client.usage.format_line())
    _log("done " + " ".join(f"{k}={v}" for k, v in result.items()))
    return result
