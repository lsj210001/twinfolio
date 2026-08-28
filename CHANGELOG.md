# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `LlmClient` / `LlmConfig`: one client per `pipeline.translate` run owns its
  cache, throttle, and token usage, replacing module-level LLM globals so
  concurrent runs in one process no longer interfere.
- Layered network error handling in `LlmClient`: typed `LlmHttpError` /
  `LlmProtocolError`, fail-fast on non-retryable 4xx, exponential backoff with
  jitter and `Retry-After` support for 429/5xx.
- CI workflow (ruff, mypy, pytest with coverage on Python 3.11–3.13),
  Dependabot config, ruff/mypy configuration, and contributor docs
  (`CONTRIBUTING.md`, `SECURITY.md`, this changelog).

### Fixed

- `translate_html` no longer duplicates `<title>` in `<head>` of translated
  documents.
- `LlmCache` is now thread-safe: resume workers sharing one instance can no
  longer corrupt the in-memory map during concurrent reads and writes.

### Changed

- Package version is now single-sourced from `book_translate.__version__`
  (`dynamic = ["version"]` in `pyproject.toml`).
- Runtime dependencies now declare lower bounds (`beautifulsoup4>=4.13`,
  `lxml>=5.0`); the `test` extra includes `pytest-cov`.

## [0.1.0] - 2026-08-27

### Added

- Initial release: translate an English EPUB into facing-page bilingual and
  Chinese-only editions via any OpenAI-compatible chat API.
- File-level resume (`resume-done.json` bound to source/base sha1), paragraph
  cache (`llm-cache.jsonl`), book-wide glossary, title/TOC post-processing,
  dedupe, and NCX/EPUB3 nav repair.
- `twinfolio` / `book-translate` CLI with `--test`, `--only`,
  `--retranslate`, and rolling-context options.
