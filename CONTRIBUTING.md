# Contributing to TwinFolio

Thanks for helping out. This page covers the local setup, the checks CI runs,
and a few conventions.

## Development setup

```bash
git clone https://github.com/lsj210001/twinfolio
cd twinfolio
python3 -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[test]" ruff mypy
```

Python 3.11+ is required; CI tests 3.11, 3.12, and 3.13.

## Running the checks

CI runs exactly these commands — run them locally before pushing:

```bash
ruff check src tests            # lint (blocking)
ruff format --check src tests   # formatting (informational for now, see below)
mypy src                        # types (blocking)
pytest -q --cov=src --cov-fail-under=78
```

Notes on the current baseline:

- `ruff format` is not yet enforced: most files predate the formatter config.
  Until the dedicated formatting-cleanup PR lands, do **not** reformat whole
  files in unrelated PRs — it drowns the diff. Formatting new code you add is
  fine.
- `[tool.ruff.lint].ignore` in `pyproject.toml` contains a block of
  temporarily ignored rules covering pre-existing violations, and
  `[tool.mypy]` exempts four modules with known type errors. Don't add new
  code that relies on those exemptions; shrinking the lists is welcome.

## Tests

- Tests live in `tests/` and run offline: LLM calls are stubbed, no API key
  or network is needed.
- New behavior needs a test. Bug fixes need a test that fails without the fix.
- Keep coverage at or above the CI gate (`--cov-fail-under=78`).

## Pull requests

- Base PRs on `main`, one logical change per PR.
- No logic changes mixed with mass reformatting.
- Update `CHANGELOG.md` (the `Unreleased` section) for user-visible changes.
- Never commit API keys or real EPUB content. `.env` is gitignored on
  purpose; secrets travel via environment variables only.
