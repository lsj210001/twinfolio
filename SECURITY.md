# Security Policy

## Supported versions

Only the latest release (and `main`) receives security fixes.

## Reporting a vulnerability

Please do not open a public issue for security problems. Instead, use
[GitHub private vulnerability reporting](https://github.com/lsj210001/twinfolio/security/advisories/new)
or email `dtbllsj@gmail.com`. You should receive a response within a few
days; please allow a reasonable disclosure window before publishing.

## Handling of secrets and data

- API keys are read from environment variables only (`OPENAI_API_KEY`, or
  the legacy `BBM_OPENAI_API_KEY`). The program never loads `.env` files
  automatically, never writes keys to disk, and never sends them anywhere
  except the configured `OPENAI_BASE_URL` endpoint.
- Book text is sent to the LLM endpoint you configure. Use a provider you
  trust with the content you are translating.
- The work directory (`.work-{book}/`) contains extracted book text and the
  translation cache (`llm-cache.jsonl`). Treat it as sensitive as the book
  itself; it is safe to delete at any time.
