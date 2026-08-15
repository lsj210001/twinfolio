"""Hand-written book glossary with prompt hints and placeholder locking.

Self-contained. Do not copy GPL/AGPL translation tooling.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from ._util import norm

DEFAULT_EXCLUDE = frozenset({"code", "pre", "sup"})
_PLACEHOLDER = "@@BT{i}@@"
_PLACEHOLDER_RE = re.compile(r"@@BT(\d+)@@")


@dataclass(frozen=True)
class Term:
    source: str
    target: str
    aliases: tuple[str, ...] = ()


def variants(term: Term) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in (term.source, *term.aliases):
        v = (v or "").strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    out.sort(key=len, reverse=True)
    return out


def _boundary_re(variant: str) -> re.Pattern[str]:
    esc = re.escape(variant)
    if re.match(r"^[A-Za-z0-9]", variant) and re.search(r"[A-Za-z0-9]$", variant):
        return re.compile(rf"(?<![A-Za-z0-9]){esc}(?![A-Za-z0-9])")
    return re.compile(esc)


def term_in(text: str, term: Term) -> bool:
    if not text:
        return False
    return any(_boundary_re(v).search(text) for v in variants(term))


def appearing_terms(texts: list[str], terms: list[Term]) -> list[Term]:
    return [t for t in terms if any(term_in(text, t) for text in texts)]


PROMPT_HINT_BUDGET = 3500


def format_hint(terms: list[Term]) -> str:
    if not terms:
        return ""
    lines = ["【术语表】下列术语必须按给定译法，不得改写："]
    for t in terms:
        extra = f"（{' / '.join(t.aliases)}）" if t.aliases else ""
        lines.append(f"- {t.source}{extra} → {t.target}")
    lines.append("原文中的 @@BTn@@ 占位符必须原样保留，不要翻译或改写。")
    return "\n".join(lines)


def terms_for_prompt(terms: list[Term], *, max_chars: int = PROMPT_HINT_BUDGET) -> list[Term]:
    """Keep a prefix of `terms` whose hint fits `max_chars`. Always keep the first term."""
    if not terms:
        return []
    chosen: list[Term] = []
    for t in terms:
        trial = chosen + [t]
        if chosen and len(format_hint(trial)) > max_chars:
            break
        chosen = trial
    return chosen


def inject_glossary_into_prompt(prompt: dict, terms: list[Term]) -> dict:
    """Copy a `{system,user}` prompt and insert a glossary block before `{text}`.

    Empty terms leave the user text unchanged. Does not mutate `prompt`.
    """
    out = {k: prompt[k] for k in prompt}
    if not terms:
        return out
    hint = format_hint(terms_for_prompt(terms))
    if not hint:
        return out
    user = out.get("user") or ""
    if not isinstance(user, str):
        return out
    needle = "{text}"
    if needle in user:
        out["user"] = user.replace(needle, hint + "\n\n" + needle, 1)
    else:
        out["user"] = user.rstrip() + "\n\n" + hint
    return out


def protect_text(text: str, terms: list[Term], *, start: int = 0) -> tuple[str, list[tuple[str, str]]]:
    """Replace appearing source/alias spans with placeholders. Longer first."""
    if not text or not terms:
        return text, []
    ordered = sorted(terms, key=lambda t: max((len(v) for v in variants(t)), default=0), reverse=True)
    mapping: list[tuple[str, str]] = []
    result = text
    i = start
    for term in ordered:
        matched = False
        for v in variants(term):
            pat = _boundary_re(v)
            if not pat.search(result):
                continue
            ph = _PLACEHOLDER.format(i=i)
            result = pat.sub(ph, result)
            mapping.append((ph, term.target))
            i += 1
            matched = True
            break
        if not matched:
            continue
    return result, mapping


def restore_text(text: str, mapping: list[tuple[str, str]], terms: list[Term] | None = None) -> str:
    """Put targets back; also rewrite leftover source/alias hits (hard lock)."""
    if not text:
        return text
    out = text
    for ph, target in mapping:
        out = out.replace(ph, target)
    if terms:
        leftover = sorted(terms, key=lambda t: max((len(v) for v in variants(t)), default=0), reverse=True)
        for term in leftover:
            for v in variants(term):
                if v == term.target:
                    continue
                out = _boundary_re(v).sub(term.target, out)
    return out


def _inside_exclude(node: NavigableString, exclude: frozenset[str] | set[str]) -> bool:
    parent = node.parent
    while parent is not None:
        name = getattr(parent, "name", None)
        if name in exclude:
            return True
        parent = getattr(parent, "parent", None)
    return False


def visible_text(el: Tag, exclude: frozenset[str] | set[str] = DEFAULT_EXCLUDE) -> str:
    """Element text skipping descendants inside EXCLUDE tags (code/pre/sup)."""
    parts: list[str] = []
    for node in el.descendants:
        if not isinstance(node, NavigableString):
            continue
        if _inside_exclude(node, exclude):
            continue
        parts.append(str(node))
    return norm(" ".join(parts))


def protect_html(
    html: str,
    terms: list[Term],
    *,
    exclude: frozenset[str] | set[str] = DEFAULT_EXCLUDE,
) -> tuple[str, list[tuple[str, str]]]:
    """Placeholder-lock terms in HTML text nodes, leaving EXCLUDE tags alone."""
    soup = BeautifulSoup(html, "html.parser")
    appearing = appearing_terms([visible_text(soup, exclude)], terms)
    mapping: list[tuple[str, str]] = []
    idx = 0
    for node in list(soup.descendants):
        if not isinstance(node, NavigableString):
            continue
        if _inside_exclude(node, exclude):
            continue
        new, local = protect_text(str(node), appearing, start=idx)
        if not local:
            continue
        node.replace_with(new)
        mapping.extend(local)
        idx += len(local)
    return str(soup), mapping


def apply_glossary(
    items: list[str],
    terms: list[Term],
    *,
    match_texts: list[str] | None = None,
) -> tuple[list[str], list[list[tuple[str, str]]], str, list[Term]]:
    """Protect a batch and build a hint from terms that actually appear.

    `match_texts` (e.g. visible text without code/pre/sup) decides which terms
    count as present; placeholders are applied to `items`.
    """
    if not items:
        return [], [], "", []
    if not terms:
        return list(items), [[] for _ in items], "", []
    haystacks = match_texts if match_texts is not None else items
    appearing = appearing_terms(list(haystacks), terms)
    hint = format_hint(appearing)
    protected: list[str] = []
    maps: list[list[tuple[str, str]]] = []
    for it in items:
        p, m = protect_text(it, appearing)
        protected.append(p)
        maps.append(m)
    return protected, maps, hint, appearing


def _parse_terms(data: object) -> list[Term]:
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("terms") or []
    else:
        return []
    if not isinstance(rows, list):
        return []
    out: list[Term] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        src = str(row.get("source") or "").strip()
        tgt = str(row.get("target") or "").strip()
        if not src or not tgt:
            continue
        raw_aliases = row.get("aliases") or []
        if isinstance(raw_aliases, str):
            raw_aliases = [raw_aliases]
        aliases = tuple(
            a.strip()
            for a in raw_aliases
            if isinstance(a, str) and a.strip() and a.strip() != src
        )
        out.append(Term(source=src, target=tgt, aliases=aliases))
    return out


def load_glossary(path: Path) -> list[Term]:
    """Load `.work-*/glossary.json`. Never overwrite a file the user already has.

    Missing file → write an empty template. Corrupt/unreadable existing file →
    empty terms, file left untouched.
    """
    path = Path(path)
    if path.is_file():
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []
        if not raw.strip():
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return _parse_terms(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"terms": []}, ensure_ascii=False, indent=2) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)
    return []
