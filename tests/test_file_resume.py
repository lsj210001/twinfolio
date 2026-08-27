import json
import re
import shutil
import zipfile
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from book_translate import file_resume
from book_translate.glossary import Term

API = {"base_url": "http://unit.test/v1", "api_key": "k", "model": "m", "reasoning_effort": ""}

OPF = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Demo</dc:title>
    <dc:identifier id="id">urn:demo</dc:identifier>
  </metadata>
  <manifest>
    <item href="Text/alpha.xhtml" id="alpha" media-type="application/xhtml+xml"/>
    <item href="Text/beta.xhtml" id="beta" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="alpha"/>
    <itemref idref="beta"/>
  </spine>
</package>
"""

ALPHA_TEXT = "The quick brown fox jumps over the lazy dog today."
BETA_TEXT = "Neural networks consist of many layers of neurons."

_ITEM_RE = re.compile(r"^(\d+)\. (.+)$", re.M)


def _echo_chat(api, prompt, **kwargs):
    """Deterministic fake LLM: answers numbered lists by echoing the source."""
    lines = [f"{m.group(1)}. 中文（{m.group(2)[:24]}）" for m in _ITEM_RE.finditer(prompt)]
    return "\n".join(lines) if lines else "中文译文内容"


def _make_src(tmp_path: Path) -> Path:
    src = tmp_path / "book.epub"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        z.writestr("EPUB/content.opf", OPF)
        z.writestr("EPUB/Text/alpha.xhtml", f"<html><body><p>{ALPHA_TEXT}</p></body></html>")
        z.writestr("EPUB/Text/beta.xhtml", f"<html><body><p>{BETA_TEXT}</p></body></html>")
    return src


def _read(epub: Path, name: str) -> str:
    with zipfile.ZipFile(epub) as z:
        return z.read(name).decode("utf-8")


def test_resume_survives_interrupted_rerun(tmp_path, monkeypatch):
    """H1/M1: translate half, crash, rerun from the checkpoint - nothing lost."""
    src = _make_src(tmp_path)
    base = tmp_path / "base.epub"
    shutil.copy2(src, base)
    resumed = tmp_path / "resumed.epub"
    state = tmp_path / "resume-done.json"

    def chat_run1(api, prompt, **kwargs):
        if "Neural networks" in prompt:
            raise RuntimeError("network down")
        return _echo_chat(api, prompt)

    monkeypatch.setattr(file_resume, "chat", chat_run1)
    with pytest.raises(RuntimeError):
        file_resume.resume_missing(src, base, resumed, api=API, log=lambda m: None, state_path=state)

    # checkpoint after file 1 exists, state is bound to it and lists only file 1
    assert resumed.exists()
    assert "中文（The quick" in _read(resumed, "EPUB/Text/alpha.xhtml")
    data = json.loads(state.read_text(encoding="utf-8"))
    assert data["done"] == ["EPUB/Text/alpha.xhtml"]
    assert data["base_sha1"] == file_resume._fingerprint(resumed)

    # rerun the way the pipeline does after the fix: base is the checkpoint itself
    prompts2: list[str] = []

    def chat_run2(api, prompt, **kwargs):
        prompts2.append(prompt)
        return _echo_chat(api, prompt)

    monkeypatch.setattr(file_resume, "chat", chat_run2)
    out, done = file_resume.resume_missing(src, resumed, resumed, api=API, log=lambda m: None, state_path=state)

    alpha = _read(out, "EPUB/Text/alpha.xhtml")
    beta = _read(out, "EPUB/Text/beta.xhtml")
    assert "中文（The quick" in alpha, "run-1 translation must survive the rerun"
    assert ALPHA_TEXT in alpha
    assert "中文（Neural networks" in beta
    assert not any("quick brown" in p for p in prompts2), "already-translated file must not be re-sent"
    assert "EPUB/Text/beta.xhtml" in done


def test_stale_or_legacy_state_does_not_block_translation(tmp_path, monkeypatch):
    """H1: a done list bound to a different base (or legacy format) is discarded."""
    src = _make_src(tmp_path)
    monkeypatch.setattr(file_resume, "chat", _echo_chat)

    # stale dict state pointing at some other base
    base1 = tmp_path / "base1.epub"
    shutil.copy2(src, base1)
    state1 = tmp_path / "state1.json"
    state1.write_text(
        json.dumps({"base_sha1": "0" * 40, "done": ["EPUB/Text/alpha.xhtml", "EPUB/Text/beta.xhtml"]}),
        encoding="utf-8",
    )
    out1 = tmp_path / "out1.epub"
    file_resume.resume_missing(src, base1, out1, api=API, log=lambda m: None, state_path=state1)
    assert "中文（The quick" in _read(out1, "EPUB/Text/alpha.xhtml")
    assert "中文（Neural networks" in _read(out1, "EPUB/Text/beta.xhtml")

    # legacy plain-list state must parse without crashing and be ignored
    base2 = tmp_path / "base2.epub"
    shutil.copy2(src, base2)
    state2 = tmp_path / "state2.json"
    state2.write_text(json.dumps(["alpha.xhtml", "beta.xhtml"]), encoding="utf-8")
    out2 = tmp_path / "out2.epub"
    file_resume.resume_missing(src, base2, out2, api=API, log=lambda m: None, state_path=state2)
    assert "中文（The quick" in _read(out2, "EPUB/Text/alpha.xhtml")
    assert "中文（Neural networks" in _read(out2, "EPUB/Text/beta.xhtml")


def test_translate_batch_number_variants_and_multiline(monkeypatch):
    """M5: "1." "1、" "1)" formats parse; multi-line answers join instead of truncating."""
    items = ["alpha one text", "beta two text", "gamma three text", "delta four text"]
    responses = iter(
        [
            "1. 译文一\n2、译文二\n3) 译文三继续\n仍是第三条",
            "第四条译文\n第二行继续",  # single-item fallback for the missing #4
        ]
    )
    calls: list[dict] = []

    def fake(api, prompt, **kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(file_resume, "chat", fake)
    out = file_resume._translate_batch(API, items)
    assert out[0] == "译文一"
    assert out[1] == "译文二"
    assert out[2] == "译文三继续 仍是第三条"
    assert out[3] == "第四条译文 第二行继续"
    # fallback budget is estimated from input length, not a fixed 400
    assert calls[1]["max_tokens"] >= 1000


def test_insertion_keeps_structure_and_ids_unique(monkeypatch):
    """M6: td/ol-li get in-place <br/>+译文; copied blocks drop their id."""
    monkeypatch.setattr(file_resume, "chat", _echo_chat)
    html = (
        '<html><body>'
        '<table><tr><td id="c1">English cell content sits here.</td></tr></table>'
        "<ol><li>First english item content here.</li></ol>"
        '<p id="p1">Standalone english paragraph content.</p>'
        "</body></html>"
    )
    out = file_resume.translate_html(html, api=API, log=lambda m: None, fname="c.xhtml")
    soup = BeautifulSoup(out, "html.parser")

    tds = soup.find_all("td")
    assert len(tds) == 1, "no sibling cell may be added"
    assert tds[0].find("br") is not None
    assert "中文（" in tds[0].get_text()

    lis = soup.find_all("li")
    assert len(lis) == 1, "no extra <li> may be added inside <ol>"
    assert "中文（" in lis[0].get_text()

    ps = soup.find_all("p")
    assert len(ps) == 2
    assert [p.get("id") for p in ps].count("p1") == 1, "copied paragraph must not duplicate the id"


def test_same_basename_in_different_dirs_gets_distinct_dests(tmp_path):
    src = tmp_path / "src.epub"
    bi = tmp_path / "bi.epub"
    body = "<html><body><p>The quick brown fox jumps over the lazy dog today.</p></body></html>"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        z.writestr("EPUB/content.opf", OPF)
        z.writestr("EPUB/Text/ch1.xhtml", body)
        z.writestr("EPUB/Notes/ch1.xhtml", body)
    with zipfile.ZipFile(bi, "w") as z:
        z.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        z.writestr("EPUB/content.opf", OPF)
        z.writestr("EPUB/Text/ch1.xhtml", body)

    missing = file_resume.missing_files(src, bi)
    dests = [d for _b, _s, d in missing]
    srcs = [s for _b, s, _d in missing]
    assert "EPUB/Notes/ch1.xhtml" in srcs
    assert len(set(dests)) == len(dests)
    notes = [d for b, s, d in missing if s == "EPUB/Notes/ch1.xhtml"]
    assert notes and notes[0] != "EPUB/Text/ch1.xhtml"


def test_checkpoint_matches_src(tmp_path):
    src = _make_src(tmp_path)
    other = tmp_path / "other.epub"
    shutil.copy2(src, other)
    other.write_bytes(other.read_bytes() + b"\0")
    state = tmp_path / "resume-done.json"
    file_resume._write_state(state, src, ["EPUB/Text/alpha.xhtml"], src)
    assert file_resume.checkpoint_matches_src(state, src)
    assert not file_resume.checkpoint_matches_src(state, other)
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps({"base_sha1": file_resume._fingerprint(src), "done": ["EPUB/Text/alpha.xhtml"]}),
        encoding="utf-8",
    )
    assert not file_resume.checkpoint_matches_src(legacy, src)


def _many_short_paragraphs(n: int = 150) -> str:
    body = "".join(
        f"<p>Short paragraph number {i:03d} is still chapter prose.</p>" for i in range(n)
    )
    return f"<html><body>{body}</body></html>"


def test_chapter_with_many_short_paragraphs_is_not_short_batch():
    html = _many_short_paragraphs()
    assert not file_resume.is_short_batch_name("Chapter_1.xhtml")
    assert not file_resume.is_short_batch_html("Chapter_1.xhtml", html)


def test_index_toc_references_are_short_batch_by_name():
    html = "<html><body><p>Anything at all can live here.</p></body></html>"
    for name in (
        "Index.xhtml",
        "toc.xhtml",
        "nav.xhtml",
        "references.html",
        "bibliography.xhtml",
        "contents.xhtml",
    ):
        assert file_resume.is_short_batch_name(name), name
        assert file_resume.is_short_batch_html(name, html), name


def test_short_batch_basenames_skips_chapters(tmp_path):
    src = tmp_path / "book.epub"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        z.writestr("OEBPS/Text/Chapter_1.xhtml", _many_short_paragraphs())
        z.writestr("OEBPS/Text/Index.xhtml", "<html><body><p>Index entry one here.</p></body></html>")
        z.writestr("OEBPS/Text/toc.xhtml", "<html><body><p>Chapter one title here.</p></body></html>")
    names = file_resume.short_batch_basenames(src)
    assert "Chapter_1.xhtml" not in names
    assert names == {"Index.xhtml", "toc.xhtml"}


def test_translate_batch_injects_only_appearing_terms_and_restores(monkeypatch):
    glossary = [
        Term("Sharpe ratio", "夏普比率", ("Sharpe",)),
        Term("absent-term", "缺席术语"),
    ]
    prompts: list[str] = []

    def fake(api, prompt, **kwargs):
        prompts.append(prompt)
        return "1. 计算 @@BT0@@ 即可"

    monkeypatch.setattr(file_resume, "chat", fake)
    out = file_resume._translate_batch(API, ["Compute the Sharpe ratio first."], glossary=glossary)
    assert out == ["计算 夏普比率 即可"]
    assert "Sharpe ratio" in prompts[0]
    assert "缺席术语" not in prompts[0]
    assert "absent-term" not in prompts[0]
    assert "【术语表】" in prompts[0]


def test_translate_html_skips_terms_only_inside_code(monkeypatch):
    glossary = [Term("Transformer", "变换器")]
    prompts: list[str] = []

    def fake(api, prompt, **kwargs):
        prompts.append(prompt)
        return "1. 请看代码示例"

    monkeypatch.setattr(file_resume, "chat", fake)
    html = (
        "<html><body><p>Please see the sample "
        "<code>Transformer</code> listed here.</p></body></html>"
    )
    file_resume.translate_html(html, api=API, log=lambda m: None, fname="c.xhtml", glossary=glossary)
    assert prompts
    assert "【术语表】" not in prompts[0]
    assert "变换器" not in prompts[0]


def _opf_spine(*ids: str, extra_manifest: str = "") -> str:
    items = "\n".join(
        f'    <item href="Text/{i}.xhtml" id="{i}" media-type="application/xhtml+xml"/>' for i in ids
    )
    refs = "\n".join(f'    <itemref idref="{i}"/>' for i in ids)
    return f"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Demo</dc:title>
    <dc:identifier id="id">urn:demo</dc:identifier>
  </metadata>
  <manifest>
{items}
    <item href="toc.ncx" id="ncx" media-type="application/x-dtbncx+xml"/>
{extra_manifest}  </manifest>
  <spine toc="ncx">
{refs}
  </spine>
</package>
"""


def _zh_pair(en: str) -> str:
    return f"<html><body><p>{en}</p><p>这段已经有中文对照了。</p></body></html>"


def test_href_for_dest_keeps_directory():
    assert file_resume.href_for_dest("EPUB/content.opf", "EPUB/Notes/ch1.xhtml") == "Notes/ch1.xhtml"
    assert file_resume.href_for_dest("OEBPS/content.opf", "OEBPS/Text/a.xhtml") == "Text/a.xhtml"
    assert file_resume.href_for_dest("EPUB/content.opf", "EPUB/Text/a.xhtml") == "Text/a.xhtml"


def test_resume_worker_count_defaults_and_caps(monkeypatch):
    monkeypatch.delenv("BOOK_TRANSLATE_RESUME_WORKERS", raising=False)
    assert file_resume.resume_worker_count() == 1
    monkeypatch.setenv("BOOK_TRANSLATE_RESUME_WORKERS", "3")
    assert file_resume.resume_worker_count() == 3
    assert file_resume.resume_worker_count("8") == 4
    assert file_resume.resume_worker_count("0") == 1
    assert file_resume.resume_worker_count("nope") == 1


def test_patch_opf_inserts_mid_chapter_in_source_order():
    opf = _opf_spine("cover", "alpha", "beta")
    out = file_resume._patch_opf(
        opf,
        ["EPUB/Text/mid.xhtml"],
        opf_zip_path="EPUB/content.opf",
        order_dests=[
            "EPUB/Text/cover.xhtml",
            "EPUB/Text/alpha.xhtml",
            "EPUB/Text/mid.xhtml",
            "EPUB/Text/beta.xhtml",
        ],
    )
    assert file_resume.spine_hrefs(out) == [
        "Text/cover.xhtml",
        "Text/alpha.xhtml",
        "Text/mid.xhtml",
        "Text/beta.xhtml",
    ]
    hrefs = file_resume.manifest_hrefs(out)
    assert hrefs.index("Text/mid.xhtml") < hrefs.index("Text/beta.xhtml")
    assert hrefs.index("Text/mid.xhtml") > hrefs.index("Text/alpha.xhtml")
    assert hrefs[-1] == "toc.ncx"
    assert 'toc="ncx"' in out


def test_patch_opf_keeps_full_path_for_same_basename():
    opf = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest>
    <item href="Text/ch1.xhtml" id="t1" media-type="application/xhtml+xml"/>
    <item href="toc.ncx" id="ncx" media-type="application/x-dtbncx+xml"/>
  </manifest>
  <spine toc="ncx">
    <itemref idref="t1"/>
  </spine>
</package>
"""
    out = file_resume._patch_opf(
        opf,
        ["EPUB/Notes/ch1.xhtml"],
        opf_zip_path="EPUB/content.opf",
        order_dests=["EPUB/Text/ch1.xhtml", "EPUB/Notes/ch1.xhtml"],
    )
    hrefs = file_resume.manifest_hrefs(out)
    assert "Notes/ch1.xhtml" in hrefs
    assert "ch1.xhtml" not in hrefs
    assert file_resume.spine_hrefs(out) == ["Text/ch1.xhtml", "Notes/ch1.xhtml"]


def test_resume_inserts_missing_chapter_in_source_spine_order(tmp_path, monkeypatch):
    ncx = (
        '<?xml version="1.0"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">'
        "<navMap><navPoint id=\"n1\" playOrder=\"1\"><navLabel><text>Alpha</text></navLabel>"
        "<content src=\"Text/alpha.xhtml\"/></navPoint></navMap></ncx>"
    )
    src = tmp_path / "book.epub"
    bi = tmp_path / "bi.epub"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        z.writestr("EPUB/content.opf", _opf_spine("cover", "alpha", "mid", "beta"))
        z.writestr("EPUB/toc.ncx", ncx)
        z.writestr("EPUB/Text/cover.xhtml", _zh_pair("Cover page explains the book at a glance."))
        z.writestr("EPUB/Text/alpha.xhtml", f"<html><body><p>{ALPHA_TEXT}</p></body></html>")
        z.writestr("EPUB/Text/mid.xhtml", "<html><body><p>The middle chapter explains gradient descent clearly.</p></body></html>")
        z.writestr("EPUB/Text/beta.xhtml", f"<html><body><p>{BETA_TEXT}</p></body></html>")
    with zipfile.ZipFile(bi, "w") as z:
        z.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        z.writestr("EPUB/content.opf", _opf_spine("cover", "alpha", "beta"))
        z.writestr("EPUB/toc.ncx", ncx)
        z.writestr("EPUB/Text/cover.xhtml", _zh_pair("Cover page explains the book at a glance."))
        z.writestr("EPUB/Text/alpha.xhtml", _zh_pair(ALPHA_TEXT))
        z.writestr("EPUB/Text/beta.xhtml", _zh_pair(BETA_TEXT))

    monkeypatch.setattr(file_resume, "chat", _echo_chat)
    out = tmp_path / "out.epub"
    file_resume.resume_missing(src, bi, out, api=API, log=lambda m: None, workers=1)
    opf = _read(out, "EPUB/content.opf")
    assert file_resume.spine_hrefs(opf) == [
        "Text/cover.xhtml",
        "Text/alpha.xhtml",
        "Text/mid.xhtml",
        "Text/beta.xhtml",
    ]
    assert file_resume.spine_hrefs(opf)[-1] != "Text/mid.xhtml"
    assert "中文（The middle" in _read(out, "EPUB/Text/mid.xhtml")
    assert _read(out, "EPUB/toc.ncx") == ncx
    assert 'playOrder="1"' in ncx
    assert "<text>Alpha</text>" in ncx
    assert 'toc="ncx"' in opf


def test_resume_workers_1_matches_serial_done_order(tmp_path, monkeypatch):
    src = _make_src(tmp_path)
    monkeypatch.setattr(file_resume, "chat", _echo_chat)
    base = tmp_path / "base.epub"
    shutil.copy2(src, base)
    out1 = tmp_path / "out1.epub"
    out2 = tmp_path / "out2.epub"
    _o1, done1 = file_resume.resume_missing(src, base, out1, api=API, log=lambda m: None, workers=1)
    shutil.copy2(src, base)
    _o2, done2 = file_resume.resume_missing(src, base, out2, api=API, log=lambda m: None)
    assert done1 == done2 == ["EPUB/Text/alpha.xhtml", "EPUB/Text/beta.xhtml"]
    assert "中文（The quick" in _read(out1, "EPUB/Text/alpha.xhtml")
    assert "中文（Neural networks" in _read(out1, "EPUB/Text/beta.xhtml")


def test_resume_workers_parallel_writes_then_marks_done(tmp_path, monkeypatch):
    import time

    src = tmp_path / "book3.epub"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        z.writestr("EPUB/content.opf", _opf_spine("alpha", "beta", "gamma"))
        z.writestr("EPUB/Text/alpha.xhtml", f"<html><body><p>{ALPHA_TEXT}</p></body></html>")
        z.writestr("EPUB/Text/beta.xhtml", f"<html><body><p>{BETA_TEXT}</p></body></html>")
        z.writestr(
            "EPUB/Text/gamma.xhtml",
            "<html><body><p>Gradient boosting builds trees from residual errors.</p></body></html>",
        )
    base = tmp_path / "base3.epub"
    shutil.copy2(src, base)
    out = tmp_path / "out3.epub"
    state = tmp_path / "resume-done.json"

    def fake(api, prompt, **kwargs):
        time.sleep(0.05)
        return _echo_chat(api, prompt)

    monkeypatch.setattr(file_resume, "chat", fake)
    _path, done = file_resume.resume_missing(
        src, base, out, api=API, log=lambda m: None, state_path=state, workers=3
    )
    assert set(done) == {
        "EPUB/Text/alpha.xhtml",
        "EPUB/Text/beta.xhtml",
        "EPUB/Text/gamma.xhtml",
    }
    assert "中文（The quick" in _read(out, "EPUB/Text/alpha.xhtml")
    assert "中文（Neural networks" in _read(out, "EPUB/Text/beta.xhtml")
    assert "中文（Gradient boosting" in _read(out, "EPUB/Text/gamma.xhtml")
    data = json.loads(state.read_text(encoding="utf-8"))
    assert set(data["done"]) == set(done)
    assert data["base_sha1"] == file_resume._fingerprint(out)
    assert file_resume.spine_hrefs(_read(out, "EPUB/content.opf")) == [
        "Text/alpha.xhtml",
        "Text/beta.xhtml",
        "Text/gamma.xhtml",
    ]


def test_resume_workers_do_not_mark_done_before_file_exists(tmp_path, monkeypatch):
    src = _make_src(tmp_path)
    base = tmp_path / "base.epub"
    shutil.copy2(src, base)
    out = tmp_path / "out.epub"
    state = tmp_path / "resume-done.json"

    def boom(api, prompt, **kwargs):
        if "Neural networks" in prompt:
            raise RuntimeError("network down")
        return _echo_chat(api, prompt)

    monkeypatch.setattr(file_resume, "chat", boom)
    with pytest.raises(RuntimeError):
        file_resume.resume_missing(
            src, base, out, api=API, log=lambda m: None, state_path=state, workers=2
        )
    data = json.loads(state.read_text(encoding="utf-8"))
    assert "EPUB/Text/beta.xhtml" not in data["done"]
    assert "EPUB/Text/alpha.xhtml" in data["done"]
    assert "中文（The quick" in _read(out, "EPUB/Text/alpha.xhtml")


def test_estimate_tokens_and_pack_batches():
    assert file_resume.estimate_tokens("") == 0
    assert file_resume.estimate_tokens("abcd") == 1
    assert file_resume.estimate_tokens("中文") == 2
    texts = ["aaaa"] * 10
    batches = file_resume.pack_batches(texts, token_budget=3, max_items=12)
    assert batches[0] == (0, 3)
    assert batches[-1][1] == 10
    huge = file_resume.pack_batches(["x" * 400], token_budget=10, max_items=12)
    assert huge == [(0, 1)]


def test_job_matches_basename_and_path():
    assert file_resume.job_matches("EPUB/Text/ch01.html", "ch01.html", ["ch01.html"])
    assert file_resume.job_matches("EPUB/Text/ch01.html", "ch01.html", ["Text/ch01.html"])
    assert not file_resume.job_matches("EPUB/Text/ch01.html", "ch01.html", ["ch02.html"])


def test_translate_html_keeps_single_head_title(monkeypatch):
    """<head> must keep exactly one <title>; translating it is title_postprocess's job.

    Inserting a translated sibling used to yield <title>EN</title><title>ZH</title>,
    which is invalid XHTML (epubcheck: head requires exactly one title).
    """
    monkeypatch.setattr(file_resume, "chat", _echo_chat)
    html = (
        "<html><head><title>My Great Book Chapter</title></head>"
        f"<body><p>{ALPHA_TEXT}</p></body></html>"
    )
    out = file_resume.translate_html(html, api=API, log=lambda m: None, fname="c.xhtml")
    soup = BeautifulSoup(out, "html.parser")
    titles = soup.head.find_all("title")
    assert len(titles) == 1, "head must keep exactly one <title>"
    assert titles[0].get_text() == "My Great Book Chapter"
    assert "中文（The quick" in soup.body.get_text(), "body translation must still happen"


def test_translate_html_respects_max_blocks(monkeypatch):
    prompts: list[str] = []

    def fake(api, prompt, **kwargs):
        prompts.append(prompt)
        return _echo_chat(api, prompt)

    monkeypatch.setattr(file_resume, "chat", fake)
    html = (
        "<html><body>"
        "<p>The quick brown fox jumps over the lazy dog today.</p>"
        "<p>Neural networks consist of many layers of neurons.</p>"
        "<p>Another english paragraph lives in this third block.</p>"
        "</body></html>"
    )
    out = file_resume.translate_html(
        html, api=API, log=lambda m: None, fname="c.xhtml", max_blocks=1
    )
    soup = BeautifulSoup(out, "html.parser")
    assert "Neural networks" in soup.get_text()
    assert "中文（The quick" in soup.get_text()
    assert "中文（Neural" not in soup.get_text()
    assert len(prompts) == 1


def test_translate_html_sends_rolling_context(monkeypatch):
    prompts: list[str] = []

    def fake(api, prompt, **kwargs):
        prompts.append(prompt)
        return _echo_chat(api, prompt)

    monkeypatch.setattr(file_resume, "chat", fake)
    html = (
        "<html><body>"
        "<p>The quick brown fox jumps over the lazy dog today.</p>"
        "<p>Neural networks consist of many layers of neurons.</p>"
        "</body></html>"
    )
    file_resume.translate_html(
        html,
        api=API,
        log=lambda m: None,
        fname="c.xhtml",
        batch_size=1,
        use_context=True,
        context_paragraphs=2,
        batch_tokens=50,
    )
    assert "【上文对照" not in prompts[0]
    assert "【上文对照" in prompts[1]
    assert "The quick brown fox" in prompts[1]
    assert "中文（The quick" in prompts[1]


def test_resume_only_and_retranslate(tmp_path, monkeypatch):
    monkeypatch.setattr(file_resume, "chat", _echo_chat)
    src = _make_src(tmp_path)
    base = tmp_path / "base.epub"
    shutil.copy2(src, base)
    out = tmp_path / "out.epub"
    file_resume.resume_missing(
        src, base, out, api=API, log=lambda m: None, only_files="beta.xhtml", workers=1
    )
    assert "中文（Neural" in _read(out, "EPUB/Text/beta.xhtml")
    assert "中文（The quick" not in _read(out, "EPUB/Text/alpha.xhtml")

    file_resume.resume_missing(
        src, out, out, api=API, log=lambda m: None, retranslate="beta.xhtml", workers=1
    )
    assert "中文（Neural" in _read(out, "EPUB/Text/beta.xhtml")


def test_resume_test_num_does_not_write_done(tmp_path, monkeypatch):
    monkeypatch.setattr(file_resume, "chat", _echo_chat)
    src = _make_src(tmp_path)
    base = tmp_path / "base.epub"
    shutil.copy2(src, base)
    out = tmp_path / "out.epub"
    state = tmp_path / "resume-done.json"
    _path, done = file_resume.resume_missing(
        src, base, out, api=API, log=lambda m: None, state_path=state, test_num=1
    )
    assert done == []
    assert not state.exists()
    assert "中文（The quick" in _read(out, "EPUB/Text/alpha.xhtml")
    assert "中文（Neural" not in _read(out, "EPUB/Text/beta.xhtml")
