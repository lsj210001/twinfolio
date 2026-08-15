import json

from bs4 import BeautifulSoup

from book_translate.glossary import (
    Term,
    apply_glossary,
    format_hint,
    inject_glossary_into_prompt,
    load_glossary,
    protect_html,
    protect_text,
    restore_text,
    terms_for_prompt,
    visible_text,
)


TERMS = [
    Term("Sharpe ratio", "夏普比率", ("Sharpe",)),
    Term("DiT", "DiT"),
    Term("never-used-term", "不会出现"),
]


def test_protect_and_restore_roundtrip():
    src = "The Sharpe ratio and DiT model are discussed."
    protected, mapping = protect_text(src, TERMS)
    assert "Sharpe" not in protected
    assert "DiT" not in protected
    assert "@@BT" in protected
    assert restore_text(protected, mapping, TERMS) == "The 夏普比率 and DiT model are discussed."


def test_restore_hard_locks_leftover_source():
    mapping = [("@@BT0@@", "夏普比率")]
    assert restore_text("计算 Sharpe ratio 即可", mapping, [TERMS[0]]) == "计算 夏普比率 即可"


def test_unused_terms_are_not_injected():
    items = ["The DiT backbone is trained on video."]
    protected, maps, hint, appearing = apply_glossary(items, TERMS)
    assert [t.source for t in appearing] == ["DiT"]
    assert "never-used-term" not in hint
    assert "Sharpe" not in hint
    assert "DiT" in hint
    assert "@@BT" in protected[0]
    assert maps[0]


def test_match_texts_control_which_terms_appear():
    items = ["See the Sharpe ratio in the appendix."]
    protected, _maps, hint, appearing = apply_glossary(
        items, TERMS, match_texts=["no glossary hits in visible text"]
    )
    assert appearing == []
    assert hint == ""
    assert protected == items


def test_protect_html_skips_exclude_tags():
    html = "<p>Use Transformer in prose <code>Transformer</code> and <pre>Transformer</pre> plus <sup>Transformer</sup>.</p>"
    terms = [Term("Transformer", "变换器")]
    out, mapping = protect_html(html, terms)
    soup = BeautifulSoup(out, "html.parser")
    assert soup.find("code").get_text() == "Transformer"
    assert soup.find("pre").get_text() == "Transformer"
    assert soup.find("sup").get_text() == "Transformer"
    assert "Transformer" not in visible_text(soup)
    assert mapping
    assert "@@BT" in soup.get_text()


def test_load_glossary_does_not_overwrite_user_file(tmp_path):
    path = tmp_path / "glossary.json"
    original = {
        "terms": [{"source": "GAN", "target": "生成对抗网络", "aliases": ["GANs"]}]
    }
    path.write_text(json.dumps(original, ensure_ascii=False, indent=2), encoding="utf-8")
    terms = load_glossary(path)
    assert [t.source for t in terms] == ["GAN"]
    assert json.loads(path.read_text(encoding="utf-8")) == original


def test_load_glossary_writes_empty_template_when_missing(tmp_path):
    path = tmp_path / "glossary.json"
    assert load_glossary(path) == []
    assert json.loads(path.read_text(encoding="utf-8")) == {"terms": []}


def test_load_glossary_leaves_corrupt_file_alone(tmp_path):
    path = tmp_path / "glossary.json"
    path.write_text("{not-json", encoding="utf-8")
    assert load_glossary(path) == []
    assert path.read_text(encoding="utf-8") == "{not-json"


def test_inject_glossary_into_prompt_inserts_before_text():
    prompt = {"system": "s", "user": "hello\n\n{text}"}
    out = inject_glossary_into_prompt(prompt, [TERMS[0]])
    assert "Sharpe ratio" in out["user"]
    assert "夏普比率" in out["user"]
    assert "【术语表】" in out["user"]
    assert out["user"].index("【术语表】") < out["user"].index("{text}")
    assert prompt["user"] == "hello\n\n{text}"


def test_inject_glossary_into_prompt_noop_when_empty():
    prompt = {"user": "hello {text}"}
    out = inject_glossary_into_prompt(prompt, [])
    assert out["user"] == "hello {text}"
    assert "【术语表】" not in out["user"]


def test_terms_for_prompt_caps_length():
    terms = [Term(f"Term{i:03d}", f"译{i:03d}") for i in range(200)]
    chosen = terms_for_prompt(terms, max_chars=200)
    assert chosen
    assert len(chosen) < 200
    assert len(chosen) == 1 or len(format_hint(chosen)) <= 200
