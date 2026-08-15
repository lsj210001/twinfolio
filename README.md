# TwinFolio

**English EPUB in. Facing-page Chinese out.**

把英文 EPUB 翻成 **中英对照** 和 **纯中文** 两份。

## 安装

```bash
python3 -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

## 用法

```bash
export OPENAI_API_KEY=sk-...
# export OPENAI_BASE_URL=https://api.openai.com/v1
export BOOK_TRANSLATE_MODEL=gpt-4o-mini

twinfolio "/path/to/book.epub" -o ./out

# 试翻前 10 段，写出 试译.epub
twinfolio "/path/to/book.epub" -o ./out --test
twinfolio "/path/to/book.epub" -o ./out --test-num 20

# 只翻或重翻指定 HTML
twinfolio "/path/to/book.epub" -o ./out --only ch01.html,ch02.html
twinfolio "/path/to/book.epub" -o ./out --retranslate ch03.html

# 滚动上下文（默认关）
twinfolio "/path/to/book.epub" -o ./out --use-context --context-paragraphs 8
```

输出：

- `book-中英双语.epub`
- `book-中文.epub`

工作目录在 `-o` 下的 `.work-{书名}/`：

- `*-resumed.epub` + `resume-done.json`：文件级断点，按源 EPUB 的 sha1 绑定
- `llm-cache.jsonl`：段落级翻译缓存
- `glossary.json`：全书术语表，可手改；已有文件不会被覆盖

整本重翻就删这个目录。只重翻、保留术语表时，删 `resume-done.json`、`*-resumed.epub`、`llm-cache.jsonl`。

## 环境变量

见 `config.example.env`。密钥走环境变量，程序不会自动加载 `.env`。

| 环境变量 | 作用 |
| --- | --- |
| `OPENAI_API_KEY` | 必填 |
| `OPENAI_BASE_URL` | OpenAI 兼容接口，默认 `https://api.openai.com/v1` |
| `BOOK_TRANSLATE_MODEL` | 模型名，默认 `gpt-4o-mini` |
| `BOOK_TRANSLATE_REASONING_EFFORT` | 推理强度 |
| `BOOK_TRANSLATE_RESUME_WORKERS` | 按文件翻译线程数，默认 `1`，最大 `4` |
| `SHORT_BATCH_SIZE` | 短文件每批最多段数，默认 `8` |
| `BOOK_TRANSLATE_BATCH_TOKENS` | 按估算 token 攒批，默认 `1600`，短文件 `600` |
| `BOOK_TRANSLATE_USE_CONTEXT` | 设为 `1` 时开启滚动上下文 |

## 术语表

`.work-{书名}/glossary.json`：

```json
{
  "terms": [
    {"source": "Sharpe ratio", "target": "夏普比率", "aliases": ["Sharpe"]}
  ]
}
```

正文和标题共用这张表。只注入本批实际出现的术语，译前占位、译后还原。`<code>` / `<pre>` / `<sup>` 内的文本不占位。

## 流水线

1. 按文件翻译正文（术语表 + 段落缓存；`--test` 只翻前 N 段）
2. 标题 / TOC 补译
3. 去重
4. 从双语拆出纯中文
5. 修复目录（NCX + EPUB3 `nav.xhtml`）

结束时打印本进程 LLM token 汇总。

## License

MIT
