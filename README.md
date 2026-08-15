# TwinFolio

**English EPUB in. Facing-page Chinese out.**

把英文 EPUB 翻成 **中英对照** 和 **纯中文** 两份。命令行工具，不依赖阅读器插件，也不走 `ebooklib` / `bilingual_book_maker`。

## 安装

```bash
python3 -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

## 用法

```bash
export OPENAI_API_KEY=sk-...
# 可选：兼容 OpenAI 的中转
# export OPENAI_BASE_URL=https://api.openai.com/v1
export BOOK_TRANSLATE_MODEL=gpt-4o-mini

twinfolio "/path/to/book.epub" -o ./out

# 先试翻前 10 段（写出 试译.epub，不改断点，不跑标题/纯中文）
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

- `*-resumed.epub` + `resume-done.json`：文件级断点。绑的是源 EPUB 的 sha1，换书或改源会自动作废。
- `llm-cache.jsonl`：段落级翻译缓存。改模型或 prompt 会失效；半写/空文件不认。写入时加锁并以临时文件 `rename`，避免并发撕坏 jsonl。
- `glossary.json`：全书术语表，可手改。程序只在文件不存在时写入空模板，**不会覆盖**已有文件。

要整本重翻：删掉这个目录（会一并丢掉术语表和段落缓存）。只想重翻、保留术语表时，删 `resume-done.json` / `*-resumed.epub` / `llm-cache.jsonl` 即可。

## 环境变量

见 `config.example.env`。密钥只走环境变量，不要写进命令行。程序**不会**自动加载 `.env`。

| 环境变量 | 作用 |
| --- | --- |
| `OPENAI_API_KEY` | 必填。也认旧名 `BBM_OPENAI_API_KEY` |
| `OPENAI_BASE_URL` | OpenAI 兼容接口，默认 `https://api.openai.com/v1` |
| `BOOK_TRANSLATE_MODEL` | 模型名，默认 `gpt-4o-mini` |
| `BOOK_TRANSLATE_REASONING_EFFORT` | 推理强度（也认旧名 `BBM_REASONING_EFFORT`） |
| `BOOK_TRANSLATE_RESUME_WORKERS=2` | 按文件翻译线程数（默认 `1` 串行，最大 `4`） |
| `SHORT_BATCH_SIZE=8` | index / references / toc / nav 等短文件每批最多段数 |
| `BOOK_TRANSLATE_BATCH_TOKENS=1600` | 按估算 token 攒批；短文件默认 600 |
| `BOOK_TRANSLATE_USE_CONTEXT=1` | 等价于 `--use-context` |

`BOOK_TRANSLATE_RESUME_WORKERS` 默认仍串行；开 2–4 时翻译可并行，但每个文件仍是先写入 EPUB 再记入 `resume-done.json`。并发时会串行化真实 API 调用（约 80ms 间隔），并复用 `llm-cache.jsonl` 的文件锁。

## 术语表

`.work-{书名}/glossary.json` 第一版只支持手写，例如：

```json
{
  "terms": [
    {"source": "Sharpe ratio", "target": "夏普比率", "aliases": ["Sharpe"]}
  ]
}
```

正文与标题补译共用这张表：只把**本批实际出现**的术语注入 prompt，并在译前占位、译后还原。`<code>` / `<pre>` / `<sup>` 内的文本不参与占位。真正锁定靠代码，不靠模型自觉。

## 流水线

1. 按文件翻译正文（按估算 token 攒批，短文件更小；缺章按源 spine 插回；术语表 + 段落缓存。`--test` 只翻前 N 段并写出 `试译.epub`）
2. 标题/TOC 补译（已有中文对照的不改；同一张术语表和缓存）
3. 去重
4. 从双语拆纯中文（改 zip，不走 ebooklib NCX）
5. 修复目录：NCX 必须保留 `<navLabel><text>` 和 `playOrder`，并补 EPUB3 `nav.xhtml`（否则阅读器书签会变成 1、2、3）

结束时打印本进程 LLM token 汇总（`input` / `output` / `total` / `calls`）。

## 不是什么

- 不是阅读器插件，也不提供 HTTP 服务或容器镜像
- 不依赖 `bilingual_book_maker` / `bbook_maker` / `ebooklib`

## License

MIT
