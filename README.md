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

## CLI 参考

`twinfolio EPUB [选项]`（`book-translate` 是同一命令的别名）：

| 选项 | 作用 |
| --- | --- |
| `EPUB` | 源 `.epub` 路径（必填） |
| `-o, --out-dir DIR` | 输出目录，默认当前目录 |
| `--bilingual-only` | 只出双语版，跳过纯中文拆分 |
| `--debug` | 出错时打印完整 traceback（默认只打印错误链） |
| `--test` | 只翻正文前 N 段（默认 10），跳过标题补译和纯中文拆分，写出 `试译.epub` |
| `--test-num N` | 配合 `--test` 指定段数；单独给出时隐含 `--test`，须 `>= 1` |
| `--only a.html,b.html` | 只翻这些 HTML（文件名或 zip 内路径，逗号分隔） |
| `--retranslate a.html` | 强制重翻这些 HTML，即使已翻过 |
| `--use-context` | 把最近 N 段译文作为只读上下文随批发送 |
| `--context-paragraphs N` | 滚动上下文段数，默认 8（须配合 `--use-context`） |

退出码：`0` 成功，`1` 翻译失败，`2` 参数错误。

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
| `OPENAI_API_KEY` | 必填；也认旧名 `BBM_OPENAI_API_KEY`（正名优先） |
| `OPENAI_BASE_URL` | OpenAI 兼容接口，默认 `https://api.openai.com/v1` |
| `BOOK_TRANSLATE_MODEL` | 模型名，默认 `gpt-4o-mini`；也认旧名 `MODEL`（正名优先） |
| `BOOK_TRANSLATE_REASONING_EFFORT` | 推理强度，默认 `low`；也认旧名 `BBM_REASONING_EFFORT`（正名优先） |
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

## 故障排查

**HTTP 401 / 403（立即失败，不重试）**
密钥无效或没有该模型的权限。检查 `OPENAI_API_KEY`（或旧名 `BBM_OPENAI_API_KEY`）和 `OPENAI_BASE_URL` 是否匹配同一家服务商。非 429 的 4xx 都会立刻报错，不会烧重试次数。

**HTTP 429 / 5xx（自动重试）**
限流和服务端错误会按指数退避加抖动自动重试（默认 3 次），`Retry-After` 响应头会拉长等待。3 次仍失败时报 `chat failed after 3 attempts`。缓解：调低 `BOOK_TRANSLATE_RESUME_WORKERS`，稍后重跑——已翻内容都在缓存和断点里，不会重复付费。

**换模型后缓存全部失效**
`llm-cache.jsonl` 的键是 `sha1(model + prompt + 归一化原文)`，改 `BOOK_TRANSLATE_MODEL`（或提示词）后旧条目全部 miss、重新翻，属预期行为。旧条目留着无害；想瘦身可直接删掉该文件。

**改过源 EPUB 后断点失效**
`resume-done.json` 同时记录断点 EPUB 的 `base_sha1` 和源 EPUB 的 `src_sha1`。重新下载或编辑过源文件后 sha1 变化，断点作废、整本重翻，属预期行为。`--test` 试翻不写断点，部分翻译的文件保持可重试。

**输出里出现漏翻 / 想强制重翻某章**
用 `--retranslate ch03.html` 强制重做；只想补漏用 `--only`。整本重来就删 `-o` 下的 `.work-{书名}/` 目录。

## License

MIT
