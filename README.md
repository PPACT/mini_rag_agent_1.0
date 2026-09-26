<a id="readme-top"></a>

<!-- ============================================================
     徽章：用「引用式链接」写法 —— 正文只写 [![名][名-shield]][名-url]，
     所有 URL 集中在文件末尾（见「链接定义」）。改 URL 只改一处。
     此处只放**静态**徽章；不用 star / fork / contributors 这类依赖仓库统计的
     （单人项目会显示 0，反而难看）。
     ============================================================ -->
[![Python][py-shield]][py-url]
[![FastAPI][fastapi-shield]][fastapi-url]
[![PostgreSQL][pg-shield]][pg-url]
[![pgvector][pgv-shield]][pgv-url]
[![Redis][redis-shield]][redis-url]
[![Ollama][ollama-shield]][ollama-url]
[![LangGraph][lg-shield]][lg-url]
[![MCP][mcp-shield]][mcp-url]
[![Docker][docker-shield]][docker-url]

<br />

<!-- ============================================================
     头部：Markdown 做不到居中与图片尺寸，必须用 HTML
     ============================================================ -->
<div align="center">
  <h3 align="center">mini_rag_agent</h3>
  <p align="center">
    <b>B 端企业内部知识库 RAG-Agent</b> —— 权限隔离 · 混合检索 · 两段式精排 · 歧义澄清
    <br />
    知识库侧全本地可跑（PG + pgvector + 本地 embedding），仅生成走云端 LLM
    <br /><br />
    <a href="docs/架构方案.md"><strong>看架构设计 »</strong></a>
    &middot; <a href="#快速开始">快速开始</a>
    &middot; <a href="#已知边界">已知边界</a>
  </p>
</div>

<br />

<details>
  <summary><b>目录</b>（点击展开）</summary>
  <ol>
    <li><a href="#这是什么">这是什么</a></li>
    <li><a href="#快速开始">快速开始</a></li>
    <li><a href="#架构一览">架构一览</a></li>
    <li><a href="#目录结构">目录结构</a></li>
    <li><a href="#配置">配置</a></li>
    <li><a href="#接口一览">接口一览</a></li>
    <li><a href="#精排后端">精排（Rerank）后端</a></li>
    <li><a href="#两库分离">两库分离（真实库 / 压测库）</a></li>
    <li><a href="#已知边界">已知边界</a></li>
    <li><a href="#文档与测试">文档与测试</a></li>
  </ol>
</details>

---

<!-- 截图占位：拿到 `/demo` 链路透视图后，把图放到**被 git 跟踪**的目录
     （如 docs/assets/demo.png），然后取消下面两行的注释。 -->
<!--
![链路透视图][demo-screenshot]
-->

<!-- ============================================================ -->
## 这是什么

企业内部知识库的问答系统 Demo。用户上传文档（PDF / Word / PPT），用自然语言提问，
系统检索知识库、按需调用业务系统工具、生成**带引用溯源**的答案；
当检索到的候选**互相矛盾**时，**主动追问而不是擅自选一个**。

四个它特意做对的地方：

- **权限在服务端**：`department` / `secret_level` 一律取自鉴权 token，不信任客户端传参；
- **引用是真的**：`sources` 只返回 LLM **实际引用**的块，不是全部召回结果；
- **区分"找不到"和"不说"**：召回不相关时靠提示词兜底，而不是编造；
- **不隐瞒取舍**：每处技术选择都记了**理由和已被证伪的方向**，见 [`docs/开发记录.md`](docs/开发记录.md)。

<p align="right">(<a href="#readme-top">回到顶部</a>)</p>

---

## 快速开始

**前置**：Docker / Ollama / Python 3.11（建议 conda 隔离环境）／ 一个 DeepSeek API key。

```bash
# 0) 依赖（锁定版本）
uv pip install -r requirements.lock
```

```bash
# 1) 基础设施：PG(15432) + Redis(6379)
docker compose up -d
```

> 若要跑**压测库**（15433，见下文「两库分离」），需显式带 profile：
> `docker compose --profile stress up -d`

```bash
# 2) Ollama + 模型（⚠️ 本项目把 Ollama 跑在 11451，不是默认的 11434）
ollama serve
ollama pull bge-m3
```

```bash
# 3) 建表（两个库都要迁）
python scripts/migrate_all.py --kb both
# 只校验不迁移：python scripts/migrate_all.py --check
```

```bash
# 4) 起 API
PYTHONIOENCODING=utf-8 python -m uvicorn src.main:app
# 起 Arq worker（另开一个终端）
PYTHONIOENCODING=utf-8 python -m arq src.tasks.worker.WorkerSettings
```

打开 <http://127.0.0.1:8000/demo> 是**链路透视页**：一次问答的候选、精排前后、耗时分解
都会显示出来，顶部还会标明**本次查询实际落在哪个库、用的哪个精排后端**。

<p align="right">(<a href="#readme-top">回到顶部</a>)</p>

---

## 架构一览

```
入口      /chat（业务问答） · /documents（上传/状态） · /demo（链路透视）
检索      retriever（混合检索 + RRF 融合）→ reranker（两段式精排）→ ambiguity（歧义判定/澄清）
抽象      VectorStore / BaseEmbedding / BaseReranker —— 各引擎各自实现，业务代码不感知
数据      PostgreSQL 17 + pgvector（向量 + 全文检索同库）；Redis（缓存 + 任务队列）
异步      Arq worker：解析 → 结构优先切块 → 本地 embedding → 原子替换切片
```

两条链路的顺序**不同**，且**只有 `/chat` 读写缓存**：

| 入口 | 顺序 |
|---|---|
| `/chat` | 缓存 → 检索 → 歧义判定 → (澄清 ｜ 生成) → 引用校验 → 写缓存 |
| `/demo/ask` | 检索 → 生成 → 歧义判定（**无缓存**） |

> 完整设计（分层 / 数据模型 / 真实链路 / 扩展点）见 [`docs/架构方案.md`](docs/架构方案.md)。

<p align="right">(<a href="#readme-top">回到顶部</a>)</p>

---

## 目录结构

```
src/
├── api/              chat_api · upload_api · demo_api（含链路透视 WebUI）
├── rag/              retriever · reranker · ambiguity · query_rewriter
├── vector_store/     base(抽象+权限过滤) · pg_vector(实现) · milvus(占位)
├── embedding/        base(抽象) · ollama_embedding · cloud_embedding(占位)
├── document_parser/  loader(多格式) · semantic_splitter(结构优先切块) · tokenizer(jieba)
├── agent/            graph_builder(LangGraph) · mcp_client_wrapper · mcp_mock_server
├── db/               kb(两库标识) · connection(按 kb 的连接池)
├── tasks/            queue · worker · document_task
├── config/           settings · litellm_client · prompts
└── auth/ cache/ schemas/ observability/
alembic/versions/     0001 建表 · 0002 切片元数据 · 0003 全文检索
eval/                 语料 + 标注数据集 + 评测脚本
prompts/rag_system.yaml   提示词（外置，便于迭代）
docs/                 开发文档
```

<p align="right">(<a href="#readme-top">回到顶部</a>)</p>

---

## 配置

复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`。其余默认值可本地跑通。

几个**必须显式设**的（默认值会连不上或走错）：

| 键 | 说明 |
|---|---|
| `OLLAMA_BASE_URL` | 本项目跑在 `http://localhost:11451`（非默认 11434） |
| `DEEPSEEK_API_KEY` | 空值 → `/chat` 直接返回 503（入库链路不受影响） |
| `RERANK_BACKEND` | `local`（默认）/ `llm`，见下节 |

鉴权用演示 token（真实系统替换为 JWT / SSO）：

| token | 身份 |
|---|---|
| `demo-it-token` | IT 部门 · 密级 3 |
| `demo-hr-token` | HR 部门 · 密级 2 |
| `demo-public-token` | 公开 · 密级 0 |

<p align="right">(<a href="#readme-top">回到顶部</a>)</p>

---

## 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/chat` | 业务问答（固定查真实库；有缓存） |
| `POST` | `/documents/upload` | 上传文档（异步解析；可指定目标库） |
| `GET` | `/documents/{id}` | 查询处理状态 |
| `POST` | `/demo/ask` | 链路透视（返回候选、精排前后、耗时分解） |
| `GET` | `/demo` | 链路透视 WebUI |
| `GET` | `/health` | 健康检查 |

除 `/health` 与 `/demo*` 外都需 `Authorization: Bearer <token>`。

<p align="right">(<a href="#readme-top">回到顶部</a>)</p>

---

<!-- ============================================================
     这一节用 <details> 折叠：内容重要但很长，不该占首屏
     这正是 Best-README-Template 里最实用的一个技法
     ============================================================ -->
## 精排后端

检索是两段式：粗排召回 `RERANK_CANDIDATES` 条（默认 20）→ 精排取 `TOP_K` 条（默认 5）。

<details>
<summary><b>两个后端的取舍与切换方式</b>（点击展开）</summary>

<br />

| 后端 | 配置 | 特点 |
|---|---|---|
| `local`（**默认**） | `RERANK_BACKEND=local` | 本地 cross-encoder（`BAAI/bge-reranker-base`）：输出确定、无网络依赖、延迟可预期、零 API 成本 |
| `llm` | `RERANK_BACKEND=llm` | 走云端大模型（统一经 `src/config/litellm_client.py` 收口）：省本地模型，但受网络与限流影响 |

> **两个后端都可用，取舍取决于你的场景**（本地模型是否已下载 / 能否接受云端依赖 / 是否要做后端对照）。
> 本项目**没有**在此宣称哪个"更好"——需要哪个，用配置切，并用你自己的语料评测。

切换优先级（从高到低）：

```bash
# ① 环境变量覆盖（临时，最优先）
RERANK_BACKEND=llm PYTHONIOENCODING=utf-8 python -m uvicorn src.main:app
# ② 写进 .env（推荐）
# ③ 改 src/config/settings.py 的 rerank_backend（代码默认值）
```

用 `local` 需本地已有模型权重；设 `HF_HUB_OFFLINE=1` 可跳过 HF 联网检查（启动快很多）。
模型**懒加载**：起服务后第一个请求会多花约十几秒载入权重，之后每次精排很快。

> ⚠️ **用 `llm` 后端时，切勿开启思考链**
> `deepseek-v4-flash` 这类推理模型会把 `reasoning_content`（思考）与 `content`（结论）
> **共用同一份 `max_tokens` 预算**。思考一旦吃满，`content` 就是空的 → 解析不到排序编号
> → **静默回退成粗排顺序，精排等于空转，且没有任何报错**。
> 本项目已用 `llm_thinking_enabled=False`（默认）关掉思考链；`local` 后端不过 LLM，不受影响。
> 另外注意：DeepSeek 的 `reasoning_effort` **除 `"none"` 外都是"开启思考"**，传 `"low"` 不是"少思考"。

</details>

<p align="right">(<a href="#readme-top">回到顶部</a>)</p>

---

## 两库分离

评测用的合成语料里有大量**刻意造出的近重复文档**（同一份制度的几十个微差版本）。
若与真实文档同库，用户问真实业务问题时 top-5 会被这些干扰版本淹没。
所以**两个库物理隔离**：`rag_real`(15432) / `rag_stress`(15433)。

| 入口 | 目标库 |
|---|---|
| `/chat` | **固定真实库**（业务入口） |
| `/demo/ask`、`/upload` | 可选，默认真实库（压测库需显式指定） |

设计上 `kb` 是**必填的一等信息**：库层（连接池 / 向量库 / 检索 / 任务）**一律无默认值**，
忘传即报错；默认值**只出现在最外层 HTTP 契约**，显式可见、可被调用方覆盖。

> 压测库用 compose `profiles` 按需启，日常 `up -d` 不会启动它（开销为 0）。
> 迁移请用 `python scripts/migrate_all.py --kb both`——它会**两个库都迁**并逐表比对 schema；
> 直接 `alembic upgrade head` 只会迁一个库。

<p align="right">(<a href="#readme-top">回到顶部</a>)</p>

---

## 已知边界

这是一个 **Demo**，以下**未实现**（写出来是为了避免误解）：
多轮对话 · 流式输出 · 检索未包装成 Agent 工具 · 扫描件 OCR · 表格内容未入库 ·
Prompt 注入防护 · OTel 真埋点 · 水平扩展 · 前端界面。

> ⚠️ 另：文档解析目前只读段落文本，**表格里的内容不会入库**——这对表格密集的制度文档是明确缺陷。

<p align="right">(<a href="#readme-top">回到顶部</a>)</p>

---

## 文档与测试

| 文档 | 内容 |
|---|---|
| [`docs/架构方案.md`](docs/架构方案.md) | 系统设计：定位 / 选型 / 分层 / 数据模型 / 真实链路 / 扩展点 |
| [`docs/开发记录.md`](docs/开发记录.md) | 搭建过程、踩坑记录、关键设计决策的**理由** |
| [`docs/README.md`](docs/README.md) | 文档目录的边界与放置规则 |

```bash
# 单元测试：切块逻辑、权限过滤 → SQL 翻译、RRF 融合、引用解析
PYTHONPATH=. python -m pytest tests/ -v
```

`eval/` 下有评测语料、标注数据集与评测脚本（Recall@K / MRR / 漏报误报 / 改述一致率）。

<p align="right">(<a href="#readme-top">回到顶部</a>)</p>

---

<!-- ============================================================
     链接定义：所有外部 URL 集中在此处。
     改链接 / 换徽章只动这里，正文不用碰。
     ============================================================ -->

[py-shield]: https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white
[py-url]: https://www.python.org/
[fastapi-shield]: https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white
[fastapi-url]: https://fastapi.tiangolo.com/
[pg-shield]: https://img.shields.io/badge/PostgreSQL_17-4169E1?style=for-the-badge&logo=postgresql&logoColor=white
[pg-url]: https://www.postgresql.org/
[pgv-shield]: https://img.shields.io/badge/pgvector-336791?style=for-the-badge
[pgv-url]: https://github.com/pgvector/pgvector
[redis-shield]: https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white
[redis-url]: https://redis.io/
[ollama-shield]: https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white
[ollama-url]: https://ollama.com/
[lg-shield]: https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white
[lg-url]: https://langchain-ai.github.io/langgraph/
[mcp-shield]: https://img.shields.io/badge/MCP-000000?style=for-the-badge
[mcp-url]: https://modelcontextprotocol.io/
[docker-shield]: https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white
[docker-url]: https://www.docker.com/

<!-- 有截图后取消 README 顶部的注释，并把图放到被 git 跟踪的目录 -->
[demo-screenshot]: docs/assets/demo.png
