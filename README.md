# RAG-Demo（B端企业内部知识库）

本地部署的企业知识库 RAG-Agent Demo。

## 技术栈

| 层 | 选型 |
|---|---|
| Web | FastAPI + Uvicorn |
| 向量库 | PostgreSQL 17 + pgvector |
| 缓存 / 任务 | Redis + Arq |
| Embedding | 本地 Ollama `bge-m3`（CUDA） |
| 精排 | 本地 cross-encoder `BAAI/bge-reranker-base`（**默认**）；云端 LLM 作兜底 |
| LLM | DeepSeek `deepseek-v4-flash`（LiteLLM 配置收口，实际经 ChatOpenAI 调用） |
| Agent | LangGraph + MCP（Mock FastMCP 工具） |
| 文档解析 | pdfplumber / python-docx / python-pptx |

## 快速开始

0. **安装依赖**（conda 环境内，锁定版本）：
   ```bash
   uv pip install -r requirements.lock
   ```
1. **基础设施**：`docker compose up -d`（起 PG + Redis）
2. **Ollama + 模型**：`ollama serve` 后 `ollama pull bge-m3`
3. **建表 + 起服务**：
   ```bash
   # 建表（用 Alembic 迁移）
   alembic upgrade head
   # 起 API（PYTHONIOENCODING 让日志中文不乱码）
   PYTHONIOENCODING=utf-8 python -m uvicorn src.main:app
   # 起 Arq worker（另开终端）
   PYTHONIOENCODING=utf-8 python -m arq src.tasks.worker.WorkerSettings
   ```

## 环境变量

复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`。其余默认值可本地跑通。

鉴权用演示 token（见 `src/auth/deps.py` 的 `DEMO_USERS`）：
- `demo-it-token` → IT 部门 · 密级 3
- `demo-hr-token` → HR 部门 · 密级 2
- `demo-public-token` → 公开 · 密级 0

## 精排（Rerank）后端：默认本地，云端 LLM 兜底

检索是两段式：粗排召回 `RERANK_CANDIDATES` 条（默认 20）→ 精排取 `TOP_K` 条（默认 5）。
精排有两个后端，**默认用本地**：

| 后端 | 配置 | 特点 | 适用场景 |
|---|---|---|---|
| `local`（**默认**） | `RERANK_BACKEND=local` | 本地 cross-encoder（`BAAI/bge-reranker-base`）：输出确定、无网络依赖、延迟可预期 | 日常开发、演示、评测 |
| `llm`（兜底） | `RERANK_BACKEND=llm` | 走云端大模型（统一经 `src/config/litellm_client.py` 收口）：省本地模型，但慢且受网络/限流影响 | 本地模型未下载、显存不足、或做后端对比 |

**为什么默认本地**：除了快和确定，更重要的是**不被推理模型的思考链拖垮**——见下面这条坑。

> ⚠️ **用 `llm` 后端时切勿开启思考链**
> `deepseek-v4-flash` 这类推理模型会把 `reasoning_content`（思考）与 `content`（结论）
> **共用同一份 `max_tokens` 预算**。思考一旦吃满，`content` 就是空的 → 解析不到排序编号
> → **静默回退成粗排顺序，精排等于空转，且没有任何报错**。
> 本项目已用 `llm_thinking_enabled=False`（默认）关掉思考链；`local` 后端不过 LLM，不受影响。
> 另外注意：DeepSeek 的 `reasoning_effort` **除 `"none"` 外都是"开启思考"**，传 `"low"` 不是"少思考"。

切换方式（优先级从高到低）：

```bash
# ① 环境变量覆盖（临时，最优先）
RERANK_BACKEND=llm PYTHONIOENCODING=utf-8 python -m uvicorn src.main:app
# ② 写进 .env（推荐）
# ③ 改 src/config/settings.py 的 rerank_backend（代码默认值）
```

用 `local` 后端需本地已有模型权重；设 `HF_HUB_OFFLINE=1` 可跳过 HF 联网检查（启动快很多）。
模型是**懒加载**的：起服务后**第一个请求会多花约 12 秒**载入权重，之后每次精排约 0.1 秒。
起服务后打开 `/demo` 页面，顶部会直接显示当前生效的精排后端与模型名。

## 说明

- 本项目建议在 conda 隔离环境中开发，依赖安装使用 uv，运行使用隔离环境的 Python 解释器。
- 依赖锁定：`requirements.txt`（宽松手写）→ `requirements.lock`（锁定版本，用 `uv pip compile requirements.txt -o requirements.lock` 重新生成）。
- `VECTOR_STORE` 支持 `pgvector`（默认）/ `milvus`（占位桩），后期迁移 Milvus 只改配置。
- 文档上传、问答、软删等接口见 `src/api/`。

## 测试

单元测试（切块逻辑、AccessFilter→SQL 翻译）在 `tests/` 下，运行：

```bash
uv pip install pytest
PYTHONPATH=. python -m pytest tests/ -v
```
