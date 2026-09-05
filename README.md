# RAG-Demo（B端企业内部知识库）

本地部署的企业知识库 RAG-Agent Demo。

## 技术栈

| 层 | 选型 |
|---|---|
| Web | FastAPI + Uvicorn |
| 向量库 | PostgreSQL 17 + pgvector |
| 缓存 / 任务 | Redis + Arq |
| Embedding | 本地 Ollama `bge-m3`（RTX 4060 CUDA） |
| LLM | DeepSeek `deepseek-chat`（LiteLLM 统一收口） |
| Agent | LangGraph + MCP（Mock FastMCP 工具） |
| 文档解析 | pdfplumber / python-docx / python-pptx |

## 三步启动

1. **基础设施**：`docker compose up -d`（起 PG + Redis）
2. **Ollama + 模型**：`ollama serve` 后 `ollama pull bge-m3`
3. **建表 + 起服务**：
   ```bash
   # 建表（documents / chunks 向量表）
   psql "postgresql://rag:rag_demo_pwd@localhost:5432/rag_demo" -f src/db/init_db.sql
   # 起 API（PYTHONIOENCODING 让日志中文不乱码）
   PYTHONIOENCODING=utf-8 python -m uvicorn src.main:app
   # 起 Arq worker（另开终端）
   PYTHONIOENCODING=utf-8 python -m arq src.tasks.worker.WorkerSettings
   ```

## 环境变量

复制 `.env`，填入 `DEEPSEEK_API_KEY`。其余默认值可本地跑通。

## 说明

- 本项目建议在 conda 隔离环境中开发，依赖安装使用 uv，运行使用隔离环境的 Python 解释器。
- `VECTOR_STORE` 支持 `pgvector`（默认）/ `milvus`（占位桩），后期迁移 Milvus 只改配置。
- 文档上传、问答、软删等接口见 `src/api/`。
