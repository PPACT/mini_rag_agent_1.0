"""全局配置：pydantic-settings 读取 .env。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（src/config/settings.py -> 上三级），避免依赖 cwd 漂移
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM（DeepSeek 云 API）
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # Embedding（本地 Ollama）
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "bge-m3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 16

    # PostgreSQL + pgvector
    database_url: str = "postgresql://rag:rag_demo_pwd@localhost:5432/rag_demo"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # 向量库开关（pgvector | milvus）
    vector_store: str = "pgvector"

    # 范围模型：公司级文档的部门标记（全员可见）。
    # 用户可见范围 = 公司级 + 本部门 —— 少了它，全员该看的制度反而看不到。
    company_scope: str = "公司"

    # 范围过滤：硬 vs 软
    # 硬过滤（默认，安全）：范围外完全排除。元数据一旦标错/缺失 → 永久静默搜不到。
    # 软过滤：范围内权重 1.0、范围外 ×penalty 降权但不排除 → 标错时文档仍能找到（排后）。
    # 高密级场景仍需硬隔离（secret_level 永远硬过滤），此开关只影响「范围偏好」。
    scope_soft_enabled: bool = False
    scope_soft_penalty: float = 0.85   # 实测：0.6 降权过重致标错文档掉出候选，0.85 可使其回到第 2

    # 应用
    upload_dir: str = "./data/uploads"
    max_upload_size: int = 50 * 1024 * 1024  # 上传大小上限（默认 50MB）
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    cache_ttl: int = 3600

    # 多查询扩展（Query Rewriting）
    # 实测（清晰题 80 条）：单独使用零增益；与精排组合后关掉它 Recall@1 反而 0.800→0.850、MRR 0.872→0.896。
    # 且它是链路里的一级 LLM（4→3 级），关掉可减少方差。故默认关闭，保留开关备用。
    query_rewrite_enabled: bool = False
    query_rewrite_count: int = 3       # 生成的改写变体数（不含原问题）
    rrf_k: int = 60                    # RRF 融合常数

    # 两段式检索：粗排召回 → 精排（Rerank）
    rerank_enabled: bool = True
    rerank_candidates: int = 20        # 粗排召回的候选数（精排后取 top_k）
    rerank_snippet_chars: int = 300    # 送进精排时每条的截断长度（控成本）

    # 精排后端：llm = 云端 LLM（实测 ~9 秒/次）｜ local = 本地 cross-encoder（~0.1 秒、确定性）
    rerank_backend: str = "llm"
    rerank_local_model: str = "BAAI/bge-reranker-base"
    rerank_local_device: str = ""      # 空 = 自动（有 CUDA 用 cuda，否则 cpu）

    # 混合检索（向量 + 全文检索 RRF 融合，兜底"精确词"查询）
    hybrid_search_enabled: bool = True

    # HNSW 检索深度。pgvector 默认 40，实测在 1231 切片时**损失 12% 召回**，
    # 提到 100 仅多 ~3ms 即可恢复满召回 —— 默认值会静默降低检索质量。
    hnsw_ef_search: int = 100

    # LLM 思考链（DeepSeek 推理模型专用开关）
    # 实测（2026-09-19）：**判定/精排这类分类任务，开思考是有害的** ——
    #   ① 思考（reasoning_content）与结论（content）**共用 max_tokens 预算**，
    #      思考一旦吃满预算 → finish_reason=length、content 为空 →
    #      parse_result("") 会**静默返回"不歧义"** → 漏报（实测有歧义题被整题漏掉）。
    #   ② 慢：开思考 4~10 秒 ｜ 关掉 0.7~1.4 秒。
    # ⚠️ litellm 语义坑：DeepSeek 的 `reasoning_effort` **除 "none" 外都是开启思考**——
    #    传 "low" 不是"少思考"，而是"开启思考"。要关只有 "none"。
    llm_thinking_enabled: bool = False

    # 歧义判定（避免"候选互相矛盾却擅自选一个"）
    ambiguity_check_enabled: bool = True
    ambiguity_source_threshold: int = 3   # top-K 来自 >=N 个不同来源才触发 LLM 判定（省调用）
    ambiguity_snippet_chars: int = 300

    @property
    def upload_dir_abs(self) -> str:
        """上传目录绝对路径（upload_dir 相对路径以项目根为基准）。"""
        p = Path(self.upload_dir)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return str(p)


@lru_cache
def get_settings() -> Settings:
    """返回单例配置。"""
    return Settings()
