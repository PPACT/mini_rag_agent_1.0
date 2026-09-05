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

    # 应用
    upload_dir: str = "./data/uploads"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    cache_ttl: int = 3600

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
