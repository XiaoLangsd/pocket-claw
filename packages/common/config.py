from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
from functools import lru_cache

# 1. 定义寻根函数 (加上 lru_cache 缓存结果，提高性能)
@lru_cache(maxsize=1)
def get_project_root() -> Path:
    current_dir = Path(__file__).resolve().parent
    for parent in [current_dir, *current_dir.parents]:
        # 找标志性文件，存在说明就是根目录
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return current_dir

BASE_DIR = get_project_root()
ENV_FILE = BASE_DIR / ".env"

class LLMSettings(BaseSettings):
    """大语言模型配置 (DeepSeek)"""
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    temperature: float = 0.7
    max_tokens: int = 4096

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file = str(ENV_FILE),
        env_file_encoding = "utf-8",
        extra="ignore",
    )

class EmbeddingSettings(BaseSettings):
    """嵌入向量模型配置 (千问)"""
    api_key: str
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "text-embedding-v3"
    dimension: int = 1024

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file = str(ENV_FILE),
        env_file_encoding = "utf-8",
        extra="ignore",
    )

class Settings(BaseSettings):
    """应用总配置"""
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)

    # 数据库
    database_url: str = "postgresql+asyncpg://postgres:chao123@localhost:5432/postgres"
    redis_url: str = "redis://localhost:6379"

    # 向量数据库
    chroma_path: str = "./chroma_data"
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 4

    # 文件上传
    upload_dir: str = "./uploads"
    max_file_size: int = 10 * 1024 * 1024

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()

# 启动时确保目录存在
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)