import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from packages.common.config import settings


DATABASE_URL = settings.DATABASE_URL


logger = logging.getLogger(__name__)

# 1. 创建异步引擎
engine = create_async_engine(
    DATABASE_URL,
    echo=False,           # 生产环境建议 False，调试时可设为 True 看 SQL
    pool_size=20,         # 连接池大小
    max_overflow=10,      # 超过 pool_size 后最多再创建的连接数
    pool_recycle=3600,    # 连接回收时间（秒），防止 MySQL/PostgreSQL 自动断开
    pool_pre_ping=True,   # 每次从连接池取连接时先 ping 一下，防止死连接
)

# 2. 创建异步 Session 工厂
# expire_on_commit=False 很重要：提交后对象依然可用，否则访问属性会触发新的查询
async_session_maker = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

# 3. 依赖注入：FastAPI 路由用的 DB 会话
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()