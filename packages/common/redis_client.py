import redis.asyncio as aioredis
from packages.common.config import settings

# 创建全局异步 redis 连接池
redis_pool = aioredis.ConnectionPool.from_url(
    settings.redis_url,
    decode_responses=True,
)

def get_redis() -> aioredis.Redis:
    """获取 Redis 客户端实例 (依赖注入用)"""
    return aioredis.Redis(connection_pool=redis_pool)
