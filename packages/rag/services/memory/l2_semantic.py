"""L2 语义记忆：跨会话事实/偏好，按需召回

依赖 Redis Stack (FT.SEARCH)。不可用时全部降级为 no-op。
"""
import json, logging, struct, uuid, time
from typing import List, Dict, Any, Optional, Callable, Awaitable

from packages.common.redis_client import get_redis
from .config import (
    L2_ENABLE, L2_KEY_PREFIX, L2_INDEX_PREFIX,
    L2_TOP_K, L2_MIN_SCORE, L2_TTL,
)

logger = logging.getLogger(__name__)
EmbedFn = Callable[[str], Awaitable[List[float]]]

_INITED_INDEX: set = set()


async def add_memory(
    session_id: str, text: str, embed_fn: EmbedFn, meta: Optional[Dict] = None,
) -> Optional[str]:
    if not L2_ENABLE:
        return None
    try:
        vec = await embed_fn(text)
        return await _upsert(session_id, text, vec, meta or {})
    except Exception as e:
        logger.exception("L2 add_memory 失败: %s", e)
        return None


async def recall(
    session_id: str, query: str, embed_fn: EmbedFn, top_k: int = L2_TOP_K,
) -> List[Dict[str, Any]]:
    if not L2_ENABLE:
        return []
    try:
        qvec = await embed_fn(query)
        return await _search(session_id, qvec, top_k)
    except Exception as e:
        logger.warning("L2 recall 失败: %s", e)
        return []


# ---------- Redis Stack 实现 ----------
async def _upsert(session_id: str, text: str, vec: List[float], meta: Dict) -> str:
    redis = get_redis()
    mem_id = uuid.uuid4().hex
    key = f"{L2_KEY_PREFIX}{session_id}:{mem_id}"
    await _ensure_index(session_id)
    vec_bytes = struct.pack(f"<{len(vec)}f", *vec)
    await redis.hset(key, mapping={
        "content": text,
        "meta": json.dumps(meta, ensure_ascii=False),
        "ts": time.time(),
        "embedding": vec_bytes,
    })
    await redis.expire(key, L2_TTL)
    return mem_id


async def _search(session_id: str, qvec: List[float], top_k: int) -> List[Dict]:
    from redis.commands.search.query import Query
    redis = get_redis()
    idx = f"{L2_INDEX_PREFIX}{session_id}"
    qbytes = struct.pack(f"<{len(qvec)}f", *qvec)
    q = (
        Query(f"*=>[KNN {top_k} @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("content", "meta", "score")
        .paging(0, top_k)
        .dialect(2)
    )
    res = await redis.ft(idx).search(q, query_params={"vec": qbytes})
    out = []
    for doc in res.docs:
        # KNN 返回的是距离；COSINE 下转为相似度
        score = 1.0 - float(getattr(doc, "score", 1.0))
        if score < L2_MIN_SCORE:
            continue
        out.append({
            "content": doc.content,
            "score": score,
            "meta": json.loads(getattr(doc, "meta", "{}")),
        })
    return out


async def _ensure_index(session_id: str):
    if session_id in _INITED_INDEX:
        return
    redis = get_redis()
    idx = f"{L2_INDEX_PREFIX}{session_id}"
    try:
        await redis.ft(idx).info()
    except Exception as e:
        # 只有"索引不存在"才去创建；连接/权限等真异常直接返回，不误创建
        msg = str(e).lower()
        if not any(m in msg for m in ("unknown index", "no such index", "does not exist")):
            logger.warning("[%s] FT.INFO 异常（非不存在），跳过: %s", session_id, e)
            return
        dim = 1024  # 需与 embed_fn 输出维度一致
        try:
            await redis.execute_command(
                "FT.CREATE", idx, "ON", "HASH",
                "PREFIX", 1, f"{L2_KEY_PREFIX}{session_id}:",
                "SCHEMA",
                "content", "TEXT",
                "meta", "TEXT",
                "ts", "NUMERIC",
                "embedding", "VECTOR", "HNSW", 6,
                "TYPE", "FLOAT32", "DIM", str(dim),
                "DISTANCE_METRIC", "COSINE",
            )
        except Exception as e:
            # 多 worker 并发：别人先建好了 → 当成功，别刷假告警
            if "already exists" in str(e).lower():
                _INITED_INDEX.add(session_id)
                return
            # 真失败：不加 set，下次重试
            logger.warning("创建 L2 索引失败（可能不是 Redis Stack）: %s", e)
            return
    _INITED_INDEX.add(session_id)


async def delete_memory(session_id: str, mem_id: str) -> bool:
    if not L2_ENABLE:
        return False
    redis = get_redis()
    key = f"{L2_KEY_PREFIX}{session_id}:{mem_id}"
    return await redis.delete(key) > 0


async def delete_by_turn(session_id: str, turn_id: str) -> int:
    """删除 meta.turn_id 匹配的 L2 记忆（需要 add 时把 turn_id 写进 meta）"""
    if not L2_ENABLE:
        return 0
    redis = get_redis()
    pattern = f"{L2_KEY_PREFIX}{session_id}:*"
    deleted = 0
    cursor = 0
    while True:
        cursor, keys = await redis.scan(cursor, match=pattern, count=100)
        for k in keys:
            meta_raw = await redis.hget(k, "meta")
            if not meta_raw:
                continue
            try:
                meta = json.loads(meta_raw)
            except Exception:
                continue
            if meta.get("turn_id") == turn_id:
                deleted += await redis.delete(k)
        if cursor == 0:
            break
    return deleted


async def clear_session(session_id: str) -> int:
    if not L2_ENABLE:
        return 0
    redis = get_redis()
    pattern = f"{L2_KEY_PREFIX}{session_id}:*"
    deleted = 0
    cursor = 0
    while True:
        cursor, keys = await redis.scan(cursor, match=pattern, count=100)
        if keys:
            deleted += await redis.delete(*keys)
        if cursor == 0:
            break
    # 顺手删索引
    try:
        await redis.execute_command("FT.DROPINDEX", f"{L2_INDEX_PREFIX}{session_id}", "DD")
    except Exception:
        logger.warning(
            "drop L2 index failed, session_id=%s index=%s",
            session_id, f"{L2_INDEX_PREFIX}{session_id}",
            exc_info=True,
        )
    return deleted


def render_for_context(items: List[Dict[str, Any]]) -> str:
    if not items:
        return ""
    return "\n".join(f"- {it['content']}" for it in items)