"""L1 情节记忆：已完成任务的结构化摘要 + ref 指针

存储：chat:l1:tasks:{sid}  HASH  field=task_id  value=JSON
记录结构：
{
  "task_id": ..., "turn_id": ..., "goal": ...,
  "summary": ..., "tags": [...],
  "ref": "refs/{sid}/{task_id}.md",
  "status": "archived", "updated_time": ...
}
"""
import json, time, uuid, logging
from typing import List, Dict, Any

from packages.common.redis_client import get_redis
from .config import L1_KEY_PREFIX, L1_TTL

logger = logging.getLogger(__name__)
_key = lambda sid: f"{L1_KEY_PREFIX}{sid}"


async def save_task_record(session_id: str, record: Dict[str, Any]) -> str:
    redis = get_redis()
    task_id = record.get("task_id") or uuid.uuid4().hex
    record["task_id"] = task_id
    record.setdefault("updated_time", time.time())
    record.setdefault("status", "archived")

    pipe = redis.pipeline()
    pipe.hset(_key(session_id), task_id, json.dumps(record, ensure_ascii=False))  # noqa
    pipe.expire(_key(session_id), L1_TTL)  # noqa
    await pipe.execute()
    return task_id


async def list_task_records(session_id: str) -> List[Dict[str, Any]]:
    redis = get_redis()
    raw = await redis.hgetall(_key(session_id)) or {}
    out = []
    for v in raw.values():
        try:
            out.append(json.loads(v))
        except Exception as e:
            logger.exception("bad L1 record in %s: %r", session_id, e)
            continue
    out.sort(key=lambda r: r.get("updated_time", 0), reverse=True)
    return out


async def delete_task_record(session_id: str, task_id: str) -> bool:
    redis = get_redis()
    return await redis.hdel(_key(session_id), task_id) > 0


async def delete_by_turn(session_id: str, turn_id: str) -> int:
    """删该 turn 对应的 L1 摘要（若已被压缩）"""
    records = await list_task_records(session_id)
    task_ids = [r["task_id"] for r in records if r.get("turn_id") == turn_id]
    if not task_ids:
        return 0
    redis = get_redis()
    return await redis.hdel(_key(session_id), *task_ids)


async def clear_session(session_id: str) -> int:
    redis = get_redis()
    return await redis.delete(_key(session_id))


def render_for_context(records: List[Dict[str, Any]], limit: int = 10) -> str:
    """渲染为轻量上下文文本（每条约 30-50 token）"""
    if not records:
        return ""
    lines = []
    for r in records[:limit]:
        lines.append(
            f"- [task:{r['task_id'][:8]}] "
            f"goal={r.get('goal','')} | "
            f"status={r.get('status','')} | "
            f"summary={r.get('summary','')} | "
            f"ref={r.get('ref','')}"
        )
    return "\n".join(lines)