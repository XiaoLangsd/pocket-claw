"""L0 工作记忆：最近 N 轮完整对话 + turn 状态

存储结构：
  chat:l0:turns:{sid}    ZSET   score=全局seq  member=JSON消息
  chat:l0:status:{sid}   HASH   field=turn_id  value=active|completed
  chat:l0:turns:current:{sid}  STRING 当前 turn_id
  chat:l0:seq            STRING 全局递增序号

设计：
  - 消息级有序存储 → 可按范围取、可按 turn_id 精确移除
  - turn 级状态 → 压缩时只处理 completed 且非保留窗口的轮次
"""
import json, time, uuid, logging
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.messages.utils import count_tokens_approximately

from packages.common.redis_client import get_redis
from .config import (
    L0_KEY_PREFIX, L0_STATUS_PREFIX, L0_SEQ_KEY, L0_TTL,
    SINGLE_MSG_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

_turns_key  = lambda sid: f"{L0_KEY_PREFIX}{sid}"
_status_key = lambda sid: f"{L0_STATUS_PREFIX}{sid}"
_current_key = lambda sid: f"{L0_KEY_PREFIX}current:{sid}"


async def start_turn(session_id: str, user_content: str) -> str:
    """开始新一轮：写入用户消息，返回 turn_id"""
    redis = get_redis()
    turn_id = uuid.uuid4().hex
    seq = await redis.incr(L0_SEQ_KEY)

    payload = json.dumps({
        "turn_id": turn_id, "role": "user",
        "content": user_content, "ts": time.time(), "seq": seq,
    }, ensure_ascii=False)

    pipe = redis.pipeline()
    pipe.zadd(_turns_key(session_id), {payload: seq})            # noqa
    pipe.hset(_status_key(session_id), turn_id, "active")         # noqa
    pipe.set(_current_key(session_id), turn_id, ex=L0_TTL)        # noqa
    pipe.expire(_turns_key(session_id), L0_TTL)                   # noqa
    pipe.expire(_status_key(session_id), L0_TTL)                  # noqa
    await pipe.execute()
    return turn_id


async def append_assistant(
    session_id: str, content: str, turn_id: Optional[str] = None,
) -> str:
    """追加 assistant 消息到当前（或指定）turn"""
    redis = get_redis()
    if turn_id is None:
        turn_id = await redis.get(_current_key(session_id))
        if not turn_id:
            # 容错：没走 start_turn
            turn_id = uuid.uuid4().hex
            await redis.set(_current_key(session_id), turn_id, ex=L0_TTL)

    seq = await redis.incr(L0_SEQ_KEY)
    payload = json.dumps({
        "turn_id": turn_id, "role": "assistant",
        "content": content, "ts": time.time(), "seq": seq,
    }, ensure_ascii=False)

    pipe = redis.pipeline()
    pipe.zadd(_turns_key(session_id), {payload: seq})  # noqa
    pipe.expire(_turns_key(session_id), L0_TTL)        # noqa
    await pipe.execute()
    return turn_id  # noqa


async def finish_turn(session_id: str, turn_id: str):
    """标记 turn 已完成（可被压缩）"""
    redis = get_redis()
    await redis.hset(_status_key(session_id), turn_id, "completed")


async def load_turns(session_id: str) -> List[Dict[str, Any]]:
    """读取全部 L0 消息（按 seq 升序）"""
    redis = get_redis()
    raw = await redis.zrange(_turns_key(session_id), 0, -1)
    out: List[Dict[str, Any]] = []
    for r in raw:
        try:
            out.append(json.loads(r))
        except Exception as e:
            logger.error("[%s] L0 解析失败: %s", session_id, e)
    return out


async def get_turn_status(session_id: str) -> Dict[str, str]:
    redis = get_redis()
    return await redis.hgetall(_status_key(session_id)) or {} # noqa


async def rebuild_from_db(session_id: str, session_maker, limit: int = 10):
    """如果 Redis 为空（重启/过期），从 DB 加载最近消息到 Redis"""
    redis = get_redis()
    if await redis.exists(_turns_key(session_id)):
        return
    from sqlalchemy import select
    from packages.rag.db.models import Message

    async with session_maker() as session:
        stmt = select(Message).where(
            Message.conversation_id == session_id, Message.is_deleted == False
        ).order_by(Message.seq.desc()).limit(limit * 2)
        result = await session.execute(stmt)
        msgs = result.scalars().all()

        if not msgs:
            return
        pipe = redis.pipeline()
        for msg in reversed(msgs):
            payload = json.dumps({
                "turn_id": f"db_{msg.seq // 2}",  # 简化的 turn_id 生成
                "role": msg.role, "content": msg.content,
                "ts": msg.created_at.timestamp(), "seq": msg.seq
            }, ensure_ascii=False)
            pipe.zadd(_turns_key(session_id), {payload: msg.seq}) # noqa
        pipe.expire(_turns_key(session_id), L0_TTL) # noqa
        await pipe.execute()


async def delete_single_msg(session_id: str, seq: int, cascade: bool = True) -> int:
    """
    删除 L0 中的单条消息。
    由于 ZSet 的 score 就是 seq，直接用 zrangebyscore 精准定位，避免全表扫描。
    """
    redis = get_redis()
    key = _turns_key(session_id)

    # 1. 精准拉取该 seq 对应的消息
    target_raw = await redis.zrangebyscore(key, seq, seq)
    if not target_raw:
        return 0

    target_msg = json.loads(target_raw[0])
    turn_id = target_msg.get("turn_id")
    role = target_msg.get("role")

    to_remove = [target_raw[0]]

    # 2. 级联删除
    if cascade and role == "user":
        # 向后取 20 条，找当前轮的 assistant
        nearby_raw = await redis.zrangebyscore(key, seq, "+inf", start=0, num=20)
        for r in nearby_raw:
            try:
                m = json.loads(r)
            except Exception:
                continue
            if m.get("turn_id") == turn_id and m.get("role") == "assistant":
                to_remove.append(r)
                break

    # 3. 执行 ZREM 删除
    await redis.zrem(key, *to_remove)

    # 4. 【修复点1】仅当该 turn 已经没有任何残留消息时，才清理 status
    if role == "user" and cascade:
        remaining_raw = await redis.zrange(key, 0, -1)
        has_remaining = False
        for r in remaining_raw:
            try:
                if json.loads(r).get("turn_id") == turn_id:
                    has_remaining = True
                    break
            except Exception:
                continue

        if not has_remaining:
            await redis.hdel(_status_key(session_id), turn_id)

    return len(to_remove)


async def remove_turns(session_id: str, turn_ids: List[str]):
    """从 L0 精确移除指定 turn 的全部消息（供压缩器调用）"""
    if not turn_ids:
        return
    redis = get_redis()
    all_msgs = await load_turns(session_id)
    target = set(turn_ids)
    to_remove = [
        json.dumps(m, ensure_ascii=False)
        for m in all_msgs if m.get("turn_id") in target
    ]
    pipe = redis.pipeline()
    if to_remove:
        pipe.zrem(_turns_key(session_id), *to_remove)  # noqa
    for tid in turn_ids:
        pipe.hdel(_status_key(session_id), tid)         # noqa
    await pipe.execute()


async def clear_session(session_id: str) -> None:
    """清空整个会话的 L0"""
    redis = get_redis()
    pipe = redis.pipeline()
    pipe.delete(_turns_key(session_id))       # noqa
    pipe.delete(_status_key(session_id))      # noqa
    pipe.delete(_current_key(session_id))     # noqa
    await pipe.execute()


# ---------- 组装 / 估算 ----------
def group_by_turn(msgs: List[Dict[str, Any]]) -> List[Tuple[str, List[Dict]]]:
    """按 turn_id 聚合，保持出现顺序"""
    turns: List[Tuple[str, List[Dict]]] = []
    index: Dict[str, int] = {}
    for m in msgs:
        tid = m.get("turn_id", "")
        if tid not in index:
            index[tid] = len(turns)
            turns.append((tid, []))
        turns[index[tid]][1].append(m)
    return turns


async def load_recent_turns(session_id: str, n: int) -> List[Dict[str, Any]]:
    """读取最近 n 轮（完整消息）"""
    all_msgs = await load_turns(session_id)
    grouped = group_by_turn(all_msgs)
    recent = grouped[-n:] if n > 0 else grouped
    out: List[Dict[str, Any]] = []
    for _, msgs in recent:
        out.extend(msgs)
    return out


def to_messages(msgs: List[Dict[str, Any]]) -> List[BaseMessage]:
    """dict → LangChain 消息；单条超长时字符级截断（仅喂给 LLM 的副本）"""
    out: List[BaseMessage] = []
    for m in msgs:
        role = m.get("role")
        content = m.get("content", "")
        if count_tokens_approximately([HumanMessage(content=content)]) > SINGLE_MSG_MAX_TOKENS:
            content = content[: SINGLE_MSG_MAX_TOKENS * 4] + "…[已截断]"
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out



def estimate_tokens(msgs: List[Dict[str, Any]]) -> int:
    return sum(
        count_tokens_approximately([HumanMessage(content=m.get("content", ""))])
        for m in msgs
    )