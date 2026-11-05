import uuid
import logging
from typing import Optional
from sqlalchemy import delete, update, select
from packages.common.redis_client import get_redis
from packages.rag.db.database import async_session_maker
from packages.rag.db.models import Conversation, Message, SessionCheckpoint
from packages.rag.services.memory import l0_working, l1_episodic, l3_evidence
from packages.rag.services.memory.context_builder import get_structured_context

logger = logging.getLogger(__name__)
STREAM_KEY = "chat:events:stream"


async def save_message_publish(session_id: str, role: str, content: str, turn_id: Optional[str] = None) -> str:
    """发消息核心：只写 Redis，并异步扔进 Stream 落库"""
    redis = get_redis()
    seq = await redis.incr(f"chat:seq:{session_id}")
    token_count = len(content) // 4  # 简单估算

    if role == "user":
        turn_id = await l0_working.start_turn(session_id, content, seq, turn_id)
    else:
        await l0_working.append_assistant(session_id, content, seq, turn_id)

    await redis.xadd(STREAM_KEY, {
        "event_id": str(uuid.uuid4()), "conversation_id": session_id,
        "seq": seq, "role": role, "content": content, "token_count": token_count,
    })
    return turn_id


async def build_context(session_id: str, query: str, llm, embed_fn=None):
    """构建上下文（包含 DB 冷启动回源）"""
    await l0_working.rebuild_from_db(session_id, async_session_maker, limit=10)

    # 复用你的四层记忆 context_builder
    return await get_structured_context(
        session_id, llm=llm, query=query, embed_fn=embed_fn
    )


from sqlalchemy import select, update


# ================= 删除功能 =================
async def delete_single_message(session_id: str, seq: int, cascade: bool = True) -> int:
    """
    删除一条消息：
      1. 删 Redis L0 热数据（内部自带 turn_id 级联）；
      2. 软删 DB 中同轮次的消息（基于 seq 顺藤摸瓜，不需加字段）。
    返回实际软删的 DB 行数。
    """
    db_deleted = 0
    role = "unknown"

    # 1. 先查目标消息（DB 是权威源）
    async with async_session_maker() as session:
        async with session.begin():
            stmt = select(Message).where(
                Message.conversation_id == session_id,
                Message.seq == seq,
                Message.is_deleted == False,
            )
            msg = (await session.execute(stmt)).scalar_one_or_none()

            if msg is None:
                # DB 中已经不存在，仍然尝试清 Redis，不做级联猜测
                await l0_working.delete_single_msg(session_id, seq, cascade=False)
                return 0

            role = msg.role
            ids_to_delete = [msg.id]

            # 2. 如果删的是 user 且开启级联：在 DB 里往后找紧跟的 assistant
            if cascade and role == "user":
                stmt_assistant = select(Message).where(
                    Message.conversation_id == session_id,
                    Message.seq > seq,  # 找比当前 user seq 大的
                    Message.role == "assistant",
                    Message.is_deleted == False
                ).order_by(Message.seq.asc()).limit(1)  # 取最近的一条 assistant

                assistant_msg = (await session.execute(stmt_assistant)).scalar_one_or_none()
                if assistant_msg:
                    ids_to_delete.append(assistant_msg.id)

            # 3. 批量软删 DB
            result = await session.execute(
                update(Message)
                .where(Message.id.in_(ids_to_delete))
                .values(is_deleted=True)
            )
            db_deleted = result.rowcount or 0

    # 4. DB 提交成功后，再删 Redis
    try:
        # 这里的 l0_working.delete_single_msg 会利用 Redis JSON 里的 turn_id 自动级联
        redis_deleted = await l0_working.delete_single_msg(
            session_id, seq, cascade=(cascade and role == "user")
        )
    except Exception:
        logger.exception(
            "DB已软删，但 Redis 清理失败: session=%s seq=%s", session_id, seq
        )
        redis_deleted = 0

    # 5. 告警排查口径漂移
    if redis_deleted and db_deleted and redis_deleted != db_deleted:
        logger.warning(
            "delete count mismatch: session=%s seq=%s redis=%s db=%s",
            session_id, seq, redis_deleted, db_deleted,
        )

    return db_deleted


async def delete_entire_conversation(session_id: str):
    """整个会话清除：Redis(L0/L1/L3) 全删，DB 物理全删。保留 L2"""
    logger.info(f"[{session_id}] 执行彻底删除...")

    # 1. 清 Redis 四层 (保留 L2)
    await l0_working.clear_session(session_id)
    await l1_episodic.clear_session(session_id)
    l3_evidence.delete_session(session_id)

    # 2. 清 DB
    async with async_session_maker() as session:
        async with session.begin():
            await session.execute(delete(Message).where(Message.conversation_id == session_id))
            await session.execute(delete(SessionCheckpoint).where(SessionCheckpoint.conversation_id == session_id))
            await session.execute(delete(Conversation).where(Conversation.id == session_id))

    await get_redis().delete(f"chat:seq:{session_id}")



