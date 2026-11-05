import asyncio
import logging
from sqlalchemy.dialects.postgresql import insert
from packages.common.redis_client import get_redis
from packages.rag.db.database import async_session_maker
from packages.rag.db.models import Message, Conversation

logger = logging.getLogger(__name__)

STREAM_KEY = "chat:events:stream"
GROUP_NAME = "db_drainer_group"
CONSUMER_NAME = "worker_1"


async def start_drainer():
    redis = get_redis()
    try:
        await redis.xgroup_create(STREAM_KEY, GROUP_NAME, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.error(f"创建消费组失败: {e}")

    logger.info("DB Drainer 已启动")
    while True:
        try:
            resp = await redis.xreadgroup(GROUP_NAME, CONSUMER_NAME, {STREAM_KEY: ">"}, count=50, block=2000)
            if not resp:
                continue
            events = resp[0][1]
            if not events:
                continue
            await _process_batch(events)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(f"Drainer 消费异常: {e}")
            await asyncio.sleep(1)


async def _process_batch(events: list):
    redis = get_redis()
    async with async_session_maker() as session:
        async with session.begin():
            ack_ids = []
            for msg_id, data in events:
                event_id = data["event_id"]
                conv_id = data["conversation_id"]
                seq = int(data["seq"])

                # 1. 幂等写入消息
                stmt = insert(Message).values(
                    event_id=event_id,
                    conversation_id=conv_id,
                    seq=seq,
                    role=data["role"],
                    content=data["content"],
                    token_count=int(data["token_count"]),
                ).on_conflict_do_nothing(index_elements=['event_id'])
                await session.execute(stmt)

                # 2. 更新会话状态
                await session.execute(
                    insert(Conversation).values(id=conv_id, last_seq=seq)
                    .on_conflict_do_update(index_elements=['id'], set_={"last_seq": seq})
                )
                ack_ids.append(msg_id)
        # 3. 事务提交后再 XACK
        if ack_ids:
            await redis.xack(STREAM_KEY, GROUP_NAME, *ack_ids)