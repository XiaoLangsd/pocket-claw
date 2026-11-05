"""对外兼容入口：签名保持与原实现一致，内部走四层记忆"""
from packages.rag.services.memory import (
    get_structured_context,
    start_turn,
    append_assistant,
    finish_turn,
    save_evidence,
    read_evidence,
)

__all__ = [
    "get_structured_context",
    "save_user_message",
    "save_assistant_message",
    "save_evidence",
    "read_evidence",
]


async def save_user_message(session_id: str, content: str) -> str:
    """用户消息先落库；返回 turn_id（可选，用于工具场景精确绑定）"""
    return await start_turn(session_id, content)


async def save_assistant_message(session_id: str, content: str):
    """助手回复完成后追加，并标记该轮完成（可被压缩）"""
    turn_id = await append_assistant(session_id, content)
    await finish_turn(session_id, turn_id)

