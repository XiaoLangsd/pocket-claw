"""水位触发 → 卸载到 L3 → LLM 摘要写 L1 → 从 L0 移除

触发条件：L0 token > BUDGET_HISTORY * WATERLINE
保留窗口：最近 KEEP_RECENT_TURNS 轮不压缩
只处理 completed 的 turn
"""
import json, re, logging
from typing import List, Dict, Any, Optional

from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseLanguageModel

from . import l0_working as l0
from . import l1_episodic as l1
from . import l3_evidence as l3
from .config import WATERLINE, BUDGET_HISTORY, KEEP_RECENT_TURNS

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """你是一个会话记忆压缩器。请把下面这轮对话压缩成一条结构化任务记录。
只输出 JSON，不要输出其他内容。字段：
- goal: 这轮对话要达成的目标（一句话）
- summary: 关键信息（2-4 句，含结论、错误、下一步）
- tags: 关键词数组（最多 5 个）

对话：
{conversation}
"""


async def maybe_compress(
    session_id: str,
    llm: Optional[BaseLanguageModel] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    检查水位并执行压缩。
    返回 {triggered, archived_turns, tokens_before, tokens_after}
    """
    all_msgs = await l0.load_turns(session_id)
    tokens = l0.estimate_tokens(all_msgs)
    triggered = force or (tokens > BUDGET_HISTORY * WATERLINE)

    if not triggered:
        return {"triggered": False, "tokens_before": tokens,
                "tokens_after": tokens, "archived_turns": 0}

    if llm is None:
        logger.warning("[%s] 水位超了但未提供 LLM，跳过压缩", session_id)
        return {"triggered": False, "tokens_before": tokens,
                "tokens_after": tokens, "archived_turns": 0}

    status = await l0.get_turn_status(session_id)
    grouped = l0.group_by_turn(all_msgs)
    keep_ids = {tid for tid, _ in grouped[-KEEP_RECENT_TURNS:]}

    to_archive = [
        (tid, msgs) for tid, msgs in grouped
        if status.get(tid) == "completed" and tid not in keep_ids
    ]
    if not to_archive:
        return {"triggered": True, "tokens_before": tokens,
                "tokens_after": tokens, "archived_turns": 0}

    archived = 0
    for tid, msgs in to_archive:
        try:
            await _archive_one_turn(session_id, tid, msgs, llm)
            archived += 1
        except Exception as e:
            logger.exception("[%s] 压缩 turn %s 失败: %s", session_id, tid, e)

    after = await l0.load_turns(session_id)
    return {
        "triggered": True,
        "tokens_before": tokens,
        "tokens_after": l0.estimate_tokens(after),
        "archived_turns": archived,
    }


async def _archive_one_turn(
    session_id: str, turn_id: str, msgs: List[Dict], llm: BaseLanguageModel,
):
    # 1) 原文卸载到 L3
    raw_text = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in msgs)
    ref = l3.save_evidence(session_id, raw_text, ref_id=f"turn-{turn_id[:8]}")

    # 2) LLM 结构化摘要
    conv = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
    prompt = _SUMMARY_PROMPT.format(conversation=conv)
    resp = await llm.ainvoke([HumanMessage(content=prompt)])
    text = resp.content if hasattr(resp, "content") else str(resp)
    summary = _parse_summary(text)

    # 3) 写 L1
    await l1.save_task_record(session_id, {
        "turn_id": turn_id,
        "goal": summary.get("goal", ""),
        "summary": summary.get("summary", ""),
        "tags": summary.get("tags", []),
        "ref": ref,
        "status": "archived",
    })

    # 4) 从 L0 移除原始轮次
    await l0.remove_turns(session_id, [turn_id])
    logger.info("[%s] 压缩 turn %s → %s", session_id, turn_id, ref)


def _parse_summary(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"summary": text.strip()[:500], "goal": "", "tags": []}