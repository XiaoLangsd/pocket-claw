"""统一上下文构建：L0 工作记忆 + L1 摘要 + L2 召回 + 可选 RAG"""
import logging
from typing import List, Dict, Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.language_models import BaseLanguageModel

from . import l0_working as l0
from . import l1_episodic as l1
from . import l2_semantic as l2
from .compressor import maybe_compress
from .config import (
    BUDGET_RAG, ENABLE_RAG, KEEP_RECENT_TURNS, L2_ENABLE,
    RAG_MIN_SCORE, RAG_MAX_DOCS,
)

logger = logging.getLogger(__name__)


async def get_structured_context(
    session_id: str,
    raw_rag_docs: Optional[List[Any]] = None,
    *,
    llm: Optional[BaseLanguageModel] = None,
    query: Optional[str] = None,
    embed_fn=None,
) -> Dict[str, Any]:
    """
    构建上下文。返回 {history, context, meta}
      history: List[BaseMessage]  直接喂 LLM
      context: str                L1 摘要 + L2 召回 + RAG
    """
    # 1) 水位检查 & 压缩（失败不影响主流程）
    compress_info: Dict[str, Any] = {}
    try:
        compress_info = await maybe_compress(session_id, llm=llm)
    except Exception as e:
        logger.exception("[%s] 压缩异常: %s", session_id, e)

    # 2) L0：最近 K 轮完整对话
    recent_msgs = await l0.load_recent_turns(session_id, KEEP_RECENT_TURNS)
    history = l0.to_messages(recent_msgs)

    # 3) L1：情节摘要
    task_records = await l1.list_task_records(session_id)
    l1_text = l1.render_for_context(task_records)

    # 4) L2：语义召回（可选）
    l2_text = ""
    if L2_ENABLE and query and embed_fn:
        try:
            hits = await l2.recall(session_id, query, embed_fn)
            l2_text = l2.render_for_context(hits)
        except Exception as e:
            logger.warning("[%s] L2 召回失败: %s", session_id, e)

    # 5) RAG（可选补充）
    rag_text = _trim_rag_context(raw_rag_docs, BUDGET_RAG) if ENABLE_RAG else ""

    # 6) 拼装 context 文本
    parts: List[str] = []
    if l1_text:
        parts.append("## 历史任务摘要\n" + l1_text)
    if l2_text:
        parts.append("## 相关记忆\n" + l2_text)
    if rag_text:
        parts.append("## 检索片段\n" + rag_text)
    context = "\n\n".join(parts)

    ctx_tokens = (
        count_tokens_approximately([HumanMessage(content=context)])
        if context else 0
    )
    return {
        "history": history,
        "context": context,
        "meta": {
            "history_tokens": count_tokens_approximately(history) if history else 0,
            "context_tokens": ctx_tokens,
            "l0_turns": len({m.get("turn_id") for m in recent_msgs}),
            "l1_tasks": len(task_records),
            "compress": compress_info,
        },
    }


def _trim_rag_context(docs: Optional[List[Any]], max_tokens: int) -> str:
    if not docs:
        return ""
    picked: List[str] = []
    used = 0
    for doc in docs[: RAG_MAX_DOCS * 2]:
        score = getattr(doc, "metadata", {}).get("score", 1.0)
        if score < RAG_MIN_SCORE:
            continue
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        t = count_tokens_approximately([HumanMessage(content=content)])
        if used + t > max_tokens:
            continue
        source = getattr(doc, "metadata", {}).get("filename", "未知")
        picked.append(f"[{source}]\n{content}")
        used += t
        if len(picked) >= RAG_MAX_DOCS:
            break
    return "\n\n".join(picked)