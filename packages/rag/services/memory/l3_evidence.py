"""L3 证据层：完整工具输出/网页/文件卸载到文件系统

上下文中只放 ref 路径，模型需要时按路径读取。
"""
import uuid, logging, shutil, time
from pathlib import Path
from typing import Optional

from .config import EVIDENCE_ROOT

logger = logging.getLogger(__name__)


def _ref_path(session_id: str, ref_id: str) -> Path:
    return Path(EVIDENCE_ROOT) / session_id / f"{ref_id}.md"


def save_evidence(session_id: str, content: str, ref_id: Optional[str] = None) -> str:
    """写入证据文件，返回相对 ref 路径"""
    ref_id = ref_id or uuid.uuid4().hex
    path = _ref_path(session_id, ref_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"refs/{session_id}/{ref_id}.md"


def read_evidence(ref: str) -> str:
    """按 ref 路径读回原文（模型下钻时调用）"""
    parts = ref.split("/")
    if len(parts) < 3 or parts[0] != "refs":
        raise ValueError(f"非法 ref: {ref}")
    path = Path(EVIDENCE_ROOT) / parts[1] / parts[2]
    return path.read_text(encoding="utf-8")


def delete_evidence(ref: str) -> bool:
    """删除单条证据。返回是否真的删了（文件不存在返回 False）"""
    parts = ref.split("/")
    if len(parts) < 3 or parts[0] != "refs":
        logger.warning("拒绝删除非法 ref: %s", ref)
        return False
    path = Path(EVIDENCE_ROOT) / parts[1] / parts[2]
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as e:
        logger.error("删除证据失败 %s: %s", ref, e)
        return False


def delete_session(session_id: str) -> int:
    """删除整个 session 的证据目录。返回删除的文件数"""
    d = Path(EVIDENCE_ROOT) / session_id
    if not d.exists():
        return 0
    count = sum(1 for _ in d.glob("*.md"))
    try:
        shutil.rmtree(d)
    except OSError as e:
        logger.error("删除 session 证据目录失败 %s: %s", session_id, e)
        return 0
    logger.info("[%s] 已删除证据文件 %d 个", session_id, count)
    return count


def purge_older_than(days: int) -> int:
    """清理 N 天前的证据文件（供定时任务调用）。返回删除文件数"""
    cutoff = time.time() - days * 86400
    root = Path(EVIDENCE_ROOT)
    if not root.exists():
        return 0
    removed = 0
    for f in root.rglob("*.md"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError as e:
            logger.warning("清理失败 %s: %s", f, e)
    # 顺带清理空目录（先清 .tmp / 系统垃圾残留，再删空目录）
    _GARBAGE = {".DS_Store", "Thumbs.db"}
    _GARBAGE_SUFFIX = {".tmp", ".swp"}
    for d in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if not d.is_dir():
            continue
        for child in list(d.iterdir()):
            if child.is_file() and (child.name in _GARBAGE or child.suffix in _GARBAGE_SUFFIX):
                try:
                    child.unlink()
                except OSError as e:
                    logger.warning("清理垃圾文件失败 %s: %s", child, e)
        if not any(d.iterdir()):
            try:
                d.rmdir()
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.warning("删除空目录失败 %s: %s", d, e)   # ENOTEMPTY/权限等，留痕
    logger.info("purge_older_than(%d天) 删除 %d 个文件", days, removed)
    return removed

