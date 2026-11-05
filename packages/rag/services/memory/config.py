"""四层记忆配置（集中管理，便于调参）"""

# ============ 预算 ============
MAX_TOTAL_TOKENS   = 8000
BUDGET_SYSTEM      = 800
BUDGET_HISTORY     = 5000   # L0 工作记忆预算
BUDGET_RAG         = 800    # RAG 补充预算
BUDGET_OUTPUT      = 1200

# 水位：L0 使用率超过该比例触发压缩
WATERLINE          = 0.7    # 70%
# 压缩时保留最近 K 轮完整对话（其余 completed 轮次转 L1）
KEEP_RECENT_TURNS  = 4
# 单条消息喂给 LLM 时的硬上限（防止单条挤爆窗口，但 L0 原文不丢）
SINGLE_MSG_MAX_TOKENS = 3000

# ============ L0 工作记忆 ============
L0_KEY_PREFIX     = "chat:l0:turns:"
L0_STATUS_PREFIX  = "chat:l0:status:"
L0_SEQ_KEY        = "chat:l0:seq"
L0_TTL            = 24 * 3600

# ============ L1 情节记忆 ============
L1_KEY_PREFIX     = "chat:l1:tasks:"
L1_TTL            = 7 * 24 * 3600

# ============ L2 语义记忆（需 Redis Stack） ============
L2_ENABLE         = False     # 没有向量库时保持 False
L2_KEY_PREFIX     = "chat:l2:mem:"
L2_INDEX_PREFIX   = "idx:chat:l2:"
L2_TOP_K          = 3
L2_MIN_SCORE      = 0.75      # 余弦相似度门限
L2_TTL            = 30 * 24 * 3600

# ============ L3 证据层 ============
EVIDENCE_ROOT     = "data/evidence"

# ============ RAG（可选补充） ============
ENABLE_RAG        = False
RAG_MIN_SCORE     = 0.5
RAG_MAX_DOCS      = 2