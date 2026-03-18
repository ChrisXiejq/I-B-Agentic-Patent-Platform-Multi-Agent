package com.inovationbehavior.backend.ai.rag.crag;

/**
 * CRAG 检索质量三分支（Corrective RAG）：根据检索文档与查询的相关性置信度决定后续行为。
 */
public enum CragDecision {
    /** 高置信度：检索文档充分相关，直接用于生成 */
    CORRECT,
    /** 模糊：文档部分相关，可过滤或降权后使用 */
    AMBIGUOUS,
    /** 低置信度：文档不相关，应弃用本地检索并触发 Web 兜底（空上下文策略） */
    INCORRECT
}
