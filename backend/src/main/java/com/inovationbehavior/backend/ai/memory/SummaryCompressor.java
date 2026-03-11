package com.inovationbehavior.backend.ai.memory;

import java.util.List;

/**
 * 将多轮对话（及可选已有摘要）压缩为一段摘要，用于短期记忆的上下文管理。
 */
public interface SummaryCompressor {

    /**
     * 生成摘要。
     *
     * @param previousSummary 已有运行摘要，可为 null 或空
     * @param turnsToCompress 待压缩的若干轮
     * @return 压缩后的摘要文本
     */
    String compress(String previousSummary, List<MemoryTurnRecord> turnsToCompress);
}
