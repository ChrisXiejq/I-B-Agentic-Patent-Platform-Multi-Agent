package com.inovationbehavior.backend.ai.memory;

import lombok.Builder;
import lombok.Value;

/**
 * 单轮对话记录，用于短期记忆滑动窗口与摘要压缩。
 */
@Value
@Builder
public class MemoryTurnRecord {

    String userMessage;
    String assistantMessage;
    int turnIndex;
    /** 原始重要性得分（未衰减） */
    double importance;
    long createdAtMillis;

    public String toCompactText() {
        return "User: " + (userMessage != null ? userMessage : "")
                + "\nAssistant: " + (assistantMessage != null ? assistantMessage : "");
    }
}
