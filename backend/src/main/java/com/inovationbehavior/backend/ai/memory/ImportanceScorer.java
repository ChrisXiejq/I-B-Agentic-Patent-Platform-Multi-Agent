package com.inovationbehavior.backend.ai.memory;

/**
 * 记忆重要性评分器（领域 NER 加权）。
 * 短期记忆用此分数参与滑动窗口与摘要优先级，长期记忆用此分数做入库阈值筛选。
 */
@FunctionalInterface
public interface ImportanceScorer {

    /**
     * 对单轮对话（用户+助手）计算重要性得分，范围 [0, 1]。
     *
     * @param userMessage    用户消息
     * @param assistantMessage 助手回复，可为 null
     * @param turnIndex      当前轮次（从 0 起），用于后续衰减计算
     * @return 重要性分数，越高越应保留/入库
     */
    double score(String userMessage, String assistantMessage, int turnIndex);
}
