package com.inovationbehavior.backend.ai.memory.nli;

import org.springframework.ai.document.Document;

import java.util.List;

/**
 * NLI 冲突检测：判断新记忆与已有记忆是否矛盾，用于长期记忆写入控制。
 */
public interface NliConflictDetector {

    /**
     * 检测新记忆是否与已有记忆集合存在事实/主张冲突。
     *
     * @param newMemoryContent 待写入的新记忆文本
     * @param existingMemories 已存在的相关记忆（通常为相似度检索得到的 top-k）
     * @return true 表示存在冲突，建议不写入或先化解后再写
     */
    boolean hasConflict(String newMemoryContent, List<Document> existingMemories);
}
