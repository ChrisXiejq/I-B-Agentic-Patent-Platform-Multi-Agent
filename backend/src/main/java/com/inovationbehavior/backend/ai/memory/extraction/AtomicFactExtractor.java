package com.inovationbehavior.backend.ai.memory.extraction;

import java.util.List;

/**
 * 原子事实抽取（Zettelkasten/卢曼卡片）：从对话中抽取 entity-attribute-value 型知识点，便于长期记忆去重与 NLI。
 */
public interface AtomicFactExtractor {

    /**
     * 从单轮对话中抽取原子事实，每条为 JSON 或结构化字符串，如 {"entity":"Patent A","attribute":"License Fee","value":"5M","source":"..."}。
     */
    List<String> extractFacts(String userMessage, String assistantMessage);
}
