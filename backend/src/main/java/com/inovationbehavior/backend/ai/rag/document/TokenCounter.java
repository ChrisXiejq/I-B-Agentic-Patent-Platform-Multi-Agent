package com.inovationbehavior.backend.ai.rag.document;

/**
 * 用于估算文本 token 数的接口（便于做 token 回退与截断控制）。
 * 若未接入真实 tokenizer，可用字符数估算（如中文约 1.5 字/token，英文约 4 字符/token）。
 */
public interface TokenCounter {

    /**
     * 估算给定文本的 token 数。
     *
     * @param text 文本，可为 null（返回 0）
     * @return 估算的 token 数
     */
    int countTokens(String text);
}
