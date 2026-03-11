package com.inovationbehavior.backend.ai.rag.document;

/**
 * 基于字符的 token 估算：中文字符约 1.5 字/token，英文/数字约 4 字符/token。
 * 用于语义切分时的 token 回退控制，无需依赖具体 embedding 模型 API。
 */
public class SimpleTokenCounter implements TokenCounter {

    @Override
    public int countTokens(String text) {
        if (text == null || text.isEmpty()) {
            return 0;
        }
        int tokens = 0;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (isCjk(c)) {
                tokens += 2; // 约 1.5，用 2 偏保守，便于留余量
            } else {
                tokens += 1; // 英文/数字约 4 字符 1 token，这里简化按 1 字符 1 单位，最后除 4 更贴近
            }
        }
        // 折中： (CJK*2 + 其他) / 2 约等于 中文 1.5 字/token、英文 4 字/token 的混合
        return (tokens + 1) / 2;
    }

    private static boolean isCjk(char c) {
        return c >= 0x4E00 && c <= 0x9FFF   // CJK 统一汉字
                || c >= 0x3400 && c <= 0x4DBF
                || c >= 0x3000 && c <= 0x303F; // 标点等
    }
}
