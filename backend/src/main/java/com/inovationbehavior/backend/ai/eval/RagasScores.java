package com.inovationbehavior.backend.ai.eval;

/**
 * RAGAS 风格单条样本得分：Faithfulness、Answer Relevancy、Context Precision、Context Recall。
 * 每项 0~1，-1 表示未计算（如无参考答案时不算 Context Recall）。
 */
public record RagasScores(
        double faithfulness,
        double answerRelevancy,
        double contextPrecision,
        double contextRecall
) {
    public static RagasScores none() {
        return new RagasScores(-1, -1, -1, -1);
    }

    @Override
    public String toString() {
        return String.format(
                "RagasScores( faithfulness=%.3f, answerRelevancy=%.3f, contextPrecision=%.3f, contextRecall=%.3f )",
                faithfulness, answerRelevancy, contextPrecision, contextRecall);
    }
}
