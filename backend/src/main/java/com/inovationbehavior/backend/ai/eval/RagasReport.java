package com.inovationbehavior.backend.ai.eval;

import java.util.List;

/**
 * RAGAS 汇总报告：各指标均值与样本数
 */
public record RagasReport(
        int samples,
        double avgFaithfulness,
        double avgAnswerRelevancy,
        double avgContextPrecision,
        double avgContextRecall,
        List<RagasScores> perSampleScores
) {
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== RAGAS Report ===\n");
        sb.append("samples: ").append(samples).append("\n");
        sb.append("faithfulness:       ").append(format(avgFaithfulness)).append("\n");
        sb.append("answer_relevancy:   ").append(format(avgAnswerRelevancy)).append("\n");
        sb.append("context_precision:  ").append(format(avgContextPrecision)).append("\n");
        sb.append("context_recall:     ").append(format(avgContextRecall)).append("\n");
        return sb.toString();
    }

    private static String format(double v) {
        return v < 0 ? "N/A" : String.format("%.4f", v);
    }
}
