package com.inovationbehavior.backend.ai.eval;

import java.util.Map;

/**
 * 检索评估指标：Recall@K、Precision@K、MRR 等聚合结果。
 */
public record RetrievalMetrics(
        double recallAt1,
        double recallAt3,
        double recallAt5,
        double recallAt10,
        double precisionAt5,
        double precisionAt10,
        double mrr,
        int totalCases,
        Map<String, Double> recallAtK
) {
    public RetrievalMetrics(double recallAt1, double recallAt3, double recallAt5, double recallAt10,
                            double mrr, int totalCases, Map<String, Double> recallAtK) {
        this(recallAt1, recallAt3, recallAt5, recallAt10,
                recallAtK.getOrDefault("Precision@5", 0.0),
                recallAtK.getOrDefault("Precision@10", 0.0),
                mrr, totalCases, recallAtK);
    }

    @Override
    public String toString() {
        return String.format(
                "RetrievalMetrics( totalCases=%d, Recall@1=%.4f, Recall@3=%.4f, Recall@5=%.4f, Recall@10=%.4f, Precision@5=%.4f, Precision@10=%.4f, MRR=%.4f )",
                totalCases, recallAt1, recallAt3, recallAt5, recallAt10, precisionAt5, precisionAt10, mrr);
    }
}
