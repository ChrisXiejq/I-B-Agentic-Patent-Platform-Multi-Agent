package com.inovationbehavior.backend.ai.eval;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;

import java.util.*;
import java.util.stream.Collectors;

/**
 * RAG 检索评估：Recall@K、MRR。
 * 通过 expectedSourceSubstrings 判断「相关」：若文档 metadata 的 source 包含任一子串则计为命中。
 */
public class RagRetrievalEvaluator {

    private final DocumentRetriever retriever;
    private final int[] kValues;

    public RagRetrievalEvaluator(DocumentRetriever retriever, int[] kValues) {
        this.retriever = retriever;
        this.kValues = kValues != null && kValues.length > 0 ? kValues : new int[]{1, 3, 5, 10};
    }

    public RagRetrievalEvaluator(DocumentRetriever retriever) {
        this(retriever, new int[]{1, 3, 5, 10});
    }

    /**
     * 判断文档是否与期望来源匹配（source 包含任一 expectedSourceSubstring）
     */
    public static boolean isRelevant(Document doc, List<String> expectedSourceSubstrings) {
        if (expectedSourceSubstrings == null || expectedSourceSubstrings.isEmpty()) return false;
        Object src = doc.getMetadata() != null ? doc.getMetadata().get("source") : null;
        String source = src != null ? src.toString() : "";
        return expectedSourceSubstrings.stream().anyMatch(source::contains);
    }

    /**
     * 对一批用例跑检索并汇总 Recall@K、MRR
     */
    public RetrievalMetrics evaluate(List<RagTestCase> cases) {
        if (cases == null || cases.isEmpty()) {
            return new RetrievalMetrics(0, 0, 0, 0, 0, 0, Map.of());
        }

        Arrays.sort(kValues);
        Map<Integer, List<Double>> recallByK = new HashMap<>();
        Map<Integer, List<Double>> precisionByK = new HashMap<>();
        for (int k : kValues) {
            recallByK.put(k, new ArrayList<>());
            precisionByK.put(k, new ArrayList<>());
        }
        List<Double> mrrList = new ArrayList<>();

        for (RagTestCase tc : cases) {
            if (tc.expectedSourceSubstrings().isEmpty()) continue;

            List<Document> retrieved = retriever.retrieve(new Query(tc.query()));
            Set<String> expectedSources = new HashSet<>(tc.expectedSourceSubstrings());

            int firstRelevantRank = -1;

            for (int i = 0; i < retrieved.size(); i++) {
                Document d = retrieved.get(i);
                String source = getSource(d);
                if (source == null) continue;
                boolean relevant = expectedSources.stream().anyMatch(source::contains);
                if (relevant && firstRelevantRank < 0) firstRelevantRank = i + 1;
            }

            for (int k : kValues) {
                int topK = Math.min(k, retrieved.size());
                int relevantCount = 0;  // 在 top-K 中相关文档（chunk）的个数
                boolean anyRelevant = false;
                for (int i = 0; i < topK; i++) {
                    Document d = retrieved.get(i);
                    String source = getSource(d);
                    if (source == null) continue;
                    if (expectedSources.stream().anyMatch(source::contains)) {
                        relevantCount++;
                        anyRelevant = true;
                    }
                }
                // Recall@K：该 query 是否在 top-K 内命中了至少一个相关文档（命中率，0/1）
                double recall = anyRelevant ? 1.0 : 0.0;
                recallByK.get(k).add(recall);
                // Precision@K：top-K 中相关文档占比
                double precision = topK == 0 ? 0 : (double) relevantCount / topK;
                precisionByK.get(k).add(precision);
            }

            if (firstRelevantRank > 0) mrrList.add(1.0 / firstRelevantRank);
            else mrrList.add(0.0);
        }

        Map<String, Double> recallAtK = new LinkedHashMap<>();
        for (int k : kValues) {
            List<Double> list = recallByK.get(k);
            double avg = list.isEmpty() ? 0 : list.stream().mapToDouble(Double::doubleValue).average().orElse(0);
            recallAtK.put("Recall@" + k, avg);
        }
        for (int k : kValues) {
            List<Double> list = precisionByK.get(k);
            double avg = list.isEmpty() ? 0 : list.stream().mapToDouble(Double::doubleValue).average().orElse(0);
            recallAtK.put("Precision@" + k, avg);
        }

        double mrr = mrrList.isEmpty() ? 0 : mrrList.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        int totalCases = cases.stream().filter(tc -> !tc.expectedSourceSubstrings().isEmpty()).toList().size();
        if (totalCases == 0) totalCases = cases.size();

        return new RetrievalMetrics(
                recallAtK.getOrDefault("Recall@1", 0.0),
                recallAtK.getOrDefault("Recall@3", 0.0),
                recallAtK.getOrDefault("Recall@5", 0.0),
                recallAtK.getOrDefault("Recall@10", 0.0),
                recallAtK.getOrDefault("Precision@5", 0.0),
                recallAtK.getOrDefault("Precision@10", 0.0),
                mrr,
                totalCases,
                recallAtK
        );
    }

    private static String getSource(Document d) {
        if (d.getMetadata() == null) return null;
        Object s = d.getMetadata().get("source");
        return s != null ? s.toString() : null;
    }
}
