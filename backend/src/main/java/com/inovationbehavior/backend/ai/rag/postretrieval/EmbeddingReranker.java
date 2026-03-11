package com.inovationbehavior.backend.ai.rag.postretrieval;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.rag.Query;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 检索后：基于 Embedding 相似度的 Reranker，对融合后的文档按 query-document 相似度重新排序
 */
public class EmbeddingReranker {

    private final EmbeddingModel embeddingModel;
    private final int topK;

    public EmbeddingReranker(EmbeddingModel embeddingModel, int topK) {
        this.embeddingModel = Objects.requireNonNull(embeddingModel);
        this.topK = Math.max(1, topK);
    }

    public List<Document> rerank(Query query, List<Document> documents) {
        if (documents == null || documents.isEmpty()) return List.of();

        String queryText = query.text();
        if (queryText == null || queryText.isBlank()) return documents;

        try {
            float[] queryEmbedding = embeddingModel.embed(queryText);
            List<ScoredDoc> scored = new ArrayList<>(documents.size());

            for (Document doc : documents) {
                String content = doc.getText();
                if (content == null || content.isBlank()) {
                    scored.add(new ScoredDoc(doc, 0.0));
                    continue;
                }
                float[] docEmbedding = embeddingModel.embed(content);
                double sim = cosineSimilarity(queryEmbedding, docEmbedding);
                scored.add(new ScoredDoc(doc, sim));
            }

            return scored.stream()
                    .sorted(Comparator.<ScoredDoc>comparingDouble(s -> s.score).reversed())
                    .limit(topK)
                    .map(s -> s.doc)
                    .collect(Collectors.toList());
        } catch (Exception e) {
            return documents.stream().limit(topK).collect(Collectors.toList());
        }
    }

    private static double cosineSimilarity(float[] a, float[] b) {
        if (a == null || b == null || a.length != b.length) return 0;
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        double denom = Math.sqrt(normA) * Math.sqrt(normB);
        return denom > 0 ? dot / denom : 0;
    }

    private record ScoredDoc(Document doc, double score) {}
}
