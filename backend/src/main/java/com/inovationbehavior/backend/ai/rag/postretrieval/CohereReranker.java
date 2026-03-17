package com.inovationbehavior.backend.ai.rag.postretrieval;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 检索后：使用 Cohere Rerank API 对候选文档精排（Cross-Encoder 风格，企业级常用）。
 * 需在 HybridRagConfig 中配置 app.rag.cohere.api-key 与 app.rag.cohere.enabled=true 时选用。
 */
@Slf4j
public class CohereReranker implements DocumentReranker {

    private static final String RERANK_URL = "https://api.cohere.ai/v1/rerank";

    private final String apiKey;
    private final String model;
    private final int topK;
    private final RestTemplate restTemplate;

    public CohereReranker(
            @org.springframework.beans.factory.annotation.Value("${app.rag.cohere.api-key:}") String apiKey,
            @org.springframework.beans.factory.annotation.Value("${app.rag.cohere.model:rerank-multilingual-v3.0}") String model,
            @org.springframework.beans.factory.annotation.Value("${app.rag.hybrid.final-top-k:6}") int topK,
            RestTemplate restTemplate) {
        this.apiKey = apiKey != null ? apiKey.trim() : "";
        this.model = model != null && !model.isBlank() ? model : "rerank-multilingual-v3.0";
        this.topK = Math.max(1, Math.min(topK, 100));
        this.restTemplate = restTemplate != null ? restTemplate : new RestTemplate();
    }

    @Override
    public List<Document> rerank(Query query, List<Document> documents) {
        if (documents == null || documents.isEmpty()) return List.of();

        String queryText = query != null ? query.text() : null;
        if (queryText == null || queryText.isBlank()) return documents.stream().limit(topK).toList();

        if (apiKey.isBlank()) {
            log.warn("Cohere api-key not set, returning documents unchanged");
            return documents.stream().limit(topK).toList();
        }

        List<String> texts = new ArrayList<>(documents.size());
        for (Document doc : documents) {
            String content = doc.getText();
            texts.add(content != null ? content : "");
        }

        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.setBearerAuth(apiKey);

            Map<String, Object> body = Map.of(
                    "model", model,
                    "query", queryText,
                    "documents", texts,
                    "top_n", Math.min(topK, documents.size())
            );

            HttpEntity<Map<String, Object>> request = new HttpEntity<>(body, headers);
            CohereRerankResponse response = restTemplate.postForObject(RERANK_URL, request, CohereRerankResponse.class);

            if (response == null || response.results == null || response.results.isEmpty()) {
                return documents.stream().limit(topK).toList();
            }

            List<Document> out = new ArrayList<>(response.results.size());
            for (CohereResult r : response.results) {
                int idx = r.index;
                if (idx >= 0 && idx < documents.size()) {
                    out.add(documents.get(idx));
                }
            }
            return out;
        } catch (Exception e) {
            log.warn("Cohere rerank failed, falling back to original order: {}", e.getMessage());
            return documents.stream().limit(topK).toList();
        }
    }

    private static class CohereRerankResponse {
        @JsonProperty("results")
        List<CohereResult> results;
    }

    private static class CohereResult {
        @JsonProperty("index")
        int index;
        @JsonProperty("relevance_score")
        double relevanceScore;
    }
}
