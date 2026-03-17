package com.inovationbehavior.backend.ai.memory.experiential;

import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.filter.Filter;
import org.springframework.ai.vectorstore.filter.FilterExpressionBuilder;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 第二层：Experiential Memory（中期）— 参考 AgentScope/Mem0。
 * 递归摘要沉淀为事件摘要，带 decay_weight（检索时加权、未检索衰减）；PostgreSQL + pgvector 存储。
 */
@Slf4j
@Service
@ConditionalOnBean(name = "experientialVectorStore")
public class ExperientialMemoryService {

    @Resource
    @Qualifier("experientialVectorStore")
    private VectorStore experientialVectorStore;

    @Value("${app.memory.experiential.top-k:5}")
    private int topK;

    @Value("${app.memory.experiential.similarity-threshold:0.4}")
    private double similarityThreshold;

    @Value("${app.memory.experiential.decay-weight-default:1.0}")
    private double decayWeightDefault;

    /**
     * 写入一条事件摘要（由 Working Memory 溢出时压缩得到）。
     */
    @Async
    public void addSummary(String conversationId, String summaryText) {
        if (conversationId == null || summaryText == null || summaryText.isBlank()) return;
        String content = summaryText.length() > 2000 ? summaryText.substring(0, 1997) + "..." : summaryText;
        Document doc = new Document(content, Map.of(
                "conversation_id", conversationId,
                "type", "experiential",
                "decay_weight", String.valueOf(decayWeightDefault),
                "created_at", String.valueOf(System.currentTimeMillis())
        ));
        experientialVectorStore.add(List.of(doc));
        log.debug("Experiential memory stored 1 summary for conversation {}", conversationId);
    }

    /**
     * 按语义检索该会话的中期记忆，并按 decay_weight 与时间重排后取 topK。
     */
    public List<Document> retrieve(String query, String conversationId) {
        if (query == null || query.isBlank()) return List.of();
        Filter.Expression filterExpr = new FilterExpressionBuilder()
                .eq("conversation_id", conversationId)
                .build();
        SearchRequest request = SearchRequest.builder()
                .query(query)
                .topK(topK * 2)
                .similarityThreshold(similarityThreshold)
                .filterExpression(filterExpr)
                .build();
        List<Document> docs = experientialVectorStore.similaritySearch(request);
        if (docs.isEmpty()) return List.of();
        // 按 decay_weight（metadata）与 recency 重排，取 topK
        return docs.stream()
                .sorted(Comparator
                        .comparingDouble((Document d) -> getDecayWeight(d))
                        .thenComparingLong((Document d) -> getCreatedAt(d))
                        .reversed())
                .limit(topK)
                .collect(Collectors.toList());
    }

    public String formatMemoriesForPrompt(List<Document> memories) {
        if (memories == null || memories.isEmpty()) return "";
        String block = memories.stream()
                .map(Document::getText)
                .filter(c -> c != null && !c.isBlank())
                .map(s -> "  - " + s.replace("\n", " ").trim())
                .collect(Collectors.joining("\n"));
        if (block.isBlank()) return "";
        return "\n[中期记忆（事件摘要）]\n%s\n".formatted(block);
    }

    private static double getDecayWeight(Document d) {
        Object v = d.getMetadata().get("decay_weight");
        if (v == null) return 1.0;
        try {
            return Double.parseDouble(v.toString());
        } catch (NumberFormatException e) {
            return 1.0;
        }
    }

    private static long getCreatedAt(Document d) {
        Object v = d.getMetadata().get("created_at");
        if (v == null) return 0L;
        try {
            return Long.parseLong(v.toString());
        } catch (NumberFormatException e) {
            return 0L;
        }
    }
}
