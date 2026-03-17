package com.inovationbehavior.backend.ai.memory.longterm;

import com.inovationbehavior.backend.ai.memory.nli.NliConflictDetector;
import com.inovationbehavior.backend.ai.memory.extraction.AtomicFactExtractor;
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

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 第三层：Semantic/Long-Term Memory — 参考 MemoryOS。
 * NLI 三段论：Recall → Conflict Check → Atomic Update（冲突则 UPDATE 覆盖旧事实，互补则 MERGE）。
 * 可选 Zettelkasten 原子事实存储。
 */
@Slf4j
@Service
@ConditionalOnBean(name = "memoryVectorStore")
public class LongTermMemoryService {

    @Resource
    @Qualifier("memoryVectorStore")
    private VectorStore memoryVectorStore;

    @Value("${app.memory.long-term.top-k:5}")
    private int topK;

    @Value("${app.memory.long-term.similarity-threshold:0.5}")
    private double similarityThreshold;

    @Value("${app.memory.long-term.importance-threshold:0.25}")
    private double importanceThreshold;

    @Value("${app.memory.long-term.duplicate-similarity-threshold:0.92}")
    private double duplicateSimilarityThreshold;

    @Value("${app.memory.long-term.nli-check-top-k:3}")
    private int nliCheckTopK;

    @Value("${app.memory.long-term.use-atomic-facts:false}")
    private boolean useAtomicFacts;

    private final NliConflictDetector nliConflictDetector;
    private final AtomicFactExtractor atomicFactExtractor;

    public LongTermMemoryService(
            @org.springframework.beans.factory.annotation.Autowired(required = false) NliConflictDetector nliConflictDetector,
            @org.springframework.beans.factory.annotation.Autowired(required = false) AtomicFactExtractor atomicFactExtractor) {
        this.nliConflictDetector = nliConflictDetector != null ? nliConflictDetector : (newContent, existing) -> false;
        this.atomicFactExtractor = atomicFactExtractor;
    }

    /**
     * 语义检索：按当前查询召回该会话相关长期记忆（BGE/向量相似度）。
     */
    public List<Document> retrieve(String query, String conversationId) {
        if (query == null || query.isBlank()) {
            return List.of();
        }
        Filter.Expression filterExpr = new FilterExpressionBuilder()
                .eq("conversation_id", conversationId)
                .build();
        SearchRequest request = SearchRequest.builder()
                .query(query)
                .topK(topK)
                .similarityThreshold(similarityThreshold)
                .filterExpression(filterExpr)
                .build();
        List<Document> docs = memoryVectorStore.similaritySearch(request);
        if (!docs.isEmpty()) {
            log.debug("Long-term memory retrieved {} chunks for conversation {}", docs.size(), conversationId);
        }
        return docs;
    }

    /**
     * 异步写入长期记忆：importance 阈值 + 相似度去重 + NLI 冲突检测通过后才入库。
     *
     * @param conversationId   会话 ID
     * @param userMessage      用户消息
     * @param assistantMessage 助手回复
     * @param importance       本轮重要性得分（由短期层或调用方计算）
     * @param turnIndex        轮次
     */
    @Async
    public void storeIfEligible(String conversationId, String userMessage, String assistantMessage,
                               double importance, int turnIndex) {
        if (userMessage == null || userMessage.isBlank()) return;
        if (importance < importanceThreshold) {
            log.debug("Long-term store skipped: importance {} < threshold {}", importance, importanceThreshold);
            return;
        }

        String content = "User: " + userMessage + "\nAssistant: " + (assistantMessage != null ? assistantMessage : "");
        if (content.length() > 2000) {
            content = content.substring(0, 1997) + "...";
        }

        // 相似度去重：与已有记忆过近则视为重复，不写入
        Filter.Expression filterExpr = new FilterExpressionBuilder()
                .eq("conversation_id", conversationId)
                .build();
        SearchRequest dedupRequest = SearchRequest.builder()
                .query(content)
                .topK(1)
                .similarityThreshold(duplicateSimilarityThreshold)
                .filterExpression(filterExpr)
                .build();
        List<Document> dupCandidates = memoryVectorStore.similaritySearch(dedupRequest);
        if (!dupCandidates.isEmpty()) {
            log.debug("Long-term store skipped: duplicate (similarity >= {})", duplicateSimilarityThreshold);
            return;
        }

        // NLI 三段论：Recall → Conflict Check → Atomic Update / MERGE
        SearchRequest nliRequest = SearchRequest.builder()
                .query(content)
                .topK(nliCheckTopK)
                .similarityThreshold(similarityThreshold)
                .filterExpression(filterExpr)
                .build();
        List<Document> similar = memoryVectorStore.similaritySearch(nliRequest);
        if (nliConflictDetector.hasConflict(content, similar)) {
            // 冲突：UPDATE — 删除旧事实并写入新事实
            String idToDelete = similar.isEmpty() ? null : getDocumentId(similar.get(0));
            if (idToDelete != null) {
                memoryVectorStore.delete(List.of(idToDelete));
                log.debug("Long-term UPDATE: deleted conflicting doc {}", idToDelete);
            }
        }
        // 互补则 MERGE（直接写入）；冲突时已删旧，此处写入新

        if (useAtomicFacts && atomicFactExtractor != null) {
            List<String> facts = atomicFactExtractor.extractFacts(userMessage, assistantMessage);
            if (!facts.isEmpty()) {
                for (String fact : facts) {
                    String toStore = fact.length() > 2000 ? fact.substring(0, 1997) + "..." : fact;
                    Document doc = new Document(toStore, Map.of(
                            "conversation_id", conversationId,
                            "type", "long_term_atomic",
                            "importance", String.valueOf(importance),
                            "turn_index", String.valueOf(turnIndex),
                            "created_at", String.valueOf(System.currentTimeMillis())
                    ));
                    memoryVectorStore.add(List.of(doc));
                }
                log.debug("Long-term stored {} atomic facts for conversation {}", facts.size(), conversationId);
                return;
            }
        }

        Document doc = new Document(content, Map.of(
                "conversation_id", conversationId,
                "type", "long_term",
                "role", "exchange",
                "importance", String.valueOf(importance),
                "turn_index", String.valueOf(turnIndex),
                "created_at", String.valueOf(System.currentTimeMillis())
        ));
        memoryVectorStore.add(List.of(doc));
        log.debug("Long-term memory stored 1 exchange for conversation {} (importance={})", conversationId, importance);
    }

    private static String getDocumentId(Document doc) {
        if (doc.getId() != null && !doc.getId().isBlank()) return doc.getId();
        Object id = doc.getMetadata().get("id");
        return id != null ? id.toString() : null;
    }

    /**
     * 将检索到的长期记忆格式化为可注入 Prompt 的字符串。
     */
    public String formatMemoriesForPrompt(List<Document> memories) {
        if (memories == null || memories.isEmpty()) return "";
        String block = memories.stream()
                .map(Document::getText)
                .filter(c -> c != null && !c.isBlank())
                .map(s -> "  - " + s.replace("\n", " ").trim())
                .collect(Collectors.joining("\n"));
        if (block.isBlank()) return "";
        return "\n[相关历史对话回忆（长期记忆）]\n%s\n".formatted(block);
    }
}
