package com.inovationbehavior.backend.ai.rag;

import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Import;
import org.springframework.jdbc.core.JdbcTemplate;

import com.inovationbehavior.backend.config.PgVectorDataSourceConfig;

import java.util.*;
import java.util.stream.Collectors;

import static org.springframework.ai.vectorstore.pgvector.PgVectorStore.PgDistanceType.COSINE_DISTANCE;
import static org.springframework.ai.vectorstore.pgvector.PgVectorStore.PgIndexType.HNSW;

/**
 * PgVector 向量存储（PostgreSQL）：与 BM25 共用同一份语料，向量写入 pgvector，关键词倒排在内存。
 * 启动时增量同步：已向量化的文档（按 chunk_key 识别）跳过，仅处理新增/变更的文档；已删除的源文件对应向量会清理。
 * 必须配置 spring.ai.vectorstore.pgvector.url，否则应用启动失败。
 */
@Configuration
@Slf4j
@Order(Ordered.HIGHEST_PRECEDENCE)
@Import(PgVectorDataSourceConfig.class)  // 确保先创建 pgvectorJdbcTemplate
@ConditionalOnProperty(name = "spring.ai.vectorstore.pgvector.url")  // 与 PgVectorDataSourceConfig 同条件，避免 @ConditionalOnBean 时序问题
public class PgVectorVectorStoreConfig {

    private static final String VECTOR_TABLE = "vector_store";
    private static final String SCHEMA = "public";

    @Resource
    private RagDocumentCorpus ragDocumentCorpus;

    @Value("${spring.ai.vectorstore.pgvector.dimensions:1536}")
    private int dimensions;

    @Bean("IBVectorStore")
    public VectorStore IBVectorStore(
            @Qualifier("pgvectorJdbcTemplate") JdbcTemplate pgvectorJdbcTemplate,
            EmbeddingModel embeddingModel) {
        VectorStore vectorStore = PgVectorStore.builder(pgvectorJdbcTemplate, embeddingModel)
                .dimensions(dimensions)
                .distanceType(COSINE_DISTANCE)
                .indexType(HNSW)
                .initializeSchema(true)
                .schemaName(SCHEMA)
                .vectorTableName(VECTOR_TABLE)
                .maxDocumentBatchSize(10000)
                .build();
        List<Document> documents = ragDocumentCorpus.getDocuments();
        if (documents.isEmpty()) {
            log.info("RAG corpus is empty, pgvector store not seeded.");
            return vectorStore;
        }
        try {
            incrementalSync(pgvectorJdbcTemplate, vectorStore, documents);
        } catch (Exception e) {
            log.warn("Failed to sync pgvector (e.g. OpenAI quota 429). App will start; BM25 still works. Error: {}", e.getMessage());
        }
        return vectorStore;
    }

    /**
     * 增量同步：已存在（chunk_key 一致）的跳过，新增/变更的文档按源文件替换写入，已删除的源对应向量清理。
     */
    private void incrementalSync(JdbcTemplate jdbc, VectorStore vectorStore, List<Document> documents) {
        String qualifiedTable = SCHEMA + "." + VECTOR_TABLE;
        Map<String, Set<String>> existingBySource = loadExistingChunkKeysBySource(jdbc, qualifiedTable);

        // 按 source 分组当前文档
        Map<String, List<Document>> bySource = documents.stream()
                .filter(d -> d.getMetadata() != null && d.getMetadata().get("source") != null)
                .collect(Collectors.groupingBy(d -> String.valueOf(d.getMetadata().get("source"))));

        Set<String> currentSources = new HashSet<>(bySource.keySet());

        // 1. 删除已不存在的源文件对应向量
        for (String source : new HashSet<>(existingBySource.keySet())) {
            if (!currentSources.contains(source)) {
                deleteBySource(jdbc, qualifiedTable, source);
                log.info("Removed vectors for deleted source: {}", truncateSource(source));
            }
        }

        // 2. 按源同步：未变更的跳过，变更的删除旧数据后写入新数据
        int added = 0;   // 新增写入（新源文件或内容变更）
        int skipped = 0; // 已有且未变更，跳过
        for (Map.Entry<String, List<Document>> e : bySource.entrySet()) {
            String source = e.getKey();
            List<Document> chunks = e.getValue();
            Set<String> currentKeys = chunks.stream()
                    .map(d -> (String) d.getMetadata().get("chunk_key"))
                    .filter(Objects::nonNull)
                    .collect(Collectors.toSet());
            Set<String> existingKeys = existingBySource.getOrDefault(source, Set.of());

            if (currentKeys.equals(existingKeys)) {
                skipped += chunks.size();
                continue;
            }

            deleteBySource(jdbc, qualifiedTable, source);
            vectorStore.add(chunks);
            added += chunks.size();
        }

        int total = documents.size();
        log.info("PgVector 增量同步完成: 共 {} chunks，其中 新增 {}（已写入向量库），已有 {}（未变更已跳过）",
                total, added, skipped);
    }

    private Map<String, Set<String>> loadExistingChunkKeysBySource(JdbcTemplate jdbc, String qualifiedTable) {
        try {
            String sql = "SELECT metadata->>'source' as s, metadata->>'chunk_key' as k FROM " + qualifiedTable +
                    " WHERE metadata->>'source' IS NOT NULL AND metadata->>'chunk_key' IS NOT NULL";
            List<Map<String, Object>> rows = jdbc.queryForList(sql);
            Map<String, Set<String>> bySource = new HashMap<>();
            for (Map<String, Object> row : rows) {
                String s = (String) row.get("s");
                String k = (String) row.get("k");
                if (s != null && k != null) {
                    bySource.computeIfAbsent(s, x -> new HashSet<>()).add(k);
                }
            }
            return bySource;
        } catch (Exception e) {
            return Map.of();
        }
    }

    private void deleteBySource(JdbcTemplate jdbc, String qualifiedTable, String source) {
        jdbc.update("DELETE FROM " + qualifiedTable + " WHERE metadata->>'source' = ?", source);
    }

    private String truncateSource(String source) {
        if (source == null || source.length() <= 60) return source;
        return source.substring(0, 57) + "...";
    }
}
