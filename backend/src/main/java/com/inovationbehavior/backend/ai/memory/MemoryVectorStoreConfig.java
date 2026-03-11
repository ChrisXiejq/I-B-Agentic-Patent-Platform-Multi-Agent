package com.inovationbehavior.backend.ai.memory;

import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Import;
import org.springframework.jdbc.core.JdbcTemplate;

import com.inovationbehavior.backend.config.PgVectorDataSourceConfig;

import static org.springframework.ai.vectorstore.pgvector.PgVectorStore.PgDistanceType.COSINE_DISTANCE;
import static org.springframework.ai.vectorstore.pgvector.PgVectorStore.PgIndexType.HNSW;

/**
 * Agent 多级记忆 - L2 情节记忆（Episodic Memory）向量存储。
 * 使用 PgVector 独立表 agent_memory 存储对话历史，支持语义检索。
 * 与 RAG 文档向量库（vector_store）分离，互不影响。
 */
@Configuration
@Import(PgVectorDataSourceConfig.class)
@ConditionalOnProperty(name = "spring.ai.vectorstore.pgvector.url")
public class MemoryVectorStoreConfig {

    private static final String MEMORY_TABLE = "agent_memory";
    private static final String SCHEMA = "public";

    @Value("${spring.ai.vectorstore.pgvector.dimensions:1536}")
    private int dimensions;

    @Bean("memoryVectorStore")
    public VectorStore memoryVectorStore(
            @Qualifier("pgvectorJdbcTemplate") JdbcTemplate pgvectorJdbcTemplate,
            EmbeddingModel embeddingModel) {
        return PgVectorStore.builder(pgvectorJdbcTemplate, embeddingModel)
                .dimensions(dimensions)
                .distanceType(COSINE_DISTANCE)
                .indexType(HNSW)
                .initializeSchema(true)
                .schemaName(SCHEMA)
                .vectorTableName(MEMORY_TABLE)
                .maxDocumentBatchSize(1000)
                .build();
    }
}
