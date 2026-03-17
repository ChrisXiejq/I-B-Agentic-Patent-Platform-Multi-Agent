package com.inovationbehavior.backend.ai.memory.config;

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
 * Agent 多级记忆向量存储。
 * - agent_memory：长期语义记忆（Layer3）
 * - agent_experiential_memory：中期事件摘要（Layer2）
 */
@Configuration
@Import(PgVectorDataSourceConfig.class)
@ConditionalOnProperty(name = "spring.ai.vectorstore.pgvector.url")
public class MemoryVectorStoreConfig {

    private static final String MEMORY_TABLE = "agent_memory";
    private static final String EXPERIENTIAL_TABLE = "agent_experiential_memory";
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

    @Bean("experientialVectorStore")
    public VectorStore experientialVectorStore(
            @Qualifier("pgvectorJdbcTemplate") JdbcTemplate pgvectorJdbcTemplate,
            EmbeddingModel embeddingModel) {
        return PgVectorStore.builder(pgvectorJdbcTemplate, embeddingModel)
                .dimensions(dimensions)
                .distanceType(COSINE_DISTANCE)
                .indexType(HNSW)
                .initializeSchema(true)
                .schemaName(SCHEMA)
                .vectorTableName(EXPERIENTIAL_TABLE)
                .maxDocumentBatchSize(500)
                .build();
    }
}
