package com.inovationbehavior.backend.ai.rag.graphrag;

import com.inovationbehavior.backend.ai.rag.document.RagDocumentCorpus;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.document.Document;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

/**
 * GraphRAG 实体索引构建：从 RAG 语料抽取实体并写入 EntityDocumentIndex。
 */
@Configuration
@Slf4j
public class GraphRAGIndexBuilder {

    @Bean
    @ConditionalOnProperty(name = "app.rag.graphrag.enabled", havingValue = "true")
    public EntityDocumentIndex entityDocumentIndex(RagDocumentCorpus corpus,
                                                  EntityExtractor entityExtractor,
                                                  @Value("${app.rag.graphrag.index-max-docs:300}") int indexMaxDocs) {
        InMemoryEntityDocumentIndex index = new InMemoryEntityDocumentIndex();
        List<Document> docs = corpus.getDocuments();
        if (docs == null || docs.isEmpty()) {
            log.info("[GraphRAG] Corpus empty, entity index empty.");
            return index;
        }
        int limit = Math.min(docs.size(), Math.max(1, indexMaxDocs));
        List<Document> toIndex = limit >= docs.size() ? docs : docs.subList(0, limit);
        index.buildFromDocuments(toIndex, entityExtractor);
        log.info("[GraphRAG] Entity index built from {} documents.", toIndex.size());
        return index;
    }
}
