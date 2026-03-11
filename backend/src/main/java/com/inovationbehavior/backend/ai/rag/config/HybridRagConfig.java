package com.inovationbehavior.backend.ai.rag.config;

import com.inovationbehavior.backend.ai.rag.document.RagDocumentCorpus;
import com.inovationbehavior.backend.ai.rag.postretrieval.EmbeddingReranker;
import com.inovationbehavior.backend.ai.rag.preretrieval.ContextualQueryAugmenterFactory;
import com.inovationbehavior.backend.ai.rag.retrieval.BM25DocumentRetriever;
import com.inovationbehavior.backend.ai.rag.retrieval.HybridDocumentRetriever;
import jakarta.annotation.Resource;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.rag.advisor.RetrievalAugmentationAdvisor;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;
import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * RAG 总配置：多路召回（向量 + BM25）→ RRF 融合 → Rerank，并组装 RetrievalAugmentationAdvisor
 */
@Configuration
public class HybridRagConfig {

    @Resource
    @Qualifier("IBVectorStore")
    private VectorStore vectorStore;

    @Resource
    private RagDocumentCorpus ragDocumentCorpus;

    @Resource
    private EmbeddingModel embeddingModel;

    @Value("${app.rag.hybrid.vector-top-k:8}")
    private int vectorTopK;

    @Value("${app.rag.hybrid.bm25-top-k:8}")
    private int bm25TopK;

    @Value("${app.rag.hybrid.final-top-k:6}")
    private int finalTopK;

    @Bean
    public BM25DocumentRetriever bm25DocumentRetriever() {
        return new BM25DocumentRetriever(ragDocumentCorpus.getDocuments(), bm25TopK);
    }

    @Bean
    public EmbeddingReranker embeddingReranker() {
        return new EmbeddingReranker(embeddingModel, finalTopK);
    }

    @Bean
    public DocumentRetriever hybridDocumentRetriever() {
        VectorStoreDocumentRetriever vectorRetriever = VectorStoreDocumentRetriever.builder()
                .vectorStore(vectorStore)
                .similarityThreshold(0.3)
                .topK(vectorTopK)
                .build();
        return new HybridDocumentRetriever(
                vectorRetriever,
                bm25DocumentRetriever(),
                embeddingReranker(),
                vectorTopK,
                bm25TopK);
    }

    @Bean
    public Advisor hybridRagAdvisor() {
        return RetrievalAugmentationAdvisor.builder()
                .documentRetriever(hybridDocumentRetriever())
                .queryAugmenter(ContextualQueryAugmenterFactory.createInstance())
                .build();
    }
}
