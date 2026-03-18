package com.inovationbehavior.backend.ai.rag.config;

import com.inovationbehavior.backend.ai.rag.crag.CragDocumentRetriever;
import com.inovationbehavior.backend.ai.rag.crag.LlmRetrievalQualityEvaluator;
import com.inovationbehavior.backend.ai.rag.crag.RetrievalQualityEvaluator;
import com.inovationbehavior.backend.ai.rag.document.RagDocumentCorpus;
import com.inovationbehavior.backend.ai.rag.postretrieval.DocumentReranker;
import com.inovationbehavior.backend.ai.rag.postretrieval.EmbeddingReranker;
import com.inovationbehavior.backend.ai.rag.postretrieval.CohereReranker;
import com.inovationbehavior.backend.ai.rag.preretrieval.ContextualQueryAugmenterFactory;
import com.inovationbehavior.backend.ai.rag.graphrag.EntityDocumentIndex;
import com.inovationbehavior.backend.ai.rag.graphrag.EntityExtractor;
import com.inovationbehavior.backend.ai.rag.graphrag.GraphRAGDocumentRetriever;
import com.inovationbehavior.backend.ai.rag.hyde.HyDEDocumentRetriever;
import com.inovationbehavior.backend.ai.rag.preretrieval.LlmMultiQueryExpander;
import com.inovationbehavior.backend.ai.rag.preretrieval.MultiQueryExpander;
import com.inovationbehavior.backend.ai.rag.retrieval.BM25DocumentRetriever;
import com.inovationbehavior.backend.ai.rag.retrieval.HybridDocumentRetriever;
import com.inovationbehavior.backend.ai.rag.retrieval.MultiQueryDocumentRetriever;
import com.inovationbehavior.backend.ai.rag.retrieval.TenantAwareDocumentRetriever;
import com.inovationbehavior.backend.ai.rag.retrieval.TracingDocumentRetriever;
import jakarta.annotation.Resource;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.rag.advisor.RetrievalAugmentationAdvisor;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;
import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;

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

    @Value("${app.rag.cohere.enabled:false}")
    private boolean cohereRerankEnabled;

    @Value("${app.rag.cohere.api-key:}")
    private String cohereApiKey;

    @Value("${app.rag.cohere.model:rerank-multilingual-v3.0}")
    private String cohereModel;

    @Value("${app.rag.crag.enabled:false}")
    private boolean cragEnabled;

    @Value("${app.rag.crag.correct-threshold:0.7}")
    private double cragCorrectThreshold;

    @Value("${app.rag.crag.ambiguous-threshold:0.35}")
    private double cragAmbiguousThreshold;

    @Value("${app.rag.crag.ambiguous-trim-to-half:true}")
    private boolean cragAmbiguousTrimToHalf;

    @Value("${app.rag.multi-query.enabled:false}")
    private boolean multiQueryEnabled;

    @Value("${app.rag.multi-query.max-queries:3}")
    private int multiQueryMaxQueries;

    @Value("${app.rag.tenant.enabled:false}")
    private boolean tenantEnabled;

    @Value("${app.rag.hyde.enabled:false}")
    private boolean hydeEnabled;

    @Value("${app.rag.graphrag.enabled:false}")
    private boolean graphragEnabled;

    @Value("${app.rag.graphrag.entity-top-k:6}")
    private int graphragEntityTopK;

    @Bean
    public BM25DocumentRetriever bm25DocumentRetriever() {
        return new BM25DocumentRetriever(ragDocumentCorpus.getDocuments(), bm25TopK);
    }

    /**
     * Reranker：当 app.rag.cohere.enabled=true 且 api-key 已配置时使用 Cohere Rerank，否则使用 Embedding 相似度精排。
     */
    @Bean
    public DocumentReranker documentReranker() {
        if (cohereRerankEnabled && cohereApiKey != null && !cohereApiKey.isBlank()) {
            return new CohereReranker(cohereApiKey, cohereModel, finalTopK, null);
        }
        return new EmbeddingReranker(embeddingModel, finalTopK);
    }

    @Bean
    public DocumentRetriever hybridDocumentRetriever(ChatModel chatModel) {
        VectorStoreDocumentRetriever rawVectorRetriever = VectorStoreDocumentRetriever.builder()
                .vectorStore(vectorStore)
                .similarityThreshold(0.3)
                .topK(vectorTopK)
                .build();
        DocumentRetriever vectorRetriever = (hydeEnabled ? new HyDEDocumentRetriever(rawVectorRetriever, chatModel) : rawVectorRetriever);
        return new HybridDocumentRetriever(
                vectorRetriever,
                bm25DocumentRetriever(),
                documentReranker(),
                vectorTopK,
                bm25TopK);
    }

    @Bean
    @ConditionalOnProperty(name = "app.rag.crag.enabled", havingValue = "true")
    public RetrievalQualityEvaluator retrievalQualityEvaluator(ChatModel chatModel) {
        return new LlmRetrievalQualityEvaluator(chatModel);
    }

    @Bean
    @ConditionalOnProperty(name = "app.rag.multi-query.enabled", havingValue = "true")
    public MultiQueryExpander multiQueryExpander(ChatModel chatModel) {
        return new LlmMultiQueryExpander(chatModel, multiQueryMaxQueries);
    }

    @Bean
    @ConditionalOnProperty(name = "app.rag.graphrag.enabled", havingValue = "true")
    public EntityExtractor entityExtractor(ChatModel chatModel) {
        return new com.inovationbehavior.backend.ai.rag.graphrag.LlmEntityExtractor(chatModel);
    }

    /** 供 Advisor 使用的检索器：多查询(可选) → GraphRAG(可选) → CRAG(可选) → 租户(可选) → Tracing。 */
    @Bean
    @Primary
    @Qualifier("ragDocumentRetriever")
    public DocumentRetriever ragDocumentRetriever(
            @Qualifier("hybridDocumentRetriever") DocumentRetriever hybridDocumentRetriever,
            @Autowired(required = false) RetrievalQualityEvaluator retrievalQualityEvaluator,
            @Autowired(required = false) MultiQueryExpander multiQueryExpander,
            @Autowired(required = false) EntityDocumentIndex entityDocumentIndex,
            @Autowired(required = false) EntityExtractor entityExtractor) {
        DocumentRetriever inner = hybridDocumentRetriever;
        if (multiQueryEnabled && multiQueryExpander != null) {
            inner = new MultiQueryDocumentRetriever(inner, multiQueryExpander);
        }
        if (graphragEnabled && entityDocumentIndex != null && entityExtractor != null) {
            inner = new GraphRAGDocumentRetriever(inner, entityDocumentIndex, entityExtractor, graphragEntityTopK);
        }
        if (cragEnabled && retrievalQualityEvaluator != null) {
            inner = new CragDocumentRetriever(inner, retrievalQualityEvaluator,
                    cragCorrectThreshold, cragAmbiguousThreshold, cragAmbiguousTrimToHalf);
        }
        if (tenantEnabled) {
            inner = new TenantAwareDocumentRetriever(inner, true);
        }
        return new TracingDocumentRetriever(inner);
    }

    @Bean
    public Advisor hybridRagAdvisor(@Qualifier("ragDocumentRetriever") DocumentRetriever ragDocumentRetriever) {
        return RetrievalAugmentationAdvisor.builder()
                .documentRetriever(ragDocumentRetriever)
                .queryAugmenter(ContextualQueryAugmenterFactory.createInstance())
                .build();
    }

    /**
     * 检索专家专用 RAG Advisor：空上下文时使用“要求先调用 searchWeb”的模板，从而触发上网查询。
     */
    @Bean
    @Qualifier("retrievalExpertRagAdvisor")
    public Advisor retrievalExpertRagAdvisor(@Qualifier("ragDocumentRetriever") DocumentRetriever ragDocumentRetriever) {
        return RetrievalAugmentationAdvisor.builder()
                .documentRetriever(ragDocumentRetriever)
                .queryAugmenter(ContextualQueryAugmenterFactory.createInstanceForRetrievalExpertWithWebFallback())
                .build();
    }
}
