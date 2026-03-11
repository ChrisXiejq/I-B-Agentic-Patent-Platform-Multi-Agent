package com.inovationbehavior.backend.ai.rag.document;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * RAG 分块器配置。支持两种策略：递归字符切分（默认）、语义切分 + Token 回退（app.rag.splitter.use-semantic=true）。
 */
@Configuration
public class RagSplitterConfig {

    @Value("${app.rag.splitter.chunk-size:800}")
    private int chunkSize;

    @Value("${app.rag.splitter.chunk-overlap:150}")
    private int chunkOverlap;

    @Value("${app.rag.splitter.use-semantic:false}")
    private boolean useSemantic;

    @Value("${app.rag.splitter.max-chunk-tokens:400}")
    private int maxChunkTokens;

    @Value("${app.rag.splitter.overlap-tokens:80}")
    private int overlapTokens;

    @Bean
    public TokenCounter tokenCounter() {
        return new SimpleTokenCounter();
    }

    @Bean
    public LangChain4jRecursiveSplitter langChain4jRecursiveSplitter() {
        return new LangChain4jRecursiveSplitter(chunkSize, chunkOverlap);
    }

    @Bean
    public SemanticChunkSplitter semanticChunkSplitter(TokenCounter tokenCounter) {
        return new SemanticChunkSplitter(maxChunkTokens, overlapTokens, tokenCounter);
    }

    @Bean
    public ChunkSplitter chunkSplitter(LangChain4jRecursiveSplitter recursiveSplitter,
                                       SemanticChunkSplitter semanticChunkSplitter) {
        return useSemantic ? semanticChunkSplitter : recursiveSplitter;
    }
}
