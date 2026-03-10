package com.inovationbehavior.backend.ai.rag;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * RAG 分块器配置（独立于 HybridRagConfig，避免 DocumentLoader → splitter → HybridRagConfig 循环依赖）。
 */
@Configuration
public class RagSplitterConfig {

    @Value("${app.rag.splitter.chunk-size:800}")
    private int chunkSize;

    @Value("${app.rag.splitter.chunk-overlap:150}")
    private int chunkOverlap;

    @Bean
    public LangChain4jRecursiveSplitter langChain4jRecursiveSplitter() {
        return new LangChain4jRecursiveSplitter(chunkSize, chunkOverlap);
    }
}
