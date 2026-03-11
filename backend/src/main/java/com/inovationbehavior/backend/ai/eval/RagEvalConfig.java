package com.inovationbehavior.backend.ai.eval;

import jakarta.annotation.Resource;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * RAG 评估模块配置：注入 DocumentRetriever 与 ChatModel，供 RagEvalRunner 使用。
 */
@Configuration
public class RagEvalConfig {

    @Resource
    @Qualifier("hybridDocumentRetriever")
    private DocumentRetriever hybridDocumentRetriever;

    @Resource
    private ChatModel chatModel;

    @Value("${app.eval.rag.test-cases-location:classpath*:eval/rag-test-cases.json}")
    private String testCasesLocation;

    @Bean
    public RagEvalRunner ragEvalRunner() {
        return new RagEvalRunner(hybridDocumentRetriever, chatModel);
    }

    public String getTestCasesLocation() {
        return testCasesLocation;
    }
}
