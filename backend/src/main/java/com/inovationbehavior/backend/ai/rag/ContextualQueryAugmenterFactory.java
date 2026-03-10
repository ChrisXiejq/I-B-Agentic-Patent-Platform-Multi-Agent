package com.inovationbehavior.backend.ai.rag;

import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.rag.generation.augmentation.ContextualQueryAugmenter;

/**
 * 创建上下文查询增强器的工厂
 */
public class ContextualQueryAugmenterFactory {

    public static ContextualQueryAugmenter createInstance() {
        PromptTemplate emptyContextPromptTemplate = new PromptTemplate("""
                You should output the following content:
                Sorry, no content related to your question was found in the current knowledge base. I mainly support patent commercialization, patent retrieval, and value assessment related consultations.
                You may directly enter a patent number to query details and heat, or describe your needs for me to invoke tools to assist.
                """);
        return ContextualQueryAugmenter.builder()
                .allowEmptyContext(false)
                .emptyContextPromptTemplate(emptyContextPromptTemplate)
                .build();
    }
}
