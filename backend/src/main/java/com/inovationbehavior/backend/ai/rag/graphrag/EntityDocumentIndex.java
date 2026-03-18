package com.inovationbehavior.backend.ai.rag.graphrag;

import org.springframework.ai.document.Document;

import java.util.List;

/**
 * GraphRAG 实体-文档索引：根据实体召回相关文档（局部/实体级检索）。
 */
public interface EntityDocumentIndex {

    /**
     * 根据实体列表召回文档，按实体命中数等排序，最多返回 topK 条。
     */
    List<Document> getDocumentsForEntities(List<String> entities, int topK);
}
