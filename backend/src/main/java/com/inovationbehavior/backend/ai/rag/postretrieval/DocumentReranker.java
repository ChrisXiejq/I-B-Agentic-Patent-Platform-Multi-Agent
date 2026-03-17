package com.inovationbehavior.backend.ai.rag.postretrieval;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;

import java.util.List;

/**
 * 检索后精排：对融合后的候选文档按与 query 的相关性重新排序并截断为 topK。
 * 实现类可以是基于 Embedding 相似度，或 Cohere/Cross-Encoder 等专用 Rerank 模型。
 */
public interface DocumentReranker {

    /**
     * 对候选文档重排序，返回按相关性从高到低的 topK 条。
     *
     * @param query     用户查询
     * @param documents 候选文档（通常为向量+BM25 融合后的列表）
     * @return 重排后的文档列表，数量不超过实现内配置的 topK
     */
    List<Document> rerank(Query query, List<Document> documents);
}
