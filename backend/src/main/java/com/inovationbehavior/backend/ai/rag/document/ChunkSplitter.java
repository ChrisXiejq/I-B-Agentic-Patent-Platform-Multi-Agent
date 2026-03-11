package com.inovationbehavior.backend.ai.rag.document;

import org.springframework.ai.document.Document;

import java.util.List;

/**
 * RAG 文档分块统一接口：将原始文档列表切分为块级 Document 列表。
 * 实现类包括：按字符递归切分（LangChain4jRecursiveSplitter）、语义切分+Token 回退（SemanticChunkSplitter）。
 */
public interface ChunkSplitter {

    /**
     * 对原始文档列表分块，返回块级 Document 列表（供向量库与 BM25 共用）。
     */
    List<Document> apply(List<Document> documents);
}
