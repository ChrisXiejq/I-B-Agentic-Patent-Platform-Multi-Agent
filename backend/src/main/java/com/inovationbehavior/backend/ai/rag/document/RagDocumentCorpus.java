package com.inovationbehavior.backend.ai.rag.document;

import org.springframework.ai.document.Document;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * RAG 文档语料：从 documents 目录加载并分块后的块列表。
 * 该语料同时用于：pgvector 向量库存储（向量检索）、BM25 内存倒排索引（关键词检索）。
 */
@Component
public class RagDocumentCorpus {

    private final List<Document> documents;

    public RagDocumentCorpus(DocumentLoader documentLoader) {
        this.documents = documentLoader.loadDocumentsForRag();
    }

    public List<Document> getDocuments() {
        return documents;
    }
}
