package com.inovationbehavior.backend.ai.rag.retrieval;

import com.inovationbehavior.backend.ai.audit.RetrievalTraceContext;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;

import java.util.List;

/**
 * 包装底层 DocumentRetriever，将本次检索的 query 与 documents 写入 {@link RetrievalTraceContext}，
 * 供 {@link com.inovationbehavior.backend.ai.advisor.PersistingTraceAdvisor} 落库审计。
 */
public class TracingDocumentRetriever implements DocumentRetriever {

    private final DocumentRetriever delegate;

    public TracingDocumentRetriever(DocumentRetriever delegate) {
        this.delegate = delegate;
    }

    @Override
    public List<Document> retrieve(Query query) {
        List<Document> docs = delegate.retrieve(query);
        RetrievalTraceContext.set(query, docs);
        return docs;
    }
}
