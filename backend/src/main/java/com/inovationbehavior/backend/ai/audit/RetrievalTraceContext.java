package com.inovationbehavior.backend.ai.audit;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;

import java.util.Collections;
import java.util.List;

/**
 * 当前请求的检索结果暂存，供 PersistingTraceAdvisor 落库。使用 ThreadLocal，请求结束时须清除。
 */
public final class RetrievalTraceContext {

    private static final ThreadLocal<RetrievalTrace> HOLDER = new ThreadLocal<>();

    public static void set(Query query, List<Document> documents) {
        HOLDER.set(new RetrievalTrace(query != null ? query.getQueryText() : null, documents));
    }

    public static RetrievalTrace getAndClear() {
        RetrievalTrace t = HOLDER.get();
        HOLDER.remove();
        return t;
    }

    public static RetrievalTrace get() {
        return HOLDER.get();
    }

    public record RetrievalTrace(String queryText, List<Document> documents) {
        public List<Document> documents() {
            return documents == null ? Collections.emptyList() : documents;
        }
    }
}
