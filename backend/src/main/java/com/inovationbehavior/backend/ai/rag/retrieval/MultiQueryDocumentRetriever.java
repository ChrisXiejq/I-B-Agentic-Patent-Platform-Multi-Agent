package com.inovationbehavior.backend.ai.rag.retrieval;

import com.inovationbehavior.backend.ai.rag.preretrieval.MultiQueryExpander;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;

import java.util.ArrayList;
import java.util.List;

/**
 * 多查询融合：将 query 扩展为多条子查询，分别检索后 RRF 融合，再返回去重排序结果（迭代/多轮 RAG）。
 */
public class MultiQueryDocumentRetriever implements DocumentRetriever {

    private final DocumentRetriever delegate;
    private final MultiQueryExpander expander;

    public MultiQueryDocumentRetriever(DocumentRetriever delegate, MultiQueryExpander expander) {
        this.delegate = delegate;
        this.expander = expander;
    }

    @Override
    public List<Document> retrieve(Query query) {
        List<Query> queries = expander.expand(query);
        if (queries == null || queries.isEmpty()) return delegate.retrieve(query);
        if (queries.size() == 1) return delegate.retrieve(queries.get(0));

        List<List<Document>> rankedLists = new ArrayList<>();
        for (Query q : queries) {
            List<Document> docs = delegate.retrieve(q);
            if (docs != null && !docs.isEmpty()) rankedLists.add(docs);
        }
        if (rankedLists.isEmpty()) return List.of();
        return RrfFusion.fuse(rankedLists);
    }
}
