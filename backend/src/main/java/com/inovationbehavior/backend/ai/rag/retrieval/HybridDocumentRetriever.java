package com.inovationbehavior.backend.ai.rag.retrieval;

import com.inovationbehavior.backend.ai.rag.postretrieval.EmbeddingReranker;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;
import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;

import java.util.ArrayList;
import java.util.List;

/**
 * 检索：多路召回（向量 + BM25）→ RRF 融合 → Rerank
 */
public class HybridDocumentRetriever implements DocumentRetriever {

    private final VectorStoreDocumentRetriever vectorRetriever;
    private final BM25DocumentRetriever bm25Retriever;
    private final EmbeddingReranker reranker;
    private final int vectorTopK;
    private final int bm25TopK;

    public HybridDocumentRetriever(
            VectorStoreDocumentRetriever vectorRetriever,
            BM25DocumentRetriever bm25Retriever,
            EmbeddingReranker reranker,
            int vectorTopK,
            int bm25TopK) {
        this.vectorRetriever = vectorRetriever;
        this.bm25Retriever = bm25Retriever;
        this.reranker = reranker;
        this.vectorTopK = Math.max(1, vectorTopK);
        this.bm25TopK = Math.max(1, bm25TopK);
    }

    @Override
    public List<Document> retrieve(Query query) {
        List<Document> vectorDocs = vectorRetriever.retrieve(query);
        List<Document> bm25Docs = bm25Retriever.retrieve(query);

        List<List<Document>> rankedLists = new ArrayList<>();
        if (!vectorDocs.isEmpty()) rankedLists.add(vectorDocs);
        if (!bm25Docs.isEmpty()) rankedLists.add(bm25Docs);

        if (rankedLists.isEmpty()) return List.of();

        List<Document> fused = RrfFusion.fuse(rankedLists);
        if (fused.isEmpty()) return List.of();

        return reranker.rerank(query, fused);
    }
}
