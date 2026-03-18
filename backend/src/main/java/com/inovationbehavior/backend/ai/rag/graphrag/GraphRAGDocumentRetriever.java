package com.inovationbehavior.backend.ai.rag.graphrag;

import com.inovationbehavior.backend.ai.rag.retrieval.RrfFusion;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;

import java.util.ArrayList;
import java.util.List;

/**
 * GraphRAG 风格检索：在底层检索基础上增加「按实体召回」一路，再 RRF 融合。
 * 查询时从 query 抽取实体，用 EntityDocumentIndex 召回相关文档，与主检索结果融合。
 */
public class GraphRAGDocumentRetriever implements DocumentRetriever {

    private final DocumentRetriever delegate;
    private final EntityDocumentIndex entityIndex;
    private final EntityExtractor entityExtractor;
    private final int entityTopK;

    public GraphRAGDocumentRetriever(DocumentRetriever delegate,
                                    EntityDocumentIndex entityIndex,
                                    EntityExtractor entityExtractor,
                                    int entityTopK) {
        this.delegate = delegate;
        this.entityIndex = entityIndex;
        this.entityExtractor = entityExtractor;
        this.entityTopK = Math.max(1, entityTopK);
    }

    @Override
    public List<Document> retrieve(Query query) {
        List<Document> mainDocs = delegate.retrieve(query);
        String queryText = query != null ? query.getQueryText() : "";
        List<String> entities = entityExtractor.extractFromText(queryText);
        List<Document> entityDocs = (entities.isEmpty() || entityIndex == null)
                ? List.of()
                : entityIndex.getDocumentsForEntities(entities, entityTopK);

        if (entityDocs.isEmpty()) return mainDocs;
        if (mainDocs.isEmpty()) return entityDocs;

        List<List<Document>> rankedLists = new ArrayList<>();
        rankedLists.add(mainDocs);
        rankedLists.add(entityDocs);
        return RrfFusion.fuse(rankedLists);
    }
}
