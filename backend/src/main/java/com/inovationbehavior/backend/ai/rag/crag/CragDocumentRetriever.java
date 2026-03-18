package com.inovationbehavior.backend.ai.rag.crag;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;

import java.util.List;

/**
 * CRAG 式检索后门控：在底层检索链之后对文档做质量评分，按置信度三分支：
 * CORRECT 直接返回文档；AMBIGUOUS 返回文档（可选过滤）；INCORRECT 返回空列表以触发空上下文策略（如 searchWeb）。
 */
@Slf4j
public class CragDocumentRetriever implements DocumentRetriever {

    private final DocumentRetriever delegate;
    private final RetrievalQualityEvaluator evaluator;
    private final double correctThreshold;
    private final double ambiguousThreshold;
    /** Ambiguous 时是否只保留前一半文档以降低噪声，false 表示全部返回 */
    private final boolean ambiguousTrimToHalf;

    public CragDocumentRetriever(DocumentRetriever delegate,
                                RetrievalQualityEvaluator evaluator,
                                double correctThreshold,
                                double ambiguousThreshold,
                                boolean ambiguousTrimToHalf) {
        this.delegate = delegate;
        this.evaluator = evaluator;
        this.correctThreshold = correctThreshold;
        this.ambiguousThreshold = ambiguousThreshold;
        this.ambiguousTrimToHalf = ambiguousTrimToHalf;
    }

    @Override
    public List<Document> retrieve(Query query) {
        List<Document> docs = delegate.retrieve(query);
        if (docs == null || docs.isEmpty()) {
            log.debug("[CRAG] No documents retrieved, INCORRECT (empty)");
            return List.of();
        }

        CragDecision decision = evaluator.evaluate(query, docs, correctThreshold, ambiguousThreshold);
        log.info("[CRAG] decision={} score_bucket between [{}, {}] docsSize={}",
                decision, ambiguousThreshold, correctThreshold, docs.size());

        return switch (decision) {
            case CORRECT -> docs;
            case AMBIGUOUS -> ambiguousTrimToHalf && docs.size() > 1
                    ? docs.subList(0, (docs.size() + 1) / 2)
                    : docs;
            case INCORRECT -> List.of();
        };
    }
}
