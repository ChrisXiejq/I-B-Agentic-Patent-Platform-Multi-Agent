package com.inovationbehavior.backend.ai.rag.crag;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;

import java.util.List;

/**
 * 检索质量评估器：对「查询 + 召回文档」给出置信度或 CRAG 分支，用于检索后门控。
 */
public interface RetrievalQualityEvaluator {

    /**
     * 评估检索结果与查询的相关性置信度。
     *
     * @param query 用户查询
     * @param documents 召回并 rerank 后的文档列表
     * @return 0.0～1.0，1 表示完全相关，0 表示完全不相关
     */
    double scoreRelevance(Query query, List<Document> documents);

    /**
     * 根据置信度与阈值返回 CRAG 三分支决策（可由默认实现基于 scoreRelevance 计算）。
     */
    default CragDecision evaluate(Query query, List<Document> documents,
                                   double correctThreshold, double ambiguousThreshold) {
        if (documents == null || documents.isEmpty()) return CragDecision.INCORRECT;
        double score = scoreRelevance(query, documents);
        if (score >= correctThreshold) return CragDecision.CORRECT;
        if (score >= ambiguousThreshold) return CragDecision.AMBIGUOUS;
        return CragDecision.INCORRECT;
    }
}
