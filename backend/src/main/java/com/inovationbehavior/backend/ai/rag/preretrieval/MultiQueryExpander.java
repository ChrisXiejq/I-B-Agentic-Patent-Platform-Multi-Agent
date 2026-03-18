package com.inovationbehavior.backend.ai.rag.preretrieval;

import org.springframework.ai.rag.Query;

import java.util.List;

/**
 * 将用户查询扩展为多条子查询，用于多路检索后 RRF 融合（迭代/多轮 RAG 的多查询融合）。
 */
public interface MultiQueryExpander {

    /**
     * 将原始 query 扩展为多条查询（可包含原查询），便于从不同角度召回后融合。
     */
    List<Query> expand(Query query);
}
