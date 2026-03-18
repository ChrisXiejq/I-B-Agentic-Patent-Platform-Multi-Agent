package com.inovationbehavior.backend.ai.rag.graphrag;

import java.util.List;

/**
 * 从文本中抽取实体（专利号、技术领域、机构、关键概念等），用于 GraphRAG 实体-文档索引与查询时召回。
 */
public interface EntityExtractor {

    /**
     * 从给定文本中抽取实体列表（可用于文档建索引或查询）。
     */
    List<String> extractFromText(String text);
}
