package com.inovationbehavior.backend.ai.eval;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * 单条 RAG 评估用例：问题 + 期望被检索到的来源（用于 Recall） + 可选参考答案（用于 Faithfulness 等）。
 */
public record RagTestCase(
        String query,
        List<String> expectedSourceSubstrings,
        String referenceAnswer
) {
    @JsonCreator
    public RagTestCase(
            @JsonProperty("query") String query,
            @JsonProperty("expectedSourceSubstrings") List<String> expectedSourceSubstrings,
            @JsonProperty("referenceAnswer") String referenceAnswer) {
        this.query = query != null ? query : "";
        this.expectedSourceSubstrings = expectedSourceSubstrings != null ? expectedSourceSubstrings : List.of();
        this.referenceAnswer = referenceAnswer != null ? referenceAnswer : "";
    }

    public RagTestCase(String query, List<String> expectedSourceSubstrings) {
        this(query, expectedSourceSubstrings, null);
    }
}
