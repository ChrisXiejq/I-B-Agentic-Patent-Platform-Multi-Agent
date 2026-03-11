package com.inovationbehavior.backend.ai.eval;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * 单条导出记录，供 Python ragas 库读取。
 * 字段名与 ragas SingleTurnSample 一致：user_input, retrieved_contexts, response, reference。
 */
public record RagasExportRecord(
        @JsonProperty("user_input") String userInput,
        @JsonProperty("retrieved_contexts") List<String> retrievedContexts,
        @JsonProperty("response") String response,
        @JsonProperty("reference") String reference
) {
    public static RagasExportRecord of(String question, List<String> contexts, String answer, String referenceAnswer) {
        return new RagasExportRecord(
                question != null ? question : "",
                contexts != null ? contexts : List.of(),
                answer != null ? answer : "",
                referenceAnswer != null ? referenceAnswer : ""
        );
    }
}
