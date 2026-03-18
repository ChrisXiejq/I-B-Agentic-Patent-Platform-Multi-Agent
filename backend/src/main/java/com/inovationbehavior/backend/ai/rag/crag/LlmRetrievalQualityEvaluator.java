package com.inovationbehavior.backend.ai.rag.crag;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;

import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * 使用 LLM 对「查询 + 召回文档」做相关性评分（0～1），用于 CRAG 分支决策。
 */
@Slf4j
public class LlmRetrievalQualityEvaluator implements RetrievalQualityEvaluator {

    private static final int MAX_DOCS_FOR_EVAL = 6;
    private static final int MAX_DOC_CHARS = 800;
    private static final Pattern NUMBER_PATTERN = Pattern.compile("0?\\.?\\d+");

    private final ChatModel chatModel;

    public LlmRetrievalQualityEvaluator(ChatModel chatModel) {
        this.chatModel = chatModel;
    }

    @Override
    public double scoreRelevance(Query query, List<Document> documents) {
        if (documents == null || documents.isEmpty()) return 0.0;
        String queryText = query != null && query.getQueryText() != null ? query.getQueryText() : "";
        List<Document> limited = documents.stream().limit(MAX_DOCS_FOR_EVAL).toList();
        String contextSnippet = limited.stream()
                .map(d -> truncate(d.getText(), MAX_DOC_CHARS))
                .collect(Collectors.joining("\n---\n"));

        String prompt = """
                You are a retrieval quality evaluator. Given the USER QUERY and the RETRIEVED DOCUMENTS, rate how relevant the documents are to the query.
                Output ONLY a single number between 0 and 1: 1 = highly relevant and sufficient to answer, 0 = irrelevant or useless. No explanation.
                USER QUERY: %s
                RETRIEVED DOCUMENTS:
                ---
                %s
                ---
                Number (0 to 1):
                """.formatted(truncate(queryText, 500), contextSnippet);

        try {
            ChatResponse resp = chatModel.call(new Prompt(prompt));
            String out = resp.getResult() != null && resp.getResult().getOutput() != null
                    ? resp.getResult().getOutput().getText() : "";
            return parseScore(out);
        } catch (Exception e) {
            log.warn("[CRAG] LLM evaluation failed, defaulting to 0.5: {}", e.getMessage());
            return 0.5;
        }
    }

    private static double parseScore(String raw) {
        if (raw == null || raw.isBlank()) return 0.5;
        var m = NUMBER_PATTERN.matcher(raw.trim());
        if (m.find()) {
            try {
                double v = Double.parseDouble(m.group());
                return Math.max(0, Math.min(1, v));
            } catch (NumberFormatException ignored) {}
        }
        return 0.5;
    }

    private static String truncate(String s, int max) {
        if (s == null) return "";
        s = s.trim();
        return s.length() <= max ? s : s.substring(0, max) + "...";
    }
}
