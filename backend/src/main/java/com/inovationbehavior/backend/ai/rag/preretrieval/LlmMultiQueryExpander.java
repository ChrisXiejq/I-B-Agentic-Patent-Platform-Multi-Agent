package com.inovationbehavior.backend.ai.rag.preretrieval;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.rag.Query;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 使用 LLM 将用户查询扩展为 2～maxQueries 条子查询，用于多路检索融合。
 */
@Slf4j
public class LlmMultiQueryExpander implements MultiQueryExpander {

    private final ChatModel chatModel;
    private final int maxQueries;

    public LlmMultiQueryExpander(ChatModel chatModel, int maxQueries) {
        this.chatModel = chatModel;
        this.maxQueries = Math.max(2, Math.min(maxQueries, 5));
    }

    @Override
    public List<Query> expand(Query query) {
        String raw = query != null ? query.getQueryText() : "";
        if (raw == null || raw.isBlank()) return List.of(query != null ? query : new Query(""));

        try {
            String prompt = """
                    Given the following user question, generate %d different reformulations or sub-questions (in the same language) to improve retrieval. Output one question per line, no numbering. Include the original question as one of the lines.
                    Question: %s
                    """.formatted(maxQueries, raw.trim());
            ChatResponse resp = chatModel.call(new Prompt(prompt));
            String out = resp.getResult() != null && resp.getResult().getOutput() != null
                    ? resp.getResult().getOutput().getText() : "";
            List<String> lines = new ArrayList<>();
            for (String line : out.split("\n")) {
                line = line.trim();
                if (line.startsWith("-")) line = line.substring(1).trim();
                if (line.matches("\\d+[.)]\\s*.*")) line = line.replaceFirst("^\\d+[.)]\\s*", "");
                if (!line.isBlank()) lines.add(line);
            }
            if (lines.isEmpty()) return List.of(query);
            return lines.stream().limit(maxQueries).map(Query::new).collect(Collectors.toList());
        } catch (Exception e) {
            log.warn("[MultiQuery] LLM expand failed, using original query: {}", e.getMessage());
            return List.of(query);
        }
    }
}
