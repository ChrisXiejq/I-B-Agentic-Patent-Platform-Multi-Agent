package com.inovationbehavior.backend.ai.rag.graphrag;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * 使用 LLM 从文本中抽取实体（专利、技术、机构、关键概念等），支持中英文。
 */
@Slf4j
public class LlmEntityExtractor implements EntityExtractor {

    private static final int MAX_INPUT_CHARS = 1500;
    private static final Pattern SPLIT = Pattern.compile("[,，、;；\\n]");

    private final ChatModel chatModel;

    public LlmEntityExtractor(ChatModel chatModel) {
        this.chatModel = chatModel;
    }

    @Override
    public List<String> extractFromText(String text) {
        if (text == null || text.isBlank()) return List.of();
        String input = text.length() > MAX_INPUT_CHARS ? text.substring(0, MAX_INPUT_CHARS) + "..." : text;
        String prompt = """
                Extract key entities from the following text. Include: patent numbers (e.g. CN..., US...), \
                technology domains, company/organization names, and important concepts. \
                Output a comma-separated list only, no explanation. Use the same language as the text. \
                If none clearly found, output the most important 1-3 noun phrases.
                Text: %s
                """.formatted(input);
        try {
            ChatResponse resp = chatModel.call(new Prompt(prompt));
            String out = resp.getResult() != null && resp.getResult().getOutput() != null
                    ? resp.getResult().getOutput().getText() : "";
            return parseEntities(out);
        } catch (Exception e) {
            log.warn("[GraphRAG] Entity extraction failed: {}", e.getMessage());
            return List.of();
        }
    }

    private static List<String> parseEntities(String raw) {
        if (raw == null || raw.isBlank()) return List.of();
        List<String> list = new ArrayList<>();
        for (String s : SPLIT.split(raw)) {
            s = s.trim();
            if (s.length() > 1 && s.length() < 80) list.add(s);
        }
        return list.stream().distinct().limit(15).collect(Collectors.toList());
    }
}
