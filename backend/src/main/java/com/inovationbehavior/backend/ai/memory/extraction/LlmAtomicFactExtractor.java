package com.inovationbehavior.backend.ai.memory.extraction;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.stereotype.Component;

import java.util.Collections;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * 基于 LLM 的原子事实抽取：专利领域 entity-attribute-value，便于长期记忆 Zettelkasten 存储。
 */
@Slf4j
@Component
@ConditionalOnBean(ChatModel.class)
public class LlmAtomicFactExtractor implements AtomicFactExtractor {

    private static final Pattern JSON_LIKE = Pattern.compile("\\{\\s*\"entity\"\\s*:");

    private final ChatClient chatClient;

    public LlmAtomicFactExtractor(ChatModel chatModel) {
        this.chatClient = ChatClient.builder(chatModel).build();
    }

    @Override
    public List<String> extractFacts(String userMessage, String assistantMessage) {
        String combined = Stream.of(userMessage, assistantMessage)
                .filter(s -> s != null && !s.isBlank())
                .reduce("", (a, b) -> a + " " + b)
                .trim();
        if (combined.isBlank()) return Collections.emptyList();

        String prompt = """
                从以下对话中抽取原子事实，每条事实为一行 JSON，格式：{"entity":"实体（如专利号/技术名）","attribute":"属性（如法律状态/许可费用/授权日）","value":"值","source":"User_Input或Assistant"}。
                只输出 JSON 行，不要其他解释；若无明确事实则输出空。
                对话：
                %s
                """
                .formatted(combined);

        try {
            String out = chatClient.prompt().user(prompt).call().content();
            if (out == null || out.isBlank()) return Collections.emptyList();
            return Stream.of(out.split("\n"))
                    .map(String::trim)
                    .filter(s -> JSON_LIKE.matcher(s).find())
                    .limit(10)
                    .collect(Collectors.toList());
        } catch (Exception e) {
            log.warn("Atomic fact extraction failed", e);
            return Collections.emptyList();
        }
    }
}
