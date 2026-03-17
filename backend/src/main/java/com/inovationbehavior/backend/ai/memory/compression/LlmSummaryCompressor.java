package com.inovationbehavior.backend.ai.memory.compression;

import com.inovationbehavior.backend.ai.memory.model.MemoryTurnRecord;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.stream.Collectors;

/**
 * 基于 LLM 的对话摘要压缩，用于短期记忆的滑动窗口溢出部分。
 */
@Slf4j
@Component
@ConditionalOnBean(ChatModel.class)
public class LlmSummaryCompressor implements SummaryCompressor {

    private final ChatClient chatClient;
    private final int maxSummaryTokens;

    public LlmSummaryCompressor(ChatModel chatModel,
                                @Value("${app.memory.working.summary-max-tokens:300}") int maxSummaryTokens) {
        this.chatClient = ChatClient.builder(chatModel).build();
        this.maxSummaryTokens = maxSummaryTokens;
    }

    @Override
    public String compress(String previousSummary, List<MemoryTurnRecord> turnsToCompress) {
        if (turnsToCompress == null || turnsToCompress.isEmpty()) {
            return previousSummary != null ? previousSummary : "";
        }
        String turnsBlock = turnsToCompress.stream()
                .map(MemoryTurnRecord::toCompactText)
                .collect(Collectors.joining("\n\n"));

        String prompt = """
                请将以下对话压缩为一段简洁摘要（保留：专利号、用户意图、关键结论与承诺），字数控制在 %d 字以内。仅输出摘要，不要解释。
                %s
                %s
                """
                .formatted(
                        Math.max(80, maxSummaryTokens),
                        (previousSummary != null && !previousSummary.isBlank())
                                ? "已有摘要（可与之合并）：\n" + previousSummary + "\n\n新增对话："
                                : "对话内容：",
                        turnsBlock);

        try {
            String summary = chatClient.prompt()
                    .user(prompt)
                    .call()
                    .content();
            return summary != null ? summary.trim() : "";
        } catch (Exception e) {
            log.warn("Summary compression failed, falling back to truncation", e);
            return fallbackCompress(previousSummary, turnsToCompress);
        }
    }

    private String fallbackCompress(String previousSummary, List<MemoryTurnRecord> turnsToCompress) {
        String block = turnsToCompress.stream()
                .map(MemoryTurnRecord::toCompactText)
                .limit(3)
                .collect(Collectors.joining(" | "));
        if (block.length() > 400) {
            block = block.substring(0, 397) + "...";
        }
        return (previousSummary != null ? previousSummary + "\n" : "") + block;
    }
}
