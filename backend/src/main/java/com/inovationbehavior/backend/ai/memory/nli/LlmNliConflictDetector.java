package com.inovationbehavior.backend.ai.memory.nli;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.document.Document;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.stream.Collectors;

/**
 * 基于 LLM 的 NLI 冲突检测：用自然语言推理判断新记忆与已有记忆是否矛盾。
 */
@Slf4j
@Component
@ConditionalOnBean(ChatModel.class)
public class LlmNliConflictDetector implements NliConflictDetector {

    private final ChatClient chatClient;

    public LlmNliConflictDetector(ChatModel chatModel) {
        this.chatClient = ChatClient.builder(chatModel).build();
    }

    @Override
    public boolean hasConflict(String newMemoryContent, List<Document> existingMemories) {
        if (newMemoryContent == null || newMemoryContent.isBlank() || existingMemories == null || existingMemories.isEmpty()) {
            return false;
        }
        String existingBlock = existingMemories.stream()
                .map(Document::getText)
                .filter(t -> t != null && !t.isBlank())
                .limit(5)
                .collect(Collectors.joining("\n---\n"));

        String prompt = """
                判断以下「新记忆」与「已有记忆」是否在事实或主张上相互矛盾（例如：同一专利的结论相反、用户意图前后冲突）。
                仅回答 YES 或 NO。
                新记忆：
                %s
                已有记忆：
                %s
                """
                .formatted(newMemoryContent, existingBlock);

        try {
            String answer = chatClient.prompt().user(prompt).call().content();
            boolean conflict = answer != null && answer.trim().toUpperCase().startsWith("YES");
            if (conflict) log.debug("NLI conflict detected for new memory (length {})", newMemoryContent.length());
            return conflict;
        } catch (Exception e) {
            log.warn("NLI check failed, assuming no conflict", e);
            return false;
        }
    }
}
