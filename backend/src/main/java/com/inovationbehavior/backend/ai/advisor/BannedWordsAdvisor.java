package com.inovationbehavior.backend.ai.advisor;

import com.inovationbehavior.backend.exception.BannedWordException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.CallAdvisor;
import org.springframework.ai.chat.client.advisor.api.CallAdvisorChain;
import org.springframework.ai.chat.client.advisor.api.StreamAdvisor;
import org.springframework.ai.chat.client.advisor.api.StreamAdvisorChain;
import org.springframework.core.Ordered;
import reactor.core.publisher.Flux;

import java.util.Set;
import java.util.stream.Collectors;

/**
 * 违禁词校验 Advisor：在调用模型前校验用户输入，若包含违禁词则直接拒绝并返回提示，不发起 LLM 调用。
 * 校验为大小写不敏感；支持配置词表。
 */
@Slf4j
public class BannedWordsAdvisor implements CallAdvisor, StreamAdvisor {

    private final Set<String> bannedWordsLower;
    private final String rejectMessage;

    /**
     * @param bannedWords 违禁词集合（可为空，空集合表示不拦截）
     * @param rejectMessage 命中时返回的提示文案，null 时使用默认文案
     */
    public BannedWordsAdvisor(Set<String> bannedWords, String rejectMessage) {
        this.bannedWordsLower = bannedWords == null
                ? Set.of()
                : bannedWords.stream().filter(w -> w != null && !w.isBlank()).map(String::trim).map(String::toLowerCase).collect(Collectors.toSet());
        this.rejectMessage = rejectMessage != null ? rejectMessage : "您的输入包含不当内容，请修改后重试。";
    }

    @Override
    public String getName() {
        return "BannedWordsAdvisor";
    }

    @Override
    public int getOrder() {
        return Ordered.HIGHEST_PRECEDENCE; // 最先执行，避免违禁请求进入后续链路
    }

    private void checkUserText(String userText) {
        if (bannedWordsLower.isEmpty() || userText == null || userText.isBlank()) {
            return;
        }
        String lower = userText.toLowerCase();
        for (String word : bannedWordsLower) {
            if (lower.contains(word)) {
                log.warn("Banned word detected in user input, word=[{}]", word);
                throw new BannedWordException(rejectMessage, word);
            }
        }
    }

    @Override
    public ChatClientResponse adviseCall(ChatClientRequest request, CallAdvisorChain chain) {
        String userText = request.prompt().getUserMessage().getText();
        checkUserText(userText);
        return chain.nextCall(request);
    }

    @Override
    public Flux<ChatClientResponse> adviseStream(ChatClientRequest request, StreamAdvisorChain chain) {
        String userText = request.prompt().getUserMessage().getText();
        checkUserText(userText);
        return chain.nextStream(request);
    }
}
