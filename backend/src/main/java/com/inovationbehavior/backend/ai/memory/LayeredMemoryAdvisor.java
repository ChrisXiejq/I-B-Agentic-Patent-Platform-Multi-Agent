package com.inovationbehavior.backend.ai.memory;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClientMessageAggregator;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.CallAdvisor;
import org.springframework.ai.chat.client.advisor.api.CallAdvisorChain;
import org.springframework.ai.chat.client.advisor.api.StreamAdvisor;
import org.springframework.ai.chat.client.advisor.api.StreamAdvisorChain;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.core.Ordered;
import reactor.core.publisher.Flux;

/**
 * 分层记忆 Advisor：短期（滑动窗口+摘要压缩+重要性衰减）+ 长期（BGE 相似度+importance 阈值+NLI 控制写入）。
 * 调用前注入短期摘要与最近轮次 + 长期语义检索结果；调用后更新短期并条件写入长期。
 */
@Slf4j
public class LayeredMemoryAdvisor implements CallAdvisor, StreamAdvisor {

    private final LongTermMemoryService longTermMemoryService;
    private final ImportanceScorer importanceScorer;
    private final ShortTermMemoryService shortTermMemoryService;

    public LayeredMemoryAdvisor(LongTermMemoryService longTermMemoryService,
                                ImportanceScorer importanceScorer,
                                ShortTermMemoryService shortTermMemoryService) {
        this.longTermMemoryService = longTermMemoryService;
        this.importanceScorer = importanceScorer;
        this.shortTermMemoryService = shortTermMemoryService;
    }

    /** 仅长期记忆（无短期摘要压缩时使用） */
    public LayeredMemoryAdvisor(LongTermMemoryService longTermMemoryService,
                                ImportanceScorer importanceScorer) {
        this(longTermMemoryService, importanceScorer, null);
    }

    @Override
    public String getName() {
        return "LayeredMemoryAdvisor";
    }

    @Override
    public int getOrder() {
        return Ordered.LOWEST_PRECEDENCE + 100;
    }

    @Override
    public ChatClientResponse adviseCall(ChatClientRequest request, CallAdvisorChain chain) {
        String conversationId = getConversationId(request);
        String userText = extractUserText(request);
        if (conversationId == null || userText == null || userText.isBlank()) {
            return chain.nextCall(request);
        }

        String memoryContext = buildMemoryContext(conversationId, userText);
        ChatClientRequest augmented = memoryContext.isBlank()
                ? request
                : augmentRequestWithMemory(request, memoryContext);

        ChatClientResponse response = chain.nextCall(augmented);

        String assistantText = response.chatResponse().getResult().getOutput().getText();
        afterTurn(conversationId, userText, assistantText);

        return response;
    }

    @Override
    public Flux<ChatClientResponse> adviseStream(ChatClientRequest request, StreamAdvisorChain chain) {
        String conversationId = getConversationId(request);
        String userText = extractUserText(request);
        if (conversationId == null || userText == null || userText.isBlank()) {
            return chain.nextStream(request);
        }

        String memoryContext = buildMemoryContext(conversationId, userText);
        ChatClientRequest augmented = memoryContext.isBlank()
                ? request
                : augmentRequestWithMemory(request, memoryContext);

        Flux<ChatClientResponse> flux = chain.nextStream(augmented);
        return new ChatClientMessageAggregator().aggregateChatClientResponse(flux, aggregated -> {
            String assistantText = aggregated.chatResponse().getResult().getOutput().getText();
            afterTurn(conversationId, userText, assistantText);
        });
    }

    private String buildMemoryContext(String conversationId, String userText) {
        StringBuilder sb = new StringBuilder();
        if (shortTermMemoryService != null) {
            String shortContext = shortTermMemoryService.getContextForPrompt(conversationId);
            if (!shortContext.isBlank()) {
                sb.append("\n[短期记忆（摘要+最近对话）]\n").append(shortContext).append("\n");
            }
        }
        String longContext = longTermMemoryService.formatMemoriesForPrompt(
                longTermMemoryService.retrieve(userText, conversationId));
        if (!longContext.isBlank()) {
            sb.append(longContext);
        }
        return sb.toString().trim();
    }

    private void afterTurn(String conversationId, String userMessage, String assistantMessage) {
        if (shortTermMemoryService != null) {
            shortTermMemoryService.addTurn(conversationId, userMessage, assistantMessage);
            double importance = shortTermMemoryService.getLatestImportance(conversationId);
            int turnIndex = Math.max(0, shortTermMemoryService.getCurrentTurnIndex(conversationId) - 1);
            longTermMemoryService.storeIfEligible(conversationId, userMessage, assistantMessage, importance, turnIndex);
        } else {
            double importance = importanceScorer.score(userMessage, assistantMessage, 0);
            longTermMemoryService.storeIfEligible(conversationId, userMessage, assistantMessage, importance, 0);
        }
    }

    private ChatClientRequest augmentRequestWithMemory(ChatClientRequest request, String memoryContext) {
        Prompt newPrompt = request.prompt().augmentSystemMessage(memoryContext);
        return new ChatClientRequest(newPrompt, request.context());
    }

    private String getConversationId(ChatClientRequest request) {
        Object id = request.context().get(ChatMemory.CONVERSATION_ID);
        return id != null ? id.toString() : null;
    }

    private String extractUserText(ChatClientRequest request) {
        return request.prompt() != null && request.prompt().getUserMessage() != null
                ? request.prompt().getUserMessage().getText() : null;
    }
}
