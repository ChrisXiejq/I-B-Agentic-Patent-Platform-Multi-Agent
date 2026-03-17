package com.inovationbehavior.backend.ai.memory.advisor;

import com.inovationbehavior.backend.ai.memory.working.WorkingMemoryService;
import com.inovationbehavior.backend.ai.memory.longterm.LongTermMemoryService;
import com.inovationbehavior.backend.ai.memory.importance.ImportanceScorer;
import com.inovationbehavior.backend.ai.memory.experiential.ExperientialMemoryService;
import com.inovationbehavior.backend.ai.memory.tool.MemoryRetrievalTool;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClientMessageAggregator;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.CallAdvisor;
import org.springframework.ai.chat.client.advisor.api.CallAdvisorChain;
import org.springframework.ai.chat.client.advisor.api.StreamAdvisor;
import org.springframework.ai.chat.client.advisor.api.StreamAdvisorChain;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.core.Ordered;
import reactor.core.publisher.Flux;

import java.util.Optional;

/**
 * 记忆持久化 Advisor：仅负责每轮对话结束后写入三层记忆，不向 Prompt 注入任何记忆。
 * 检索由 Agent 按需调用 {@link MemoryRetrievalTool#retrieveHistory} 完成，节省 Token 并减少干扰。
 */
@Slf4j
public class MemoryPersistenceAdvisor implements CallAdvisor, StreamAdvisor {

    private final WorkingMemoryService workingMemoryService;
    private final LongTermMemoryService longTermMemoryService;
    private final ImportanceScorer importanceScorer;
    private final ExperientialMemoryService experientialMemoryService;

    public MemoryPersistenceAdvisor(WorkingMemoryService workingMemoryService,
                                    LongTermMemoryService longTermMemoryService,
                                    ImportanceScorer importanceScorer,
                                    ExperientialMemoryService experientialMemoryService) {
        this.workingMemoryService = workingMemoryService;
        this.longTermMemoryService = longTermMemoryService;
        this.importanceScorer = importanceScorer;
        this.experientialMemoryService = experientialMemoryService;
    }

    public MemoryPersistenceAdvisor(WorkingMemoryService workingMemoryService,
                                    LongTermMemoryService longTermMemoryService,
                                    ImportanceScorer importanceScorer) {
        this(workingMemoryService, longTermMemoryService, importanceScorer, null);
    }

    @Override
    public String getName() {
        return "MemoryPersistenceAdvisor";
    }

    @Override
    public int getOrder() {
        return Ordered.LOWEST_PRECEDENCE + 100;
    }

    @Override
    public ChatClientResponse adviseCall(ChatClientRequest request, CallAdvisorChain chain) {
        ChatClientResponse response = chain.nextCall(request);
        String conversationId = getConversationId(request);
        String userText = extractUserText(request);
        if (conversationId != null && userText != null && !userText.isBlank()) {
            String assistantText = response.chatResponse().getResult().getOutput().getText();
            persistTurn(conversationId, userText, assistantText);
        }
        return response;
    }

    @Override
    public Flux<ChatClientResponse> adviseStream(ChatClientRequest request, StreamAdvisorChain chain) {
        Flux<ChatClientResponse> flux = chain.nextStream(request);
        String conversationId = getConversationId(request);
        String userText = extractUserText(request);
        return new ChatClientMessageAggregator().aggregateChatClientResponse(flux, aggregated -> {
            if (conversationId != null && userText != null && !userText.isBlank()) {
                String assistantText = aggregated.chatResponse().getResult().getOutput().getText();
                persistTurn(conversationId, userText, assistantText);
            }
        });
    }

    private void persistTurn(String conversationId, String userMessage, String assistantMessage) {
        Optional<String> compressed = workingMemoryService.addTurn(conversationId, userMessage, assistantMessage);
        if (experientialMemoryService != null) {
            compressed.ifPresent(summary -> experientialMemoryService.addSummary(conversationId, summary));
        }
        double importance = workingMemoryService.getLatestImportance(conversationId);
        int turnIndex = Math.max(0, workingMemoryService.getCurrentTurnIndex(conversationId) - 1);
        longTermMemoryService.storeIfEligible(conversationId, userMessage, assistantMessage, importance, turnIndex);
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
