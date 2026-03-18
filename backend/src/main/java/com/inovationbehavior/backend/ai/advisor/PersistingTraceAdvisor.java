package com.inovationbehavior.backend.ai.advisor;

import com.inovationbehavior.backend.ai.audit.AgentTraceRecord;
import com.inovationbehavior.backend.ai.audit.AgentTraceRepository;
import com.inovationbehavior.backend.ai.audit.RetrievalTraceContext;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClientMessageAggregator;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.CallAdvisor;
import org.springframework.ai.chat.client.advisor.api.CallAdvisorChain;
import org.springframework.ai.chat.client.advisor.api.StreamAdvisor;
import org.springframework.ai.chat.client.advisor.api.StreamAdvisorChain;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.core.Ordered;
import reactor.core.publisher.Flux;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 在 AgentTraceAdvisor 基础上将每次调用的请求、响应、工具调用与检索结果落库，支持按 session 查询审计。
 * 检索结果来自 {@link com.inovationbehavior.backend.ai.rag.retrieval.TracingDocumentRetriever} 写入的 {@link RetrievalTraceContext}。
 */
@Slf4j
public class PersistingTraceAdvisor implements CallAdvisor, StreamAdvisor {

    public static final String CONTEXT_STEP_TYPE = "agent.stepType";

    private final AgentTraceRepository repository;
    private static final int PREVIEW_LEN = 2000;

    public PersistingTraceAdvisor(AgentTraceRepository repository) {
        this.repository = repository;
    }

    @Override
    public String getName() {
        return "PersistingTraceAdvisor";
    }

    @Override
    public int getOrder() {
        return Ordered.LOWEST_PRECEDENCE - 199; // 在 AgentTraceAdvisor 之后
    }

    @Override
    public ChatClientResponse adviseCall(ChatClientRequest request, CallAdvisorChain chain) {
        ChatClientResponse response = chain.nextCall(request);
        persist(request, response);
        return response;
    }

    @Override
    public Flux<ChatClientResponse> adviseStream(ChatClientRequest request, StreamAdvisorChain chain) {
        Flux<ChatClientResponse> flux = chain.nextStream(request);
        return new ChatClientMessageAggregator().aggregateChatClientResponse(flux, resp -> persist(request, resp));
    }

    private void persist(ChatClientRequest request, ChatClientResponse clientResponse) {
        try {
            String sessionId = getSessionId(request);
            String stepType = getStepType(request);
            String requestPreview = getRequestPreview(request);
            String responsePreview = "";
            String toolCallsJson = null;
            var chatResponse = clientResponse.chatResponse();
            if (chatResponse != null && chatResponse.getResult() != null && chatResponse.getResult().getOutput() != null) {
                Message out = chatResponse.getResult().getOutput();
                responsePreview = abbreviate(out.getText(), PREVIEW_LEN);
                if (out instanceof AssistantMessage am && am.getToolCalls() != null && !am.getToolCalls().isEmpty()) {
                    toolCallsJson = buildToolCallsJson(am.getToolCalls());
                }
            }
            String retrievalDocsJson = buildRetrievalDocsJson(RetrievalTraceContext.getAndClear());

            AgentTraceRecord record = AgentTraceRecord.builder()
                    .sessionId(sessionId)
                    .stepType(stepType)
                    .requestPreview(requestPreview)
                    .responsePreview(responsePreview)
                    .toolCallsJson(toolCallsJson)
                    .retrievalDocsJson(retrievalDocsJson)
                    .build();
            repository.save(record);
        } catch (Exception e) {
            log.warn("[PersistingTraceAdvisor] Failed to persist trace: {}", e.getMessage());
        }
    }

    private static String getSessionId(ChatClientRequest request) {
        Object cid = request.context().get(ChatMemory.CONVERSATION_ID);
        return cid != null ? cid.toString() : "";
    }

    private static String getStepType(ChatClientRequest request) {
        Object v = request.context().get(CONTEXT_STEP_TYPE);
        return v != null ? v.toString() : null;
    }

    private static String getRequestPreview(ChatClientRequest request) {
        if (request.prompt() == null || request.prompt().getUserMessage() == null) return "";
        return abbreviate(request.prompt().getUserMessage().getText(), PREVIEW_LEN);
    }

    private static String buildToolCallsJson(List<AssistantMessage.ToolCall> toolCalls) {
        if (toolCalls == null || toolCalls.isEmpty()) return null;
        List<Map<String, String>> list = new ArrayList<>();
        for (AssistantMessage.ToolCall tc : toolCalls) {
            list.add(Map.of("name", tc.name() != null ? tc.name() : "", "arguments", abbreviate(tc.arguments(), 500)));
        }
        return simpleJsonList(list);
    }

    private static String buildRetrievalDocsJson(RetrievalTraceContext.RetrievalTrace trace) {
        if (trace == null || trace.documents().isEmpty()) return null;
        List<String> sources = trace.documents().stream()
                .map(d -> d.getMetadata() != null && d.getMetadata().get("source") != null
                        ? String.valueOf(d.getMetadata().get("source"))
                        : "")
                .limit(20)
                .toList();
        return "{\"query\":\"" + escapeJson(trace.queryText()) + "\",\"count\":" + trace.documents().size() + ",\"sources\":" + simpleJsonStringList(sources) + "}";
    }

    private static String simpleJsonList(List<Map<String, String>> list) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < list.size(); i++) {
            if (i > 0) sb.append(",");
            Map<String, String> m = list.get(i);
            sb.append("{\"name\":\"").append(escapeJson(m.get("name"))).append("\",\"arguments\":\"").append(escapeJson(m.get("arguments"))).append("\"}");
        }
        sb.append("]");
        return sb.toString();
    }

    private static String simpleJsonStringList(List<String> list) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < list.size(); i++) {
            if (i > 0) sb.append(",");
            sb.append("\"").append(escapeJson(list.get(i))).append("\"");
        }
        sb.append("]");
        return sb.toString();
    }

    private static String escapeJson(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r");
    }

    private static String abbreviate(String s, int maxLen) {
        if (s == null) return "";
        s = s.trim();
        return s.length() <= maxLen ? s : s.substring(0, maxLen) + "...";
    }
}
