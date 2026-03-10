package com.inovationbehavior.backend.controller;

import com.inovationbehavior.backend.ai.agent.IBManus;
import com.inovationbehavior.backend.ai.app.IBApp;
import jakarta.annotation.Resource;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;
import reactor.core.publisher.Flux;

import java.io.IOException;
import java.util.Optional;

/**
 * AI 与 Agent 统一入口：支持基础对话、RAG、工具调用及全能力 Agent（记忆 + RAG + 工具）。
 */
@RestController
@RequestMapping("/ai")
@Validated
public class AiController {

    @Resource
    private IBApp ibApp;

    @Resource
    private ToolCallback[] allTools;

    @Resource
    private ChatModel dashscopeChatModel;

    // ==================== 全能力 Agent 统一入口（推荐） ====================

    /**
     * 全能力 Agent 统一入口（POST，推荐）
     * 能力：多轮记忆 + RAG 检索增强（向量+BM25 融合）+ 工具调用（专利详情/热度/用户身份等）
     * 请求体：{ "message": "用户输入", "chatId": "会话ID（可选）", "stream": true/false }
     * stream=true 时返回 SSE 流，stream=false 时返回 JSON。
     */
    @PostMapping(value = "/agent", consumes = MediaType.APPLICATION_JSON_VALUE)
    public Object fullAgentPost(@Valid @RequestBody AgentRequest request) {
        String chatId = Optional.ofNullable(request.chatId()).orElse("default");
        if (Boolean.FALSE.equals(request.stream())) {
            String content = ibApp.doChatWithFullAgent(request.message(), chatId);
            return new AgentResponse(true, content, chatId, null);
        }
        return fullAgentStreamSse(request.message(), chatId);
    }

    /**
     * 全能力 Agent 统一入口（GET）
     * 参数：message（必填）、chatId（可选）、stream（可选，默认 true 表示 SSE）
     */
    @GetMapping(value = "/agent", produces = { MediaType.APPLICATION_JSON_VALUE, MediaType.TEXT_EVENT_STREAM_VALUE })
    public Object fullAgentGet(
            @RequestParam @NotBlank(message = "message is required") String message,
            @RequestParam(required = false) String chatId,
            @RequestParam(required = false, defaultValue = "true") boolean stream) {
        String resolvedChatId = Optional.ofNullable(chatId).orElse("default");
        if (!stream) {
            String content = ibApp.doChatWithFullAgent(message, resolvedChatId);
            return new AgentResponse(true, content, resolvedChatId, null);
        }
        return fullAgentStreamSse(message, resolvedChatId);
    }

    /** 全能力 Agent SSE 流：返回 SseEmitter，便于前端 EventSource 消费 */
    private SseEmitter fullAgentStreamSse(String message, String chatId) {
        SseEmitter emitter = new SseEmitter(300_000L); // 5 分钟
        ibApp.doChatWithFullAgentStream(message, chatId)
                .subscribe(
                        chunk -> {
                            try {
                                emitter.send(SseEmitter.event().data(chunk));
                            } catch (IOException e) {
                                emitter.completeWithError(e);
                            }
                        },
                        emitter::completeWithError,
                        emitter::complete
                );
        return emitter;
    }

    // ==================== 按能力拆分的入口（可按需调用） ====================

    /** 同步：仅多轮对话（无 RAG、无工具） */
    @GetMapping("/chat/sync")
    public String doChatSync(
            @RequestParam String message,
            @RequestParam(required = false) String chatId) {
        return ibApp.doChat(message, Optional.ofNullable(chatId).orElse("default"));
    }

    /** 流式：仅多轮对话 */
    @GetMapping(value = "/chat/sse", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> doChatSse(
            @RequestParam String message,
            @RequestParam(required = false) String chatId) {
        return ibApp.doChatByStream(message, Optional.ofNullable(chatId).orElse("default"));
    }

    /** 流式：多轮对话，返回标准 SSE 事件 */
    @GetMapping(value = "/chat/server_sent_event", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<ServerSentEvent<String>> doChatServerSentEvent(
            @RequestParam String message,
            @RequestParam(required = false) String chatId) {
        return ibApp.doChatByStream(message, Optional.ofNullable(chatId).orElse("default"))
                .map(chunk -> ServerSentEvent.<String>builder().data(chunk).build());
    }

    /** 流式：多轮对话，SseEmitter（长超时） */
    @GetMapping(value = "/chat/sse_emitter", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter doChatSseEmitter(
            @RequestParam String message,
            @RequestParam(required = false) String chatId) {
        SseEmitter emitter = new SseEmitter(180_000L);
        ibApp.doChatByStream(message, Optional.ofNullable(chatId).orElse("default"))
                .subscribe(
                        chunk -> {
                            try {
                                emitter.send(chunk);
                            } catch (IOException e) {
                                emitter.completeWithError(e);
                            }
                        },
                        emitter::completeWithError,
                        emitter::complete
                );
        return emitter;
    }

    /** 同步：RAG 知识库问答（多轮记忆 + 向量+BM25 融合检索） */
    @GetMapping("/chat/rag")
    public String doChatRag(
            @RequestParam String message,
            @RequestParam(required = false) String chatId) {
        return ibApp.doChatWithRag(message, Optional.ofNullable(chatId).orElse("default"));
    }

    /** 流式：多轮对话 + 工具调用（无 RAG） */
    @GetMapping(value = "/chat/tools", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter doChatWithTools(
            @RequestParam String message,
            @RequestParam(required = false) String chatId) {
        SseEmitter emitter = new SseEmitter(180_000L);
        ibApp.doChatWithTools(message, Optional.ofNullable(chatId).orElse("default"))
                .subscribe(
                        chunk -> {
                            try {
                                emitter.send(chunk);
                            } catch (IOException e) {
                                emitter.completeWithError(e);
                            }
                        },
                        emitter::completeWithError,
                        emitter::complete
                );
        return emitter;
    }

    /** 流式：Manus 超级智能体（ReAct 多步工具调用，无记忆、无 RAG） */
    @GetMapping(value = "/manus/chat", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter doChatWithManus(@RequestParam String message) {
        IBManus ibManus = new IBManus(allTools, dashscopeChatModel);
        return ibManus.runStream(message);
    }

    // ==================== DTO ====================

    /** 全能力 Agent 请求体 */
    public record AgentRequest(
            @NotBlank(message = "message is required") String message,
            String chatId,
            Boolean stream
    ) {}

    /** 全能力 Agent 同步响应 */
    public record AgentResponse(
            boolean success,
            String content,
            String chatId,
            String error
    ) {}
}
