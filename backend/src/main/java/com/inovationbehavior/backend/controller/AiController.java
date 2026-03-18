package com.inovationbehavior.backend.controller;

import com.inovationbehavior.backend.ai.app.IBApp;
import com.inovationbehavior.backend.ai.audit.AgentTraceRecord;
import com.inovationbehavior.backend.ai.audit.AgentTraceService;
import jakarta.annotation.Resource;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

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

    @Autowired(required = false)
    private AgentTraceService agentTraceService;

    // ==================== Agent Trace 审计（按 session 查询） ====================

    /**
     * 按会话 ID 查询 Agent 审计 Trace（请求/响应/工具调用/检索结果）。
     * 需先执行 docs/schema-agent-trace.sql 建表并启用 PersistingTraceAdvisor。
     */
    @GetMapping(value = "/agent/trace", produces = MediaType.APPLICATION_JSON_VALUE)
    public List<AgentTraceRecord> getTraceBySession(
            @RequestParam @NotBlank(message = "sessionId is required") String sessionId,
            @RequestParam(required = false, defaultValue = "50") int limit) {
        if (agentTraceService == null) return List.of();
        return agentTraceService.findBySessionId(sessionId, limit);
    }

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
        String content = ibApp.doChatWithMultiAgentOrFull(request.message(), chatId);
        return new AgentResponse(true, content, chatId, null);
    }

    // ==================== P&E 图 SSE 流式（多步规划 + Replan，完成后推送整段回复） ====================

    /**
     * P&E 图 SSE 流式（GET）
     * 执行 Planner → Executor(retrieval/analysis/advice) → CheckResult → Replan/Synthesize，完成后通过 SSE 推送最终回复。
     * 非逐 token 流式，而是图跑完后以 SSE 事件推送整段 content，便于前端用 EventSource 统一接入。
     */
    @GetMapping(value = "/agent/pe-stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter peStreamGet(
            @RequestParam @NotBlank(message = "message is required") String message,
            @RequestParam(required = false) String chatId) {
        return peStreamSse(message, Optional.ofNullable(chatId).orElse("default"));
    }

    /**
     * P&E 图 SSE 流式（POST）
     * 请求体：{ "message": "用户输入", "chatId": "会话ID（可选）" }
     */
    @PostMapping(value = "/agent/pe-stream", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter peStreamPost(@Valid @RequestBody AgentRequest request) {
        String chatId = Optional.ofNullable(request.chatId()).orElse("default");
        return peStreamSse(request.message(), chatId);
    }

    private SseEmitter peStreamSse(String message, String chatId) {
        SseEmitter emitter = new SseEmitter(300_000L); // 5 分钟
        CompletableFuture
                .supplyAsync(() -> ibApp.doChatWithMultiAgentOrFull(message, chatId))
                .thenAccept(content -> {
                    try {
                        emitter.send(SseEmitter.event().data(content != null ? content : ""));
                        emitter.complete();
                    } catch (IOException e) {
                        emitter.completeWithError(e);
                    }
                })
                .exceptionally(ex -> {
                    try {
                        String err = ex.getCause() != null ? ex.getCause().getMessage() : ex.getMessage();
                        emitter.send(SseEmitter.event().name("error").data(err != null ? err : "Unknown error"));
                    } catch (IOException ignored) { }
                    emitter.completeWithError(ex);
                    return null;
                });
        return emitter;
    }

    // ==================== DTO ====================

    /** 全能力 Agent 请求体 */
    public record AgentRequest(
            @NotBlank(message = "message is required") String message,
            String chatId
    ) {}

    /** 全能力 Agent 同步响应 */
    public record AgentResponse(
            boolean success,
            String content,
            String chatId,
            String error
    ) {}
}
