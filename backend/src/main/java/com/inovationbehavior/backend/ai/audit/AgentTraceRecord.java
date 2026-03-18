package com.inovationbehavior.backend.ai.audit;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;

/**
 * 单次 Agent 调用的审计记录：请求、响应、工具调用、检索结果，支持按 session 查询。
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AgentTraceRecord {

    private Long id;
    /** 会话/对话 ID，与 chatId 或 conversationId 对应 */
    private String sessionId;
    /** 步骤类型：e.g. retrieval, analysis, advice, synthesize */
    private String stepType;
    private String requestPreview;
    private String responsePreview;
    /** JSON 数组：工具调用 name + arguments 摘要 */
    private String toolCallsJson;
    /** JSON：本次检索的 query + doc count + 各 doc source 摘要，便于审计与 Citation 追溯 */
    private String retrievalDocsJson;
    private Instant createdAt;
}
