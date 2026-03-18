package com.inovationbehavior.backend.ai.audit;

import java.util.List;
import java.util.Optional;

/**
 * Agent 审计 Trace 持久化与按 session 查询。
 */
public interface AgentTraceRepository {

    AgentTraceRecord save(AgentTraceRecord record);

    Optional<AgentTraceRecord> findById(Long id);

    /** 按会话 ID 查询，按时间倒序，限制条数 */
    List<AgentTraceRecord> findBySessionId(String sessionId, int limit);
}
