package com.inovationbehavior.backend.ai.audit;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * 按 session 查询 Agent 审计 Trace。仅在启用 app.agent.trace-persist 且建表后可用。
 */
@Service
@ConditionalOnBean(AgentTraceRepository.class)
public class AgentTraceService {

    private final AgentTraceRepository repository;

    @Value("${app.agent.trace-query-limit:50}")
    private int defaultLimit;

    public AgentTraceService(AgentTraceRepository repository) {
        this.repository = repository;
    }

    public List<AgentTraceRecord> findBySessionId(String sessionId) {
        return repository.findBySessionId(sessionId, defaultLimit);
    }

    public List<AgentTraceRecord> findBySessionId(String sessionId, int limit) {
        return repository.findBySessionId(sessionId, Math.min(limit, 200));
    }

    public java.util.Optional<AgentTraceRecord> findById(Long id) {
        return repository.findById(id);
    }
}
