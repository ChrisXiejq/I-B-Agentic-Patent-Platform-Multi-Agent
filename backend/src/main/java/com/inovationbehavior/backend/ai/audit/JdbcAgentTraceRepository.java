package com.inovationbehavior.backend.ai.audit;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.support.GeneratedKeyHolder;
import org.springframework.stereotype.Repository;

import java.sql.Timestamp;
import java.time.Instant;
import java.util.List;
import java.util.Optional;

/**
 * 基于 JdbcTemplate 的 Agent Trace 持久化。表结构见 docs/schema-agent-trace.sql；需 MySQL 主库建表。
 */
@Repository
@ConditionalOnProperty(name = "app.agent.trace-persist", havingValue = "true")
public class JdbcAgentTraceRepository implements AgentTraceRepository {

    private static final String TABLE = "agent_trace";

    private final JdbcTemplate jdbc;

    public JdbcAgentTraceRepository(JdbcTemplate jdbcTemplate) {
        this.jdbc = jdbcTemplate;
    }

    @Override
    public AgentTraceRecord save(AgentTraceRecord record) {
        if (record.getCreatedAt() == null) record.setCreatedAt(Instant.now());
        GeneratedKeyHolder keyHolder = new GeneratedKeyHolder();
        jdbc.update(con -> {
            var ps = con.prepareStatement(
                    "INSERT INTO " + TABLE + " (session_id, step_type, request_preview, response_preview, tool_calls_json, retrieval_docs_json, created_at) VALUES (?,?,?,?,?,?,?)",
                    new String[]{"id"});
            ps.setString(1, record.getSessionId());
            ps.setString(2, record.getStepType());
            ps.setString(3, record.getRequestPreview());
            ps.setString(4, record.getResponsePreview());
            ps.setString(5, record.getToolCallsJson());
            ps.setString(6, record.getRetrievalDocsJson());
            ps.setTimestamp(7, Timestamp.from(record.getCreatedAt()));
            return ps;
        }, keyHolder);
        Number key = keyHolder.getKey();
        if (key != null) record.setId(key.longValue());
        return record;
    }

    @Override
    public Optional<AgentTraceRecord> findById(Long id) {
        List<AgentTraceRecord> list = jdbc.query(
                "SELECT id, session_id, step_type, request_preview, response_preview, tool_calls_json, retrieval_docs_json, created_at FROM " + TABLE + " WHERE id = ?",
                (rs, i) -> toRecord(rs),
                id);
        return list.isEmpty() ? Optional.empty() : Optional.of(list.get(0));
    }

    @Override
    public List<AgentTraceRecord> findBySessionId(String sessionId, int limit) {
        return jdbc.query(
                "SELECT id, session_id, step_type, request_preview, response_preview, tool_calls_json, retrieval_docs_json, created_at FROM " + TABLE + " WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (rs, i) -> toRecord(rs),
                sessionId, Math.max(1, limit));
    }

    private static AgentTraceRecord toRecord(java.sql.ResultSet rs) throws java.sql.SQLException {
        AgentTraceRecord r = new AgentTraceRecord();
        r.setId(rs.getLong("id"));
        r.setSessionId(rs.getString("session_id"));
        r.setStepType(rs.getString("step_type"));
        r.setRequestPreview(rs.getString("request_preview"));
        r.setResponsePreview(rs.getString("response_preview"));
        r.setToolCallsJson(rs.getString("tool_calls_json"));
        r.setRetrievalDocsJson(rs.getString("retrieval_docs_json"));
        Timestamp ts = rs.getTimestamp("created_at");
        r.setCreatedAt(ts != null ? ts.toInstant() : null);
        return r;
    }
}
