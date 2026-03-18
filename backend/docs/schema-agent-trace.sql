-- Agent 审计 Trace 表：工具调用与检索结果落库，支持按 session 查询。
-- MySQL 示例（可按需在业务库执行）。
CREATE TABLE IF NOT EXISTS agent_trace (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    step_type VARCHAR(64) DEFAULT NULL,
    request_preview TEXT,
    response_preview TEXT,
    tool_calls_json TEXT,
    retrieval_docs_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session_id (session_id),
    INDEX idx_created_at (created_at)
);
