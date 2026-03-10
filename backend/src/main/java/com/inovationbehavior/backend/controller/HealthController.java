package com.inovationbehavior.backend.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/health")
public class HealthController {

    @Autowired(required = false)
    @Qualifier("pgvectorJdbcTemplate")
    private JdbcTemplate pgvectorJdbcTemplate;

    @GetMapping
    public String healthCheck() {
        return "ok";
    }

    /**
     * 测试 pgvector 连通性。需配置 spring.ai.vectorstore.pgvector.url 后生效。
     */
    @GetMapping("/pgvector")
    public Map<String, Object> pgvectorCheck() {
        if (pgvectorJdbcTemplate == null) {
            return Map.of(
                    "connected", false,
                    "message", "pgvector not configured. Set spring.ai.vectorstore.pgvector.url to enable."
            );
        }
        try {
            Long count = pgvectorJdbcTemplate.queryForObject(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'vector_store'",
                    Long.class);
            boolean tableExists = count != null && count > 0;
            return Map.of(
                    "connected", true,
                    "vector_store_table_exists", tableExists,
                    "message", tableExists ? "pgvector connected, vector_store table ready" : "pgvector connected, vector_store table will be created on first use"
            );
        } catch (Exception e) {
            return Map.of(
                    "connected", false,
                    "error", e.getMessage(),
                    "message", "pgvector connection failed"
            );
        }
    }
}
