package com.inovationbehavior.backend.config;

import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.jdbc.core.JdbcTemplate;

import javax.sql.DataSource;

/**
 * PgVector 专用 PostgreSQL 数据源配置。
 * 当设置 spring.ai.vectorstore.pgvector.url 时生效，创建 pgvectorJdbcTemplate 供 PgVectorVectorStoreConfig 使用。
 */
@Configuration
@Slf4j
@Order(Ordered.HIGHEST_PRECEDENCE)
@ConditionalOnProperty(name = "spring.ai.vectorstore.pgvector.url")
public class PgVectorDataSourceConfig {

    @Bean(name = "pgvectorDataSource")
    public DataSource pgvectorDataSource(
            @Value("${spring.ai.vectorstore.pgvector.url}") String url,
            @Value("${spring.ai.vectorstore.pgvector.username:postgres}") String username,
            @Value("${spring.ai.vectorstore.pgvector.password:}") String password) {
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl(url);
        config.setUsername(username);
        config.setPassword(password);
        config.setDriverClassName("org.postgresql.Driver");
        config.setPoolName("pgvector-pool");
        log.info("PgVector datasource configured: {}", url.replaceAll(":[^:@]+@", ":****@"));
        return new HikariDataSource(config);
    }

    @Bean(name = "pgvectorJdbcTemplate")
    public JdbcTemplate pgvectorJdbcTemplate(@Qualifier("pgvectorDataSource") DataSource dataSource) {
        return new JdbcTemplate(dataSource);
    }
}
