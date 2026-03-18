package com.inovationbehavior.backend.ai.config;

import com.inovationbehavior.backend.ai.advisor.AgentTraceAdvisor;
import com.inovationbehavior.backend.ai.advisor.PersistingTraceAdvisor;
import com.inovationbehavior.backend.ai.audit.AgentTraceRepository;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Agent 可观测配置：{@link AgentTraceAdvisor} 打日志；{@link PersistingTraceAdvisor} 将请求/响应/工具调用/检索结果落库，按 session 查询。
 */
@Configuration
public class AgentTraceConfig {

    @Bean
    public Advisor agentTraceAdvisor() {
        return new AgentTraceAdvisor();
    }

    @Bean
    @ConditionalOnBean(AgentTraceRepository.class)
    public Advisor persistingTraceAdvisor(AgentTraceRepository repository) {
        return new PersistingTraceAdvisor(repository);
    }
}
