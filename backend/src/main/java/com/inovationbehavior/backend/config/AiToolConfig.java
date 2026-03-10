package com.inovationbehavior.backend.config;

import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.ToolCallbackProvider;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * 提供默认的 ToolCallback / ToolCallbackProvider Bean，在未启用 MCP 或自定义工具时保证正常启动。
 * 启用 MCP 后会自动注册 ToolCallbackProvider，本处通过 @ConditionalOnMissingBean 避免覆盖。
 */
@Configuration
public class AiToolConfig {

    @Bean(name = "allTools")
    @ConditionalOnMissingBean(name = "allTools")
    public ToolCallback[] allTools() {
        return new ToolCallback[0];
    }

    @Bean
    @ConditionalOnMissingBean(ToolCallbackProvider.class)
    public ToolCallbackProvider toolCallbackProvider(ToolCallback[] allTools) {
        return () -> allTools;
    }
}
