package com.inovationbehavior.backend.ai.tools;

import com.inovationbehavior.backend.ai.memory.tool.MemoryRetrievalTool;
import org.springframework.ai.support.ToolCallbacks;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 集中的工具注册类（含专利平台工具、记忆检索 MCP 工具 retrieve_history）
 */
@Configuration
public class ToolRegistration {

    @Value("${search-api.api-key:}")
    private String searchApiKey;

    @Bean
    public ToolCallback[] allTools(
            PatentDetailTool patentDetailTool,
            PatentHeatTool patentHeatTool,
            UserIdentityTool userIdentityTool,
            @Autowired(required = false) MemoryRetrievalTool memoryRetrievalTool) {
        List<Object> toolBeans = new ArrayList<>(Arrays.asList(
                new FileOperationTool(),
                new WebSearchTool(searchApiKey),
                new WebScrapingTool(),
                new ResourceDownloadTool(),
                new TerminalOperationTool(),
                new PDFGenerationTool(),
                new TerminateTool(),
                patentDetailTool,
                patentHeatTool,
                userIdentityTool
        ));
        if (memoryRetrievalTool != null) {
            toolBeans.add(memoryRetrievalTool);
        }
        return ToolCallbacks.from(toolBeans.toArray(new Object[0]));
    }
}
