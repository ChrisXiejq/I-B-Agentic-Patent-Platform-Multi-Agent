package com.inovationbehavior.backend.ai.memory.config;

import com.inovationbehavior.backend.ai.memory.working.WorkingMemoryService;
import com.inovationbehavior.backend.ai.memory.longterm.LongTermMemoryService;
import com.inovationbehavior.backend.ai.memory.experiential.ExperientialMemoryService;
import com.inovationbehavior.backend.ai.memory.importance.ImportanceScorer;
import com.inovationbehavior.backend.ai.memory.compression.SummaryCompressor;
import com.inovationbehavior.backend.ai.memory.advisor.MemoryPersistenceAdvisor;
import com.inovationbehavior.backend.ai.memory.tool.MemoryRetrievalTool;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableAsync;

/**
 * 三层分层记忆：仅持久化 Advisor + 按需检索由 MCP Tool retrieve_history 完成。
 * - WorkingMemoryService / MemoryRetrievalTool 仅当 SummaryCompressor 存在时注册，避免启动时找不到 Bean。
 * - MemoryPersistenceAdvisor：每轮结束后写入 Working / Experiential / Long-Term。
 * - MemoryRetrievalTool：Agent 显式调用 retrieve_history(conversation_id, query) 时再拉取记忆。
 */
@Configuration
@EnableAsync
public class MultiLevelMemoryConfig {

    @Bean
    @ConditionalOnBean(SummaryCompressor.class)
    public WorkingMemoryService workingMemoryService(ImportanceScorer importanceScorer,
                                                    SummaryCompressor summaryCompressor,
                                                    @Value("${app.memory.working.window-size:10}") int windowSize,
                                                    @Value("${app.memory.working.max-total-chars:8000}") int maxTotalChars,
                                                    @Value("${app.memory.working.prune-threshold:0.1}") double pruneThreshold,
                                                    @Value("${app.memory.working.turns-to-compress:5}") int turnsToCompressWhenFull) {
        return new WorkingMemoryService(importanceScorer, summaryCompressor,
                windowSize, maxTotalChars, pruneThreshold, turnsToCompressWhenFull);
    }

    @Bean
    @ConditionalOnBean(WorkingMemoryService.class)
    public MemoryRetrievalTool memoryRetrievalTool(WorkingMemoryService workingMemoryService,
                                                   LongTermMemoryService longTermMemoryService,
                                                   @Autowired(required = false) ExperientialMemoryService experientialMemoryService,
                                                   @Value("${app.memory.long-term.inject-top-k:3}") int longTermInjectTopK) {
        return new MemoryRetrievalTool(workingMemoryService, longTermMemoryService,
                experientialMemoryService, longTermInjectTopK);
    }

    @Bean("memoryPersistenceAdvisor")
    @ConditionalOnBean({WorkingMemoryService.class, LongTermMemoryService.class})
    public Advisor memoryPersistenceAdvisor(WorkingMemoryService workingMemoryService,
                                            LongTermMemoryService longTermMemoryService,
                                            ImportanceScorer importanceScorer,
                                            @Autowired(required = false) ExperientialMemoryService experientialMemoryService) {
        if (experientialMemoryService != null) {
            return new MemoryPersistenceAdvisor(workingMemoryService, longTermMemoryService,
                    importanceScorer, experientialMemoryService);
        }
        return new MemoryPersistenceAdvisor(workingMemoryService, longTermMemoryService, importanceScorer);
    }
}
