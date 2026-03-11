package com.inovationbehavior.backend.ai.memory;

import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableAsync;

/**
 * 分层记忆架构配置（企业级）
 * - 短期记忆：滑动窗口 + 摘要压缩，importance 由领域 NER 加权并随轮次衰减
 * - 长期记忆：向量相似度检索 + importance 阈值筛选入库，NLI 冲突检测控制写入
 * L1 工作记忆仍由 MessageWindowChatMemory（IBApp）提供；此处注册分层记忆 Advisor。
 */
@Configuration
@EnableAsync
public class MultiLevelMemoryConfig {

    /** 分层记忆 Advisor（短期 + 长期） */
    @Bean("layeredMemoryAdvisor")
    @ConditionalOnBean(LongTermMemoryService.class)
    public Advisor layeredMemoryAdvisor(LongTermMemoryService longTermMemoryService,
                                       ImportanceScorer importanceScorer,
                                       @Autowired(required = false) ShortTermMemoryService shortTermMemoryService) {
        if (shortTermMemoryService != null) {
            return new LayeredMemoryAdvisor(longTermMemoryService, importanceScorer, shortTermMemoryService);
        }
        return new LayeredMemoryAdvisor(longTermMemoryService, importanceScorer);
    }
}
