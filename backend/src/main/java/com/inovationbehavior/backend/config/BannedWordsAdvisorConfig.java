package com.inovationbehavior.backend.config;

import com.inovationbehavior.backend.ai.advisor.BannedWordsAdvisor;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.Set;

/**
 * 违禁词校验 Advisor 配置：从 app.ai.banned-words 读取词表并注册 BannedWordsAdvisor。
 */
@Configuration
@EnableConfigurationProperties(AppAiProperties.class)
public class BannedWordsAdvisorConfig {

    @Bean
    public BannedWordsAdvisor bannedWordsAdvisor(AppAiProperties appAi) {
        Set<String> words = appAi.getBannedWords() == null
                ? Set.of()
                : Set.copyOf(appAi.getBannedWords());
        return new BannedWordsAdvisor(words, appAi.getBannedWordsRejectMessage());
    }
}
