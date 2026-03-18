package com.inovationbehavior.backend.ai.skills;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.support.ResourcePatternResolver;

/**
 * Agent Skills 注册表配置：从 classpath:skills/*/SKILL.md 加载技能（Anthropic Agent Skills 风格）。
 */
@Configuration
public class SkillRegistryConfig {

    @Value("${app.skills.location:classpath:skills/*/SKILL.md}")
    private String skillsLocation;

    @Bean
    public SkillRegistry skillRegistry(ResourcePatternResolver resourcePatternResolver) {
        return new ClasspathSkillRegistry(skillsLocation, resourcePatternResolver);
    }
}
