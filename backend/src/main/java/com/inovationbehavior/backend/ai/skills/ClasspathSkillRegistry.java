package com.inovationbehavior.backend.ai.skills;

import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.ResourcePatternResolver;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 从 classpath 或指定路径扫描 skills/*/SKILL.md，实现渐进式披露：启动时解析元数据，按需返回完整指令。
 */
@Slf4j
public class ClasspathSkillRegistry implements SkillRegistry {

    private final String skillsLocation;
    private final ResourcePatternResolver resourcePatternResolver;
    private final Map<String, Skill> skillsById = new ConcurrentHashMap<>();

    public ClasspathSkillRegistry(String skillsLocation, ResourcePatternResolver resourcePatternResolver) {
        this.skillsLocation = skillsLocation != null && !skillsLocation.isBlank()
                ? skillsLocation : "classpath:skills/*/SKILL.md";
        this.resourcePatternResolver = resourcePatternResolver;
        loadAll();
    }

    private void loadAll() {
        try {
            Resource[] resources = resourcePatternResolver.getResources(skillsLocation);
            for (Resource res : resources) {
                String path = res.getURL().toString();
                String skillId = extractSkillId(path);
                if (skillId == null) continue;
                String content = SkillLoader.readResourceUtf8(res);
                if (content == null) continue;
                Skill skill = SkillLoader.parseSkillMd(skillId, content);
                if (skill != null) {
                    skillsById.put(skillId.toLowerCase(Locale.ROOT), skill);
                    log.info("Loaded skill: id={} name={}", skillId, skill.getName());
                }
            }
        } catch (Exception e) {
            log.warn("Could not load skills from {}: {}", skillsLocation, e.getMessage());
        }
    }

    private static String extractSkillId(String url) {
        // .../skills/retrieval/SKILL.md or ...\skills\retrieval\SKILL.md -> retrieval
        String normalized = url.replace('\\', '/');
        int i = normalized.indexOf("/skills/");
        if (i < 0) return null;
        int start = i + "/skills/".length();
        int end = normalized.indexOf("/", start);
        if (end < 0) end = normalized.length();
        String segment = normalized.substring(start, end);
        return segment.isBlank() ? null : segment;
    }

    @Override
    public List<SkillMetadata> listSkills() {
        return skillsById.values().stream()
                .map(s -> new SkillMetadata(s.getId(), s.getName(), s.getDescription()))
                .sorted(Comparator.comparing(SkillMetadata::id))
                .toList();
    }

    @Override
    public Optional<String> getInstructions(String skillId) {
        if (skillId == null || skillId.isBlank()) return Optional.empty();
        Skill skill = skillsById.get(skillId.trim().toLowerCase(Locale.ROOT));
        if (skill == null) return Optional.empty();
        String instructions = skill.getInstructions();
        return instructions != null && !instructions.isBlank() ? Optional.of(instructions) : Optional.empty();
    }
}
