package com.inovationbehavior.backend.ai.skills;

import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.ResourcePatternResolver;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Pattern;

/**
 * 从文件系统/classpath 解析 SKILL.md（YAML frontmatter + 正文），符合 Anthropic Agent Skills 约定。
 */
@Slf4j
public class SkillLoader {

    private static final Pattern FRONTMATTER_DELIM = Pattern.compile("^---\\s*$", Pattern.MULTILINE);

    /**
     * 解析 SKILL.md 内容：提取 name、description（frontmatter）与 instructions（正文）。
     */
    public static Skill parseSkillMd(String skillId, String content) {
        if (content == null || content.isBlank()) return null;
        String trimmed = content.trim();
        if (!trimmed.startsWith("---")) {
            return Skill.builder().id(skillId).name(skillId).description("").instructions(trimmed).build();
        }
        String[] parts = FRONTMATTER_DELIM.split(trimmed, 3);
        String frontmatter = parts.length > 1 ? parts[1].trim() : "";
        String body = parts.length > 2 ? parts[2].trim() : "";

        Map<String, String> meta = parseFrontmatter(frontmatter);
        String name = meta.getOrDefault("name", skillId);
        String description = meta.getOrDefault("description", "");

        return Skill.builder()
                .id(skillId)
                .name(name)
                .description(description)
                .instructions(body.isBlank() ? "" : body)
                .build();
    }

    private static Map<String, String> parseFrontmatter(String yaml) {
        Map<String, String> out = new HashMap<>();
        for (String line : yaml.split("\n")) {
            int colon = line.indexOf(':');
            if (colon <= 0) continue;
            String key = line.substring(0, colon).trim();
            String value = line.substring(colon + 1).trim();
            if (value.startsWith("'") && value.endsWith("'") || value.startsWith("\"") && value.endsWith("\"")) {
                value = value.substring(1, value.length() - 1);
            }
            out.put(key, value);
        }
        return out;
    }

    public static String readResourceUtf8(Resource resource) {
        try (InputStream is = resource.getInputStream()) {
            return new String(is.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            log.warn("Failed to read skill resource {}: {}", resource, e.getMessage());
            return null;
        }
    }
}
