package com.inovationbehavior.backend.ai.skills;

import java.util.List;
import java.util.Optional;

/**
 * Skill 注册表：按 id 解析并返回 Skill 的元数据或完整指令（渐进式披露）。
 */
public interface SkillRegistry {

    /**
     * 所有已加载 Skill 的 Level 1 元数据（name + description），用于 Planner 感知可用技能。
     */
    List<SkillMetadata> listSkills();

    /**
     * 按任务/技能 id（如 retrieval、analysis、advice）获取完整指令体（Level 2）。
     * 若该 id 无对应 SKILL.md 或加载失败，返回 empty。
     */
    Optional<String> getInstructions(String skillId);

    /** Level 1 元数据：仅 name + description */
    record SkillMetadata(String id, String name, String description) {}
}
