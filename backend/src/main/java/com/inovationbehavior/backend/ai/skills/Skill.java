package com.inovationbehavior.backend.ai.skills;

import lombok.Builder;
import lombok.Value;

/**
 * 单个 Skill 的元数据与指令体（对应 Anthropic Agent Skills 的 SKILL.md）。
 * Level 1：name + description（启动时加载）；Level 2：instructions（触发时加载）。
 */
@Value
@Builder
public class Skill {

    String id;
    String name;
    String description;
    String instructions;
}
