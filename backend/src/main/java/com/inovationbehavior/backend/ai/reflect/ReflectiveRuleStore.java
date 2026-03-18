package com.inovationbehavior.backend.ai.reflect;

import java.util.List;

/**
 * 存储反思产生的策略规则，供规划与 prompt 注入。
 */
public interface ReflectiveRuleStore {

    void addRule(String chatId, String ruleText);

    /** 按会话取最近若干条规则，用于注入 plan / task prompt */
    List<String> getRecentRules(String chatId, int limit);
}
