package com.inovationbehavior.backend.ai.reflect;

import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 内存存储反思规则，按 chatId 分桶，每桶保留最近 maxPerChat 条。
 */
@Component
@ConditionalOnMissingBean(ReflectiveRuleStore.class)
public class InMemoryReflectiveRuleStore implements ReflectiveRuleStore {

    private static final int MAX_PER_CHAT = 20;
    private final Map<String, List<String>> byChat = new ConcurrentHashMap<>();

    @Override
    public void addRule(String chatId, String ruleText) {
        if (chatId == null || ruleText == null || ruleText.isBlank()) return;
        byChat.compute(chatId, (k, list) -> {
            if (list == null) list = new ArrayList<>();
            list.add(0, ruleText.trim());
            if (list.size() > MAX_PER_CHAT) list = new ArrayList<>(list.subList(0, MAX_PER_CHAT));
            return list;
        });
    }

    @Override
    public List<String> getRecentRules(String chatId, int limit) {
        if (chatId == null) return List.of();
        List<String> list = byChat.get(chatId);
        if (list == null || list.isEmpty()) return List.of();
        int n = Math.min(limit, list.size());
        return new ArrayList<>(list.subList(0, n));
    }
}
