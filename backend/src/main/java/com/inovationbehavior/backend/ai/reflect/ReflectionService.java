package com.inovationbehavior.backend.ai.reflect;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * 反思与策略更新：将步骤成败归纳为简短规则写入 ReflectiveRuleStore，供规划与 prompt 注入。
 */
@Slf4j
@Service
@ConditionalOnBean(ReflectiveRuleStore.class)
public class ReflectionService {

    private static final String REFLECT_FAILURE_PROMPT = """
            You are summarizing a lesson from an agent step that had INSUFFICIENT or FAILED result.
            User question (preview): %s
            Step results (last part): %s
            Output ONE short sentence in Chinese or English as a reusable rule, e.g. "When patent API fails, retry with searchWeb." or "检索无结果时应先调用 searchWeb 再回答."
            Output only the rule, no prefix.
            """;
    private static final String REFLECT_SUCCESS_PROMPT = """
            You are summarizing a lesson from an agent step that SUCCEEDED.
            User question (preview): %s
            Step result (preview): %s
            Output ONE short sentence as a reusable rule for future similar tasks. Output only the rule, no prefix.
            """;

    private final ChatModel chatModel;
    private final ReflectiveRuleStore ruleStore;

    public ReflectionService(ChatModel chatModel, ReflectiveRuleStore ruleStore) {
        this.chatModel = chatModel;
        this.ruleStore = ruleStore;
    }

    /**
     * 根据上一步结果与成败进行反思，生成一条规则并写入 store。
     */
    public void reflect(String chatId, String userMessagePreview, List<String> stepResults, boolean success) {
        if (chatId == null || ruleStore == null) return;
        String lastResult = stepResults != null && !stepResults.isEmpty()
                ? stepResults.get(stepResults.size() - 1) : "";
        if (lastResult.length() > 800) lastResult = lastResult.substring(0, 800) + "...";
        String userPreview = userMessagePreview != null && userMessagePreview.length() > 200
                ? userMessagePreview.substring(0, 200) + "..." : (userMessagePreview != null ? userMessagePreview : "");

        String prompt = success
                ? REFLECT_SUCCESS_PROMPT.formatted(userPreview, lastResult)
                : REFLECT_FAILURE_PROMPT.formatted(userPreview, lastResult);
        try {
            ChatResponse resp = chatModel.call(new Prompt(prompt));
            String rule = resp.getResult() != null && resp.getResult().getOutput() != null
                    ? resp.getResult().getOutput().getText() : "";
            rule = rule.trim();
            if (!rule.isBlank()) {
                ruleStore.addRule(chatId, rule);
                log.info("[Reflect] chatId={} success={} -> rule: {}", chatId, success, rule);
            }
        } catch (Exception e) {
            log.warn("[Reflect] Failed to generate rule: {}", e.getMessage());
        }
    }

    /**
     * 获取该会话最近规则，用于注入规划或 task prompt。
     */
    public List<String> getRecentRules(String chatId, int limit) {
        return ruleStore != null ? ruleStore.getRecentRules(chatId, limit) : List.of();
    }
}
