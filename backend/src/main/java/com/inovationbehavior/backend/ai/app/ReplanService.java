package com.inovationbehavior.backend.ai.app;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * 重规划逻辑：整计划重算、仅重规划剩余任务、以及结果不足/检索失败可恢复等判断。
 * 由 {@link com.inovationbehavior.backend.ai.graph.nodes.ReplanNode} 调用，与「应用入口」IBApp 解耦。
 */
@Component
@Slf4j
public class ReplanService {

    private static final String REPLAN_PROMPT = """
            You are the replanner. We already executed some steps but the last step had insufficient or empty result (e.g. no patent found, no RAG context). Output a new comma-separated list of remaining steps. Options: retrieval (try again, or suggest searchWeb), analysis, advice, synthesize.
            If retrieval failed due to missing context, you may output: retrieval,synthesize (retrieval will be prompted to use searchWeb). If we should give up and summarize: synthesize.
            Reply with only the comma-separated list.
            User message: %s
            Step results so far: %s
            """;
    private static final String REPLAN_REMAINING_PROMPT = """
            You are the replanner. We detected an environment change (e.g. patent invalid/expired, no data, or API/connection failure). The remaining planned steps were: %s.
            Output a new comma-separated list of remaining steps only. Options: retrieval, analysis, advice, synthesize.
            - If the last result shows API or connection failure (e.g. "Unable to connect to Redis", "Failed to query patent details", "Failed to query patent heat"), you MUST output: retrieval,synthesize (so we retry retrieval using web search to supplement).
            - If patent is invalid/expired, output: synthesize. Or: retrieval,synthesize if we should try searchWeb first.
            Reply with only the comma-separated list.
            User message: %s
            Step results so far: %s
            """;

    private final ChatClient chatClient;

    public ReplanService(ChatModel chatModel) {
        this.chatClient = ChatClient.builder(chatModel).build();
    }

    /**
     * 整计划重算：根据当前已执行结果重新规划（如检索无结果时改为 retrieval+synthesize 或直接 synthesize）。
     */
    public List<String> replan(String userMessage, List<String> stepResults) {
        if (userMessage == null) userMessage = "";
        if (stepResults == null || stepResults.isEmpty()) {
            log.info("[ReplanService][replan] 重规划 无已有结果，返回 synthesize");
            return List.of("synthesize");
        }
        String prior = String.join("\n---\n", stepResults);
        String prompt = REPLAN_PROMPT.formatted(userMessage, prior);
        ChatResponse resp = chatClient.prompt().user(prompt).call().chatResponse();
        String raw = resp.getResult().getOutput().getText();
        List<String> plan = parsePlan(raw);
        log.info("[ReplanService][replan] 重规划 stepResultsSize={} -> plan={}", stepResults.size(), plan);
        return plan.isEmpty() ? List.of("synthesize") : plan;
    }

    /**
     * 仅重规划「剩余任务」：用于环境变化时更新剩余列表，返回新剩余步骤（不含已执行部分）。
     * 若上一步为检索且结果不足（接口/连接失败，或 "I don't know"/无有效数据），强制返回 retrieval,synthesize 以便用 searchWeb 重试补足。
     */
    public List<String> replanRemaining(String userMessage, List<String> stepResults, List<String> remainingTasks) {
        if (userMessage == null) userMessage = "";
        if (stepResults == null) stepResults = List.of();
        String lastResult = stepResults.isEmpty() ? "" : stepResults.get(stepResults.size() - 1);
        if (shouldRetryRetrievalWithWeb(lastResult)) {
            log.info("[ReplanService][replanRemaining] 重规划剩余 检测到检索结果不足或接口失败，强制 retrieval,synthesize 以用 searchWeb 重试");
            return List.of("retrieval", "synthesize");
        }
        String prior = stepResults.isEmpty() ? "None" : String.join("\n---\n", stepResults);
        String remainingStr = (remainingTasks == null || remainingTasks.isEmpty()) ? "synthesize" : String.join(", ", remainingTasks);
        String prompt = REPLAN_REMAINING_PROMPT.formatted(remainingStr, userMessage, prior);
        ChatResponse resp = chatClient.prompt().user(prompt).call().chatResponse();
        String raw = resp.getResult().getOutput().getText();
        List<String> newRemaining = parsePlan(raw);
        log.info("[ReplanService] 重规划剩余 remaining={} -> newRemaining={}", remainingTasks, newRemaining);
        return newRemaining.isEmpty() ? List.of("synthesize") : newRemaining;
    }

    /** 上一步结果是否为「专利接口/连接失败」且可用 searchWeb 补足（用于强制重试 retrieval） */
    public static boolean isRetrievalFailureRecoverableWithWeb(String lastStepResult) {
        if (lastStepResult == null || lastStepResult.isBlank()) return false;
        String lower = lastStepResult.toLowerCase();
        return lower.contains("unable to connect to redis") || lower.contains("failed to query patent")
                || lower.contains("failed to query patent details") || lower.contains("failed to query patent heat")
                || (lower.contains("connection") && lower.contains("redis"));
    }

    /**
     * 是否应对检索步用 searchWeb 再试一次：接口/连接失败，或上一步为 retrieval 且结果不足（如 "I don't know"、无有效数据）。
     * 为 true 时 Replan 强制 retrieval,synthesize，且第二次 retrieval 会注入「请用 searchWeb 补足」提示。
     */
    public static boolean shouldRetryRetrievalWithWeb(String lastStepResult) {
        if (lastStepResult == null || lastStepResult.isBlank()) return false;
        if (isRetrievalFailureRecoverableWithWeb(lastStepResult)) return true;
        String lower = lastStepResult.toLowerCase();
        if (!lower.contains("task:retrieval") && !lower.contains("[task:retrieval]")) return false;
        return isResultInsufficient(lastStepResult);
    }

    /**
     * 判断专家输出是否“结果不足”，用于触发 Replan（如检索无结果、分析无法进行等）。
     */
    public static boolean isResultInsufficient(String expertOutput) {
        if (expertOutput == null || expertOutput.isBlank()) return true;
        String lower = expertOutput.trim().toLowerCase();
        if (lower.length() < 10) return true;
        if (lower.contains("i don't know") || lower.contains("i do not know") || lower.contains("don't know")) return true;
        if (lower.contains("无法") || lower.contains("没有找到") || lower.contains("暂无") || lower.contains("无相关")) return true;
        if (lower.contains("no patent") || lower.contains("no result") || lower.contains("insufficient")) return true;
        if (lower.contains("请提供") && lower.length() < 80) return true;
        return false;
    }

    private static List<String> parsePlan(String raw) {
        if (raw == null) return List.of();
        List<String> out = new java.util.ArrayList<>();
        for (String s : raw.trim().toLowerCase().split("[,，\\s]+")) {
            String t = s.trim();
            if (t.isEmpty()) continue;
            if (t.contains("retrieval")) out.add("retrieval");
            else if (t.contains("analysis")) out.add("analysis");
            else if (t.contains("advice")) out.add("advice");
            else if (t.contains("synthesize") || t.contains("end")) out.add("synthesize");
        }
        if (!out.isEmpty() && !"synthesize".equals(out.get(out.size() - 1))) {
            out.add("synthesize");
        }
        return out;
    }
}
