package com.inovationbehavior.backend.ai.graph.nodes;

import com.inovationbehavior.backend.ai.app.IBApp;
import com.inovationbehavior.backend.ai.app.ReplanService;
import com.inovationbehavior.backend.ai.graph.PatentGraphState;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Lazy;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

/**
 * P&E 执行器：针对当前子任务（plan[currentStepIndex]）启动一次 ReAct 循环（RAG + 工具调用），
 * 写回 stepResults、stepCount+1、currentStepIndex+1、needReplan。
 */
@Component
@Slf4j
public class ExecutorNode {

    private final IBApp ibApp;

    public ExecutorNode(@Lazy IBApp ibApp) {
        this.ibApp = ibApp;
    }

    private static final String LOG_PREFIX = "[AgentGraph.Executor] ";

    public Map<String, Object> apply(PatentGraphState state) {
        List<String> plan = state.plan();
        int idx = state.currentStepIndex();
        String task = (plan != null && idx >= 0 && idx < plan.size()) ? plan.get(idx) : "retrieval";
        String message = state.userMessage().orElse("");
        String chatId = state.chatId().orElse("default");
        List<String> stepResults = state.stepResults();

        log.info("{}>>> 进入节点 | task={} chatId={} userMessage(preview)={}",
                LOG_PREFIX, task, chatId, abbreviate(message, 80));
        long start = System.currentTimeMillis();
        String out = ibApp.doReActForTask(task, message, chatId, stepResults);
        long elapsed = System.currentTimeMillis() - start;

        int nextCount = state.stepCount() + 1;
        int nextStepIndex = idx + 1;
        int needReplan = ReplanService.isResultInsufficient(out) ? 1 : 0;

        log.info("{}<<< 离开节点 | task={} 输出长度={} stepCount->{} currentStepIndex->{} needReplan={} elapsedMs={}",
                LOG_PREFIX, task, out != null ? out.length() : 0, nextCount, nextStepIndex, needReplan, elapsed);

        return Map.of(
                PatentGraphState.STEP_RESULTS, "[Task:" + task + "]\n" + (out != null ? out : ""),
                PatentGraphState.STEP_COUNT, nextCount,
                PatentGraphState.CURRENT_STEP_INDEX, nextStepIndex,
                PatentGraphState.NEED_REPLAN, needReplan
        );
    }

    private static String abbreviate(String s, int maxLen) {
        if (s == null) return "null";
        s = s.trim();
        if (s.length() <= maxLen) return s;
        return s.substring(0, maxLen) + "...";
    }
}
