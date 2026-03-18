package com.inovationbehavior.backend.ai.graph.nodes;

import com.inovationbehavior.backend.ai.app.IBApp;
import com.inovationbehavior.backend.ai.graph.PatentGraphState;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Lazy;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * P&E 规划节点：根据用户问题生成执行计划，写入 plan、currentStepIndex=0、needReplan=0。
 */
@Component
@Slf4j
public class PlannerNode {

    private final IBApp ibApp;

    public PlannerNode(@Lazy IBApp ibApp) {
        this.ibApp = ibApp;
    }

    private static final String LOG_PREFIX = "[AgentGraph.Planner] ";

    public Map<String, Object> apply(PatentGraphState state) {
        String userMessage = state.userMessage().orElse("");
        List<String> stepResults = state.stepResults();
        log.info("{}>>> 进入节点 | chatId={} userMessage(preview)={}",
                LOG_PREFIX, state.chatId().orElse("default"), abbreviate(userMessage, 80));
        List<String> plan = ibApp.createPlan(userMessage, stepResults, state.chatId().orElse(""));
        List<String> planCopy = new ArrayList<>(plan);
        log.info("{}<<< 离开节点 | plan={}", LOG_PREFIX, planCopy);
        return Map.of(
                PatentGraphState.PLAN, planCopy,
                PatentGraphState.CURRENT_STEP_INDEX, 0,
                PatentGraphState.NEED_REPLAN, 0
        );
    }

    private static String abbreviate(String s, int maxLen) {
        if (s == null) return "null";
        s = s.trim();
        if (s.length() <= maxLen) return s;
        return s.substring(0, maxLen) + "...";
    }
}
