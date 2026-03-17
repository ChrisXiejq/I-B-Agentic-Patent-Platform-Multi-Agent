package com.inovationbehavior.backend.ai.graph.nodes;

import com.inovationbehavior.backend.ai.app.ReplanService;
import com.inovationbehavior.backend.ai.graph.PatentGraphState;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * P&E 重规划节点：根据当前 stepResults 与「剩余任务」更新计划。
 * - 若为环境变化触发：只更新剩余任务列表（replanRemaining），保留已执行部分，currentStepIndex 不变。
 * - 若为结果不足触发：整计划重算（replan），currentStepIndex=0。
 */
@Component
@Slf4j
public class ReplanNode {

    private final ReplanService replanService;

    public ReplanNode(ReplanService replanService) {
        this.replanService = replanService;
    }

    private static final String LOG_PREFIX = "[AgentGraph.Replan] ";

    public Map<String, Object> apply(PatentGraphState state) {
        String userMessage = state.userMessage().orElse("");
        List<String> stepResults = state.stepResults();
        List<String> plan = state.plan();
        int currentStepIndex = state.currentStepIndex();
        boolean envChanged = state.environmentChanged();

        log.info("{}>>> 进入节点 | chatId={} stepResultsSize={} environmentChanged={} currentStepIndex={}",
                LOG_PREFIX, state.chatId().orElse("default"), stepResults != null ? stepResults.size() : 0, envChanged, currentStepIndex);

        List<String> newPlan;
        int newStepIndex;
        if (envChanged && plan != null && currentStepIndex > 0 && currentStepIndex <= plan.size()) {
            List<String> executed = new ArrayList<>(plan.subList(0, currentStepIndex));
            List<String> remaining = plan.subList(currentStepIndex, plan.size());
            List<String> newRemaining = replanService.replanRemaining(userMessage, stepResults, remaining);
            newPlan = new ArrayList<>(executed);
            newPlan.addAll(newRemaining);
            newStepIndex = currentStepIndex;
            log.info("{} 环境变化：仅更新剩余任务 executed={} newRemaining={}", LOG_PREFIX, executed, newRemaining);
        } else {
            newPlan = new ArrayList<>(replanService.replan(userMessage, stepResults));
            newStepIndex = 0;
            log.info("{} 结果不足：整计划重算 newPlan={}", LOG_PREFIX, newPlan);
        }

        log.info("{}<<< 离开节点 | newPlan={} newStepIndex={}", LOG_PREFIX, newPlan, newStepIndex);
        return Map.of(
                PatentGraphState.PLAN, newPlan,
                PatentGraphState.CURRENT_STEP_INDEX, newStepIndex,
                PatentGraphState.NEED_REPLAN, 0,
                PatentGraphState.ENVIRONMENT_CHANGED, 0
        );
    }
}
