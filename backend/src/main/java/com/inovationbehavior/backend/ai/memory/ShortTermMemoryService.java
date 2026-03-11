package com.inovationbehavior.backend.ai.memory;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 短期记忆：滑动窗口 + 摘要压缩，importance 随轮次衰减参与上下文管理。
 * 窗口满时将最旧若干轮压缩为 runningSummary，仅保留最近 windowSize 轮完整内容。
 */
@Slf4j
@Service
@ConditionalOnBean(SummaryCompressor.class)
public class ShortTermMemoryService {

    private final ImportanceScorer importanceScorer;
    private final SummaryCompressor summaryCompressor;
    private final int windowSize;
    private final int turnsToCompressWhenFull;
    private final double decayFactorPerTurn;

    /** conversationId -> (runningSummary, recentTurns) */
    private final ConcurrentHashMap<String, ConversationShortTermState> stateByConversation = new ConcurrentHashMap<>();

    public ShortTermMemoryService(ImportanceScorer importanceScorer,
                                  SummaryCompressor summaryCompressor,
                                  @Value("${app.memory.short-term.window-size:10}") int windowSize,
                                  @Value("${app.memory.short-term.turns-to-compress:5}") int turnsToCompressWhenFull,
                                  @Value("${app.memory.short-term.decay-factor-per-turn:0.92}") double decayFactorPerTurn) {
        this.importanceScorer = importanceScorer;
        this.summaryCompressor = summaryCompressor;
        this.windowSize = windowSize;
        this.turnsToCompressWhenFull = turnsToCompressWhenFull;
        this.decayFactorPerTurn = decayFactorPerTurn;
    }

    /**
     * 追加一轮对话，并可能触发滑动窗口压缩。
     */
    public void addTurn(String conversationId, String userMessage, String assistantMessage) {
        if (conversationId == null || userMessage == null) return;

        ConversationShortTermState state = stateByConversation.computeIfAbsent(
                conversationId, k -> new ConversationShortTermState());

        int turnIndex = state.nextTurnIndex();
        double importance = importanceScorer.score(userMessage, assistantMessage, turnIndex);

        MemoryTurnRecord record = MemoryTurnRecord.builder()
                .userMessage(userMessage)
                .assistantMessage(assistantMessage)
                .turnIndex(turnIndex)
                .importance(importance)
                .createdAtMillis(System.currentTimeMillis())
                .build();

        synchronized (state) {
            state.recentTurns.add(record);
            while (state.recentTurns.size() > windowSize) {
                int excess = state.recentTurns.size() - windowSize;
                int toCompress = Math.min(turnsToCompressWhenFull, Math.max(1, excess));
                List<MemoryTurnRecord> toMerge = new ArrayList<>(state.recentTurns.subList(0, toCompress));
                for (int i = 0; i < toCompress; i++) state.recentTurns.remove(0);
                String newSummary = summaryCompressor.compress(state.runningSummary, toMerge);
                state.runningSummary = newSummary != null ? newSummary : state.runningSummary;
            }
        }
        log.debug("Short-term memory: conversation {} turn {} importance {}", conversationId, turnIndex, importance);
    }

    /**
     * 获取注入 Prompt 的短期上下文：运行摘要 + 最近若干轮（按衰减后重要性可排序，此处简化为按时间顺序）。
     */
    public String getContextForPrompt(String conversationId) {
        ConversationShortTermState state = stateByConversation.get(conversationId);
        if (state == null) return "";

        StringBuilder sb = new StringBuilder();
        synchronized (state) {
            if (state.runningSummary != null && !state.runningSummary.isBlank()) {
                sb.append("\n[此前对话摘要]\n").append(state.runningSummary).append("\n");
            }
            if (!state.recentTurns.isEmpty()) {
                sb.append("\n[最近对话]\n");
                int currentTurn = state.recentTurns.stream().mapToInt(MemoryTurnRecord::getTurnIndex).max().orElse(0);
                List<String> lines = state.recentTurns.stream()
                        .map(r -> {
                            double decayed = r.getImportance() * Math.pow(decayFactorPerTurn, currentTurn - r.getTurnIndex());
                            return "  - " + r.toCompactText().replace("\n", " ");
                        })
                        .collect(Collectors.toList());
                sb.append(String.join("\n", lines));
            }
        }
        return sb.length() > 0 ? sb.toString() : "";
    }

    /** 当前轮次（用于长期记忆写入时携带 turnIndex）。 */
    public int getCurrentTurnIndex(String conversationId) {
        ConversationShortTermState state = stateByConversation.get(conversationId);
        if (state == null) return 0;
        return state.nextTurnIndexValue;
    }

    /** 获取最近一轮的 importance（供长期记忆写入阈值判断）。 */
    public double getLatestImportance(String conversationId) {
        ConversationShortTermState state = stateByConversation.get(conversationId);
        if (state == null || state.recentTurns.isEmpty()) return 0.0;
        return state.recentTurns.get(state.recentTurns.size() - 1).getImportance();
    }

    public void clear(String conversationId) {
        stateByConversation.remove(conversationId);
    }

    private static class ConversationShortTermState {
        String runningSummary = "";
        final List<MemoryTurnRecord> recentTurns = new ArrayList<>();
        int nextTurnIndexValue = 0;

        int nextTurnIndex() {
            return nextTurnIndexValue++;
        }
    }
}
