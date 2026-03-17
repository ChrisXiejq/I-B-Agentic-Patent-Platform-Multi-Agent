package com.inovationbehavior.backend.ai.memory.working;

import com.inovationbehavior.backend.ai.memory.importance.ImportanceScorer;
import com.inovationbehavior.backend.ai.memory.compression.SummaryCompressor;
import com.inovationbehavior.backend.ai.memory.model.MemoryTurnRecord;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 第一层：Working Memory（短期）— 参考 Coze 运行时状态。
 * 内存存储、滑动窗口、动态剪枝（importance &lt; 阈值剔除冗余句），溢出时产出摘要供 Layer2 写入。
 */
@Slf4j
public class WorkingMemoryService {

    private final ImportanceScorer importanceScorer;
    private final SummaryCompressor summaryCompressor;
    private final int windowSize;
    private final int maxTotalChars;
    private final double pruneThreshold;
    private final int turnsToCompressWhenFull;

    private final ConcurrentHashMap<String, WorkingMemoryState> stateByConversation = new ConcurrentHashMap<>();

    public WorkingMemoryService(ImportanceScorer importanceScorer,
                                SummaryCompressor summaryCompressor,
                                @Value("${app.memory.working.window-size:10}") int windowSize,
                                @Value("${app.memory.working.max-total-chars:8000}") int maxTotalChars,
                                @Value("${app.memory.working.prune-threshold:0.1}") double pruneThreshold,
                                @Value("${app.memory.working.turns-to-compress:5}") int turnsToCompressWhenFull) {
        this.importanceScorer = importanceScorer;
        this.summaryCompressor = summaryCompressor;
        this.windowSize = windowSize;
        this.maxTotalChars = maxTotalChars;
        this.pruneThreshold = pruneThreshold;
        this.turnsToCompressWhenFull = turnsToCompressWhenFull;
    }

    /**
     * 追加一轮对话。当接近 token/字符临界时按 importance 剪枝；窗口满时压缩最旧若干轮并产出摘要供 Layer2 写入。
     *
     * @return 若发生压缩，返回本次压缩得到的摘要文本，供 ExperientialMemory 写入；否则 empty
     */
    public Optional<String> addTurn(String conversationId, String userMessage, String assistantMessage) {
        if (conversationId == null || userMessage == null) return Optional.empty();

        WorkingMemoryState state = stateByConversation.computeIfAbsent(conversationId, k -> new WorkingMemoryState());
        int turnIndex = state.nextTurnIndex();
        double importance = importanceScorer.score(userMessage, assistantMessage, turnIndex);

        MemoryTurnRecord record = MemoryTurnRecord.builder()
                .userMessage(userMessage)
                .assistantMessage(assistantMessage)
                .turnIndex(turnIndex)
                .importance(importance)
                .createdAtMillis(System.currentTimeMillis())
                .build();

        String compressedForExperiential = null;
        synchronized (state) {
            state.recentTurns.add(record);

            // 动态剪枝：总字符接近临界时，剔除 importance < pruneThreshold 的轮次
            pruneLowImportanceTurns(state);

            // 滑动窗口溢出：压缩最旧若干轮到 runningSummary，并产出摘要供 Layer2
            while (state.recentTurns.size() > windowSize) {
                int toCompress = Math.min(turnsToCompressWhenFull, Math.max(1, state.recentTurns.size() - windowSize));
                List<MemoryTurnRecord> toMerge = new ArrayList<>(state.recentTurns.subList(0, toCompress));
                for (int i = 0; i < toCompress; i++) state.recentTurns.remove(0);
                String newSegment = summaryCompressor.compress(state.runningSummary, toMerge);
                state.runningSummary = (state.runningSummary != null && !state.runningSummary.isBlank())
                        ? state.runningSummary + "\n" + (newSegment != null ? newSegment : "")
                        : (newSegment != null ? newSegment : "");
                compressedForExperiential = newSegment != null ? newSegment : compressedForExperiential;
            }
        }
        log.debug("Working memory: conversation {} turn {} importance {}", conversationId, turnIndex, importance);
        return Optional.ofNullable(compressedForExperiential);
    }

    private void pruneLowImportanceTurns(WorkingMemoryState state) {
        int totalChars = state.recentTurns.stream()
                .mapToInt(r -> (r.getUserMessage() != null ? r.getUserMessage().length() : 0)
                        + (r.getAssistantMessage() != null ? r.getAssistantMessage().length() : 0))
                .sum();
        if (totalChars <= maxTotalChars) return;
        state.recentTurns.removeIf(r -> r.getImportance() < pruneThreshold);
    }

    public String getContextForPrompt(String conversationId) {
        WorkingMemoryState state = stateByConversation.get(conversationId);
        if (state == null) return "";

        StringBuilder sb = new StringBuilder();
        synchronized (state) {
            if (state.runningSummary != null && !state.runningSummary.isBlank()) {
                sb.append("\n[此前对话摘要]\n").append(state.runningSummary).append("\n");
            }
            if (!state.recentTurns.isEmpty()) {
                sb.append("\n[最近对话]\n");
                List<String> lines = state.recentTurns.stream()
                        .map(r -> "  - " + r.toCompactText().replace("\n", " "))
                        .collect(Collectors.toList());
                sb.append(String.join("\n", lines));
            }
        }
        return sb.toString().trim();
    }

    public int getCurrentTurnIndex(String conversationId) {
        WorkingMemoryState state = stateByConversation.get(conversationId);
        return state == null ? 0 : state.nextTurnIndexValue;
    }

    public double getLatestImportance(String conversationId) {
        WorkingMemoryState state = stateByConversation.get(conversationId);
        if (state == null || state.recentTurns.isEmpty()) return 0.0;
        return state.recentTurns.get(state.recentTurns.size() - 1).getImportance();
    }

    public void clear(String conversationId) {
        stateByConversation.remove(conversationId);
    }

    private static class WorkingMemoryState {
        String runningSummary = "";
        final List<MemoryTurnRecord> recentTurns = new ArrayList<>();
        int nextTurnIndexValue = 0;

        int nextTurnIndex() {
            return nextTurnIndexValue++;
        }
    }
}
