package com.inovationbehavior.backend.ai.memory.tool;

import com.inovationbehavior.backend.ai.memory.working.WorkingMemoryService;
import com.inovationbehavior.backend.ai.memory.longterm.LongTermMemoryService;
import com.inovationbehavior.backend.ai.memory.experiential.ExperientialMemoryService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.document.Document;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * MCP 风格记忆检索工具：Agent 在需要时显式调用 retrieve_history，按需拉取当前会话的短期/中期/长期记忆，节省 Token 并减少干扰。
 * 由 {@link com.inovationbehavior.backend.ai.memory.config.MultiLevelMemoryConfig} 在 WorkingMemoryService 存在时注册为 Bean。
 */
@Slf4j
public class MemoryRetrievalTool {

    private final WorkingMemoryService workingMemoryService;
    private final LongTermMemoryService longTermMemoryService;
    private final ExperientialMemoryService experientialMemoryService;
    private final int longTermTopK;

    public MemoryRetrievalTool(WorkingMemoryService workingMemoryService,
                              LongTermMemoryService longTermMemoryService,
                              @Autowired(required = false) ExperientialMemoryService experientialMemoryService,
                              @Value("${app.memory.long-term.inject-top-k:3}") int longTermTopK) {
        this.workingMemoryService = workingMemoryService;
        this.longTermMemoryService = longTermMemoryService;
        this.experientialMemoryService = experientialMemoryService;
        this.longTermTopK = longTermTopK;
    }

    @Tool(
            name = "retrieve_history",
            description = "按需检索当前会话的历史记忆（短期上下文、中期事件摘要、长期事实）。在需要回忆用户之前说过什么、或需要基于历史做判断时调用。参数：conversation_id 为当前会话 ID，query 为本次检索意图或关键词（如「用户问过的专利号」「许可费用」）。"
    )
    public String retrieveHistory(
            @ToolParam(description = "当前会话 ID，与当前对话使用的 conversation_id 一致") String conversation_id,
            @ToolParam(description = "检索意图或关键词，用于语义召回相关记忆，例如「用户询问的专利」「许可意向」") String query) {
        if (conversation_id == null || conversation_id.isBlank()) {
            return "[retrieve_history] 缺少 conversation_id。";
        }
        String q = (query != null && !query.isBlank()) ? query : "最近对话与关键信息";

        StringBuilder out = new StringBuilder();

        String working = workingMemoryService.getContextForPrompt(conversation_id);
        if (!working.isBlank()) {
            out.append("[短期记忆（当前会话上下文）]\n").append(working).append("\n\n");
        }

        if (experientialMemoryService != null) {
            List<Document> expDocs = experientialMemoryService.retrieve(q, conversation_id);
            String expBlock = experientialMemoryService.formatMemoriesForPrompt(expDocs);
            if (!expBlock.isBlank()) out.append(expBlock).append("\n");
        }

        List<Document> longDocs = longTermMemoryService.retrieve(q, conversation_id);
        List<Document> reranked = rerankByScoreAndRecency(longDocs);
        List<Document> top = reranked.size() > longTermTopK ? reranked.subList(0, longTermTopK) : reranked;
        String longBlock = longTermMemoryService.formatMemoriesForPrompt(top);
        if (!longBlock.isBlank()) out.append(longBlock);

        if (out.length() == 0) {
            return "[retrieve_history] 当前会话暂无相关历史记忆。";
        }
        log.debug("retrieve_history called for conversation {} query {}", conversation_id, q);
        return out.toString().trim();
    }

    private List<Document> rerankByScoreAndRecency(List<Document> docs) {
        if (docs == null || docs.isEmpty()) return List.of();
        List<Document> list = new ArrayList<>(docs);
        list.sort(Comparator
                .comparingDouble(this::getScoreFromMetadata)
                .thenComparingLong(this::getCreatedAtFromMetadata)
                .reversed());
        return list;
    }

    private double getScoreFromMetadata(Document d) {
        Object v = d.getMetadata().get("importance");
        if (v == null) return 0.5;
        try {
            return Double.parseDouble(v.toString());
        } catch (NumberFormatException e) {
            return 0.5;
        }
    }

    private long getCreatedAtFromMetadata(Document d) {
        Object v = d.getMetadata().get("created_at");
        if (v == null) return 0L;
        try {
            return Long.parseLong(v.toString());
        } catch (NumberFormatException e) {
            return 0L;
        }
    }
}
