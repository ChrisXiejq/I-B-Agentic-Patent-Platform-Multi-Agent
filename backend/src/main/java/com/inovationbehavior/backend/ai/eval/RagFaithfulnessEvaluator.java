package com.inovationbehavior.backend.ai.eval;

import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.document.Document;

import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * RAG 忠实度评估：判断「回答是否完全基于给定上下文、无捏造」。
 * 使用 LLM-as-judge：给定 context + answer，让模型输出 0~1 或 yes/no，再解析为分数。
 */
public class RagFaithfulnessEvaluator {

    private static final String JUDGE_PROMPT = """
            You are an evaluator. Given the following RETRIEVED CONTEXT and the MODEL ANSWER, determine whether the answer is fully supported by the context (no hallucination or unsupported claims).
            Output ONLY a single number between 0 and 1: 1 = fully faithful (all claims supported by context), 0 = not faithful (contains hallucination or unsupported claims). No explanation.
            
            RETRIEVED CONTEXT:
            ---
            %s
            ---
            
            MODEL ANSWER:
            ---
            %s
            ---
            
            Score (0 or 1, or a decimal between 0 and 1):
            """;

    private final ChatModel chatModel;
    private static final Pattern NUMBER_PATTERN = Pattern.compile("0?\\.?\\d+");

    public RagFaithfulnessEvaluator(ChatModel chatModel) {
        this.chatModel = chatModel;
    }

    /**
     * 对单条 (context, answer) 打忠实度分
     *
     * @param contextText 检索到的上下文拼接成的字符串
     * @param answer      模型生成的回答
     * @return 0.0 ~ 1.0，解析失败时返回 -1
     */
    public double score(String contextText, String answer) {
        if (contextText == null || contextText.isBlank() || answer == null || answer.isBlank()) {
            return 0.0;
        }
        String prompt = JUDGE_PROMPT.formatted(
                contextText.length() > 8000 ? contextText.substring(0, 7997) + "..." : contextText,
                answer.length() > 4000 ? answer.substring(0, 3997) + "..." : answer
        );
        String out = chatModel.call(new Prompt(prompt)).getResult().getOutput().getText();
        return parseScore(out);
    }

    /**
     * 用检索到的 Document 列表拼成 context，再与 answer 一起打分
     */
    public double score(List<Document> contextDocs, String answer) {
        if (contextDocs == null || contextDocs.isEmpty()) return 0.0;
        String contextText = contextDocs.stream()
                .map(Document::getText)
                .filter(t -> t != null && !t.isBlank())
                .collect(Collectors.joining("\n\n"));
        return score(contextText, answer);
    }

    private static double parseScore(String text) {
        if (text == null || text.isBlank()) return -1;
        var m = NUMBER_PATTERN.matcher(text.trim());
        if (m.find()) {
            try {
                double v = Double.parseDouble(m.group());
                if (v >= 0 && v <= 1) return v;
                if (v > 1) return 1.0;
                return 0.0;
            } catch (NumberFormatException e) {
                if (text.toLowerCase().contains("yes") || text.toLowerCase().startsWith("1")) return 1.0;
                if (text.toLowerCase().contains("no") || text.toLowerCase().startsWith("0")) return 0.0;
            }
        }
        return -1;
    }
}
