package com.inovationbehavior.backend.ai.eval;

import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.Prompt;

import java.util.regex.Pattern;

/**
 * RAGAS 风格评估器：Faithfulness、Answer Relevancy、Context Precision、Context Recall。
 * 使用 LLM-as-judge，输出 0~1 标量；与 Python RAGAS 定义对齐，便于自动化评估与回归。
 */
public class RagasEvaluator {

    private static final int MAX_CTX_LEN = 6000;
    private static final int MAX_ANSWER_LEN = 2000;
    private static final Pattern NUMBER_PATTERN = Pattern.compile("0?\\.?\\d+");

    private final ChatModel chatModel;

    public RagasEvaluator(ChatModel chatModel) {
        this.chatModel = chatModel;
    }

    /**
     * 对单条样本计算 RAGAS 四维分数
     *
     * @param question         用户问题
     * @param context          检索到的上下文（拼接文本）
     * @param answer           模型生成的回答
     * @param referenceAnswer  参考答案（可选；无则 contextRecall 返回 -1）
     */
    public RagasScores evaluate(String question, String context, String answer, String referenceAnswer) {
        double faithfulness = scoreFaithfulness(context, answer);
        double answerRelevancy = scoreAnswerRelevancy(question, answer);
        double contextPrecision = scoreContextPrecision(question, context);
        double contextRecall = referenceAnswer != null && !referenceAnswer.isBlank()
                ? scoreContextRecall(question, context, referenceAnswer)
                : -1;
        return new RagasScores(faithfulness, answerRelevancy, contextPrecision, contextRecall);
    }

    /**
     * Faithfulness：回答中的事实性陈述有多少能被上下文支持（RAGAS 定义）
     */
    public double scoreFaithfulness(String context, String answer) {
        if (context == null || context.isBlank() || answer == null || answer.isBlank()) return 0;
        String c = truncate(context, MAX_CTX_LEN);
        String a = truncate(answer, MAX_ANSWER_LEN);
        String prompt = """
                You are an evaluator. Given the RETRIEVED CONTEXT and the MODEL ANSWER, estimate what fraction of the factual claims in the answer are supported by the context.
                Output ONLY a single number between 0 and 1: 1 = all claims supported, 0 = no claims supported. No explanation.
                CONTEXT:
                ---
                %s
                ---
                ANSWER:
                ---
                %s
                ---
                Number (0 to 1):
                """.formatted(c, a);
        return parseScore(callLlm(prompt));
    }

    /**
     * Answer Relevancy：回答与问题的相关程度（RAGAS：回答是否切题）
     */
    public double scoreAnswerRelevancy(String question, String answer) {
        if (question == null || question.isBlank() || answer == null || answer.isBlank()) return 0;
        String a = truncate(answer, MAX_ANSWER_LEN);
        String prompt = """
                You are an evaluator. Given the QUESTION and the MODEL ANSWER, rate how relevant the answer is to the question.
                Output ONLY a single number between 0 and 1: 1 = fully relevant and on-topic, 0 = irrelevant or off-topic. No explanation.
                QUESTION: %s
                ANSWER: %s
                Number (0 to 1):
                """.formatted(question, a);
        return parseScore(callLlm(prompt));
    }

    /**
     * Context Precision：检索到的上下文与问题的相关程度（RAGAS：检索精度）
     */
    public double scoreContextPrecision(String question, String context) {
        if (question == null || question.isBlank() || context == null || context.isBlank()) return 0;
        String c = truncate(context, MAX_CTX_LEN);
        String prompt = """
                You are an evaluator. Given the QUESTION and the RETRIEVED CONTEXT, rate how relevant the context is for answering the question.
                Output ONLY a single number between 0 and 1: 1 = highly relevant, 0 = not relevant. No explanation.
                QUESTION: %s
                CONTEXT:
                ---
                %s
                ---
                Number (0 to 1):
                """.formatted(question, c);
        return parseScore(callLlm(prompt));
    }

    /**
     * Context Recall：上下文是否包含支撑参考答案所需的信息（RAGAS 定义，需参考答案）
     */
    public double scoreContextRecall(String question, String context, String referenceAnswer) {
        if (context == null || context.isBlank() || referenceAnswer == null || referenceAnswer.isBlank()) return 0;
        String c = truncate(context, MAX_CTX_LEN);
        String r = truncate(referenceAnswer, 1000);
        String prompt = """
                You are an evaluator. Given the QUESTION, the REFERENCE ANSWER (ground truth), and the RETRIEVED CONTEXT, rate whether the context contains the information needed to support the reference answer.
                Output ONLY a single number between 0 and 1: 1 = context fully supports the reference answer, 0 = context does not support it. No explanation.
                QUESTION: %s
                REFERENCE ANSWER: %s
                CONTEXT:
                ---
                %s
                ---
                Number (0 to 1):
                """.formatted(question, r, c);
        return parseScore(callLlm(prompt));
    }

    private String callLlm(String prompt) {
        try {
            return chatModel.call(new Prompt(prompt)).getResult().getOutput().getText();
        } catch (Exception e) {
            return "";
        }
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
            } catch (NumberFormatException ignored) {
            }
        }
        if (text.toLowerCase().contains("yes") || text.trim().startsWith("1")) return 1.0;
        if (text.toLowerCase().contains("no") || text.trim().startsWith("0")) return 0.0;
        return -1;
    }

    private static String truncate(String s, int maxLen) {
        if (s == null) return "";
        return s.length() <= maxLen ? s : s.substring(0, maxLen - 3) + "...";
    }
}
