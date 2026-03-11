package com.inovationbehavior.backend.ai.eval;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;

import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * RAG 评估入口：加载测试集 → 跑检索指标（Recall@K、MRR）→ 可选 RAGAS（Faithfulness / Answer Relevancy / Context Precision / Context Recall）。
 * 可通过 main 或单元测试调用，也可在 CI 中跑。
 */
@Slf4j
public class RagEvalRunner {

    private final DocumentRetriever retriever;
    private final ChatModel chatModel;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public RagEvalRunner(DocumentRetriever retriever, ChatModel chatModel) {
        this.retriever = retriever;
        this.chatModel = chatModel;
    }

    /**
     * 从 classpath 或文件路径加载测试用例（JSON 数组）
     */
    public List<RagTestCase> loadTestCases(String location) throws Exception {
        PathMatchingResourcePatternResolver resolver = new PathMatchingResourcePatternResolver();
        Resource[] resources = resolver.getResources(location);
        List<RagTestCase> all = new ArrayList<>();
        for (Resource r : resources) {
            if (!r.isReadable()) continue;
            try (InputStream is = r.getInputStream()) {
                List<RagTestCase> list = objectMapper.readValue(is, new TypeReference<>() {});
                all.addAll(list);
            }
        }
        return all;
    }

    /**
     * 只跑检索评估，返回指标
     */
    public RetrievalMetrics runRetrievalEval(List<RagTestCase> cases) {
        RagRetrievalEvaluator evaluator = new RagRetrievalEvaluator(retriever);
        return evaluator.evaluate(cases);
    }

    /**
     * 跑检索 + 忠实度：对每条用例检索 → 用 context 生成 answer → 打忠实度分
     */
    public EvalReport runFullEval(List<RagTestCase> cases, int maxFaithfulnessSamples) {
        RetrievalMetrics retrievalMetrics = runRetrievalEval(cases);

        List<Double> faithfulnessScores = new ArrayList<>();
        RagFaithfulnessEvaluator faithfulnessEvaluator = new RagFaithfulnessEvaluator(chatModel);
        int limit = maxFaithfulnessSamples <= 0 ? cases.size() : Math.min(maxFaithfulnessSamples, cases.size());
        AtomicInteger done = new AtomicInteger(0);

        for (int i = 0; i < limit; i++) {
            RagTestCase tc = cases.get(i);
            try {
                List<Document> docs = retriever.retrieve(new Query(tc.query()));
                String contextText = docs.stream()
                        .map(Document::getText)
                        .filter(t -> t != null && !t.isBlank())
                        .reduce("", (a, b) -> a + "\n\n" + b);
                if (contextText.length() > 6000) contextText = contextText.substring(0, 5997) + "...";
                String answer = chatModel.call(new Prompt(
                        "Based ONLY on the following context, answer the question concisely. If the context does not contain the answer, say 'I don't know based on the context.'\n\nContext:\n" + contextText + "\n\nQuestion: " + tc.query()
                )).getResult().getOutput().getText();
                double score = faithfulnessEvaluator.score(contextText, answer);
                if (score >= 0) faithfulnessScores.add(score);
            } catch (Exception e) {
                log.warn("Faithfulness eval failed for case: {}", tc.query(), e);
            }
            done.incrementAndGet();
            if (done.get() % 5 == 0) {
                log.info("Faithfulness progress: {}/{}", done.get(), limit);
            }
        }

        double avgFaithfulness = faithfulnessScores.isEmpty() ? -1 : faithfulnessScores.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        return new EvalReport(retrievalMetrics, faithfulnessScores.size(), avgFaithfulness);
    }

    /**
     * RAGAS 自动化评估：对每条用例检索 → 生成回答 → 计算 Faithfulness / Answer Relevancy / Context Precision / Context Recall。
     *
     * @param cases           测试用例（referenceAnswer 非空时才会算 context_recall）
     * @param maxSamples      最多评估条数，0 表示全部
     * @return RAGAS 汇总报告
     */
    public RagasReport runRagasEval(List<RagTestCase> cases, int maxSamples) {
        if (cases == null || cases.isEmpty()) {
            return new RagasReport(0, -1, -1, -1, -1, List.of());
        }
        int limit = maxSamples <= 0 ? cases.size() : Math.min(maxSamples, cases.size());
        RagasEvaluator ragas = new RagasEvaluator(chatModel);
        List<RagasScores> perSample = new ArrayList<>();

        for (int i = 0; i < limit; i++) {
            RagTestCase tc = cases.get(i);
            try {
                List<Document> docs = retriever.retrieve(new Query(tc.query()));
                String contextText = docs.stream()
                        .map(Document::getText)
                        .filter(t -> t != null && !t.isBlank())
                        .collect(Collectors.joining("\n\n"));
                if (contextText.length() > 6000) contextText = contextText.substring(0, 5997) + "...";

                String answer = chatModel.call(new Prompt(
                        "Based ONLY on the following context, answer the question concisely. If the context does not contain the answer, say 'I don't know based on the context.'\n\nContext:\n" + contextText + "\n\nQuestion: " + tc.query()
                )).getResult().getOutput().getText();

                RagasScores scores = ragas.evaluate(
                        tc.query(),
                        contextText,
                        answer,
                        tc.referenceAnswer() != null && !tc.referenceAnswer().isBlank() ? tc.referenceAnswer() : null
                );
                perSample.add(scores);
            } catch (Exception e) {
                log.warn("RAGAS eval failed for case: {}", tc.query(), e);
                perSample.add(RagasScores.none());
            }
            if ((i + 1) % 3 == 0) {
                log.info("RAGAS progress: {}/{}", i + 1, limit);
            }
        }

        List<RagasScores> valid = perSample.stream().filter(s -> s.faithfulness() >= 0).toList();
        int n = valid.size();
        double avgF = n == 0 ? -1 : valid.stream().mapToDouble(RagasScores::faithfulness).average().orElse(0);
        double avgR = n == 0 ? -1 : valid.stream().mapToDouble(RagasScores::answerRelevancy).average().orElse(0);
        double avgP = n == 0 ? -1 : valid.stream().mapToDouble(RagasScores::contextPrecision).average().orElse(0);
        List<RagasScores> withRecall = valid.stream().filter(s -> s.contextRecall() >= 0).toList();
        double avgCR = withRecall.isEmpty() ? -1 : withRecall.stream().mapToDouble(RagasScores::contextRecall).average().orElse(0);

        return new RagasReport(n, avgF, avgR, avgP, avgCR, List.copyOf(perSample));
    }

    /**
     * 收集测试数据并导出为 JSON，供 Python ragas 库读取评估。
     * 对每条用例：检索 → 生成回答 → 写入 user_input, retrieved_contexts（按 chunk 的字符串列表）, response, reference。
     *
     * @param testCasesLocation 测试用例路径（同 loadTestCases）
     * @param outputPath        输出 JSON 文件路径，如 target/ragas-eval-input.json
     * @param maxSamples        最多收集条数，0 表示全部
     * @return 导出的记录数
     */
    public int collectAndExportForRagas(String testCasesLocation, String outputPath, int maxSamples) throws Exception {
        List<RagTestCase> cases = loadTestCases(testCasesLocation);
        if (cases.isEmpty()) {
            log.warn("No test cases found at {}", testCasesLocation);
            return 0;
        }
        int limit = maxSamples <= 0 ? cases.size() : Math.min(maxSamples, cases.size());
        List<RagasExportRecord> records = new ArrayList<>();

        for (int i = 0; i < limit; i++) {
            RagTestCase tc = cases.get(i);
            try {
                List<Document> docs = retriever.retrieve(new Query(tc.query()));
                List<String> contexts = docs.stream()
                        .map(Document::getText)
                        .filter(t -> t != null && !t.isBlank())
                        .toList();
                String contextText = String.join("\n\n", contexts);
                if (contextText.length() > 6000) contextText = contextText.substring(0, 5997) + "...";

                String answer = chatModel.call(new Prompt(
                        "Based ONLY on the following context, answer the question concisely. If the context does not contain the answer, say 'I don't know based on the context.'\n\nContext:\n" + contextText + "\n\nQuestion: " + tc.query()
                )).getResult().getOutput().getText();

                records.add(RagasExportRecord.of(
                        tc.query(),
                        contexts.isEmpty() ? List.of(contextText) : contexts,
                        answer,
                        tc.referenceAnswer() != null && !tc.referenceAnswer().isBlank() ? tc.referenceAnswer() : ""
                ));
            } catch (Exception e) {
                log.warn("Export failed for case: {}", tc.query(), e);
            }
            if ((i + 1) % 5 == 0) {
                log.info("Export progress: {}/{}", i + 1, limit);
            }
        }

        Path path = Paths.get(outputPath);
        Path parent = path.getParent();
        if (parent != null && !Files.exists(parent)) {
            Files.createDirectories(parent);
        }
        try (OutputStream os = Files.newOutputStream(path)) {
            objectMapper.writerWithDefaultPrettyPrinter().writeValue(os, records);
        }
        log.info("Exported {} records to {}", records.size(), path.toAbsolutePath());
        return records.size();
    }

    /**
     * 汇总报告：检索指标 + 忠实度
     */
    public record EvalReport(RetrievalMetrics retrieval, int faithfulnessSamples, double avgFaithfulness) {
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("=== RAG Eval Report ===\n");
            sb.append(retrieval).append("\n");
            sb.append("Faithfulness: samples=").append(faithfulnessSamples);
            if (avgFaithfulness >= 0) sb.append(", avg=").append(String.format("%.4f", avgFaithfulness));
            else sb.append(", (no valid scores)");
            sb.append("\n");
            return sb.toString();
        }
    }
}
