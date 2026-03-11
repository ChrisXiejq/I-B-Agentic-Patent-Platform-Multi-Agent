package com.inovationbehavior.backend.ai.eval;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * RAG 自动化评估测试：加载测试集，跑 Recall@K、MRR；可选 RAGAS（Faithfulness / Answer Relevancy / Context Precision / Context Recall）。
 */
@SpringBootTest
class RagEvalRunnerTest {

    @Autowired
    private RagEvalRunner ragEvalRunner;

    @Value("${app.eval.rag.test-cases-location:classpath*:eval/rag-test-cases.json}")
    private String testCasesLocation;

    @Test
    @DisplayName("加载测试用例并跑检索评估（Recall@K、MRR）")
    void runRetrievalEval() throws Exception {
        List<RagTestCase> cases = ragEvalRunner.loadTestCases(testCasesLocation);
        assertNotNull(cases);
        if (cases.isEmpty()) {
            return; // 无测试数据时跳过
        }
        RetrievalMetrics metrics = ragEvalRunner.runRetrievalEval(cases);
        assertNotNull(metrics);
        assertTrue(metrics.totalCases() >= 0);
        System.out.println("Retrieval: " + metrics);
    }

    @Test
    @DisplayName("全量评估：检索 + Faithfulness（仅 2 条样本以控制 LLM 调用）")
    void runFullEvalSmallSample() throws Exception {
        List<RagTestCase> cases = ragEvalRunner.loadTestCases(testCasesLocation);
        if (cases.size() < 2) {
            return;
        }
        RagEvalRunner.EvalReport report = ragEvalRunner.runFullEval(cases, 2);
        assertNotNull(report);
        assertNotNull(report.retrieval());
        System.out.println(report);
    }

    @Test
    @DisplayName("RAGAS 评估：Faithfulness / Answer Relevancy / Context Precision / Context Recall（2 条样本）")
    void runRagasEvalSmallSample() throws Exception {
        List<RagTestCase> cases = ragEvalRunner.loadTestCases(testCasesLocation);
        if (cases.isEmpty()) {
            return;
        }
        RagasReport report = ragEvalRunner.runRagasEval(cases, 2);
        assertNotNull(report);
        assertTrue(report.samples() >= 0);
        System.out.println(report);
    }

    @Test
    @DisplayName("收集测试数据并导出为 JSON，供 Python ragas 读取")
    void exportForRagas() throws Exception {
        List<RagTestCase> cases = ragEvalRunner.loadTestCases(testCasesLocation);
        if (cases.isEmpty()) {
            return;
        }
        int n = ragEvalRunner.collectAndExportForRagas(testCasesLocation, "target/ragas-eval-input.json", 3);
        assertTrue(n >= 0);
        System.out.println("Exported " + n + " records to target/ragas-eval-input.json. Run: python scripts/ragas_eval.py");
    }
}
