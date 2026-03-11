# RAGAS 评估流程（Java 收集数据 → Python ragas 评估）

## 架构

1. **Java**：根据测试用例跑 RAG（检索 + 生成），把每条结果写成一条记录，导出为 JSON。
2. **Python**：用 ragas 库读取该 JSON，计算 Faithfulness / Answer Relevancy / Context Precision / Context Recall。

## 步骤

### 1. Java 导出测试数据

在工程里执行一次「收集并导出」：

- **方式 A**：跑单元测试  
  - 运行 `RagEvalRunnerTest.exportForRagas()`  
  - 会从 `eval/rag-test-cases.json` 加载用例，最多 3 条，导出到 `target/ragas-eval-input.json`。

- **方式 B**：在代码里调用  
  ```java
  int n = ragEvalRunner.collectAndExportForRagas(
      "classpath*:eval/rag-test-cases.json",
      "target/ragas-eval-input.json",
      0  // 0 = 全部用例
  );
  ```

导出 JSON 每行一条记录，字段与 ragas `SingleTurnSample` 对齐：

- `user_input`: 问题
- `retrieved_contexts`: 检索到的上下文（按 chunk 的字符串列表）
- `response`: 模型回答
- `reference`: 参考答案（可选）

### 2. Python 环境与依赖

在 **backend** 目录下：

```bash
pip install -r scripts/requirements-ragas.txt
```

在 `scripts/ragas_eval.py` 顶部将 `OPENAI_API_KEY = "your-openai-api-key-here"` 改为你的 key（ragas 默认用 OpenAI 做 LLM/embedding）。

### 3. 运行 RAGAS 评估

仍在 **backend** 目录下：

```bash
python scripts/ragas_eval.py
```

默认会读 `target/ragas-eval-input.json`，用 ragas 算四项指标并打印。

指定输入/输出：

```bash
python scripts/ragas_eval.py --input target/ragas-eval-input.json --output target/ragas-eval-result.json
```

## 配置

- 测试用例路径：`app.eval.rag.test-cases-location`（默认 `classpath*:eval/rag-test-cases.json`）
- 导出文件路径：`app.eval.rag.ragas-export-path`（默认 `target/ragas-eval-input.json`）
