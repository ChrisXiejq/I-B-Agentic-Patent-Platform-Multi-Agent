#!/usr/bin/env python3
"""
RAGAS 评估脚本：读取 Java 导出的 JSON，用 ragas 库计算 Faithfulness / Answer Relevancy / Context Precision / Context Recall。

用法（在 backend 目录下）:
  pip install -r scripts/requirements-ragas.txt
  python scripts/ragas_eval.py
  python scripts/ragas_eval.py --input target/ragas-eval-input.json --output target/ragas-eval-result.json
"""

import argparse
import json
import os
import sys

# 在此直接填写 OpenAI API Key，ragas 默认使用 OpenAI 做 LLM/embedding
OPENAI_API_KEY = ""


def main():
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on Java-exported JSON")
    parser.add_argument(
        "--input", "-i",
        default="target/ragas-eval-input.json",
        help="Input JSON path (default: target/ragas-eval-input.json)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Optional: save evaluation result JSON to this path",
    )
    args = parser.parse_args()

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    except ImportError as e:
        print("Missing dependency. Install with: pip install -r scripts/requirements-ragas.txt", file=sys.stderr)
        raise SystemExit(1) from e

    try:
        from ragas.dataset_schema import EvaluationDataset
    except ImportError:
        from ragas import EvaluationDataset

    if not os.path.isfile(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        print("Run Java export first, e.g. RagEvalRunnerTest.exportForRagas() or collectAndExportForRagas(...)", file=sys.stderr)
        raise SystemExit(2)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("No records in input file.", file=sys.stderr)
        raise SystemExit(3)

    # Java 导出格式与 ragas SingleTurnSample 一致：user_input, retrieved_contexts, response, reference
    if hasattr(EvaluationDataset, "from_list"):
        dataset = EvaluationDataset.from_list(data)
    else:
        from ragas.dataset_schema import SingleTurnSample
        samples = [
            SingleTurnSample(
                user_input=row.get("user_input", ""),
                retrieved_contexts=row.get("retrieved_contexts") or [],
                response=row.get("response", ""),
                reference=row.get("reference") or "",
            )
            for row in data
        ]
        dataset = EvaluationDataset(samples=samples)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    print("Running RAGAS evaluation (faithfulness, answer_relevancy, context_precision, context_recall)...")
    result = evaluate(dataset, metrics=metrics)

    print("\n=== RAGAS Results ===")
    print(result)

    if args.output:
        # result is EvaluationResult; save scores
        out_data = {
            "scores": getattr(result, "scores", None) or [],
            "dataset_size": len(data),
        }
        if hasattr(result, "__dict__"):
            for k, v in result.__dict__.items():
                if k not in ("scores",) and v is not None:
                    out_data[k] = str(v) if not isinstance(v, (int, float, list, dict)) else v
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"\nResult saved to {args.output}")


if __name__ == "__main__":
    main()
