# 多 Agent 图 — P&E + Replan 流程说明（Planner / Executor / Re-planner）

当前图采用 **Planner → Executor(ReAct) → CheckResult → Re-planner 动态更新剩余任务** 架构。

## 1. 架构与流程概览

1. **Planner（规划器）**：接收用户请求，生成 Task List（如 `[retrieval, analysis, advice, synthesize]`）。
2. **Executor（执行器）**：针对每个子任务启动一次 **ReAct 循环**（RAG + 工具调用），完成该任务后写 stepResults、currentStepIndex+1、needReplan。
3. **Re-planner（重规划器）**：每执行完一个子任务经 **CheckResult** 检查；若发现**环境变化**（如专利已失效）则**只更新剩余任务列表**；若为结果不足则整计划重算。

```
START
  → planner（生成 plan）
  → dispatch（plan[currentStepIndex] → executor | synthesize）
  → executor（当前子任务 ReAct 循环）→ stepResults、currentStepIndex+1、needReplan
  → checkResult（检查环境变化 → environmentChanged）
  → afterExpert（environmentChanged|needReplan → replan；否则 plan 未完成 → dispatch，完成 → synthesize）
  → replan（环境变化时只更新剩余任务；结果不足时整计划重算）→ dispatch
  → synthesize → END
```

## 2. 状态字段（P&E 相关）

| 字段 | 说明 |
|------|------|
| `plan` | 当前执行计划，如 `["retrieval","analysis","synthesize"]` |
| `currentStepIndex` | 当前执行到计划的第几步（0-based） |
| `needReplan` | 上一步执行结果是否“不足”（1=是），触发 Replan |
| `environmentChanged` | 是否检测到环境变化（如专利已失效）（1=是），触发只更新剩余任务 |

## 3. 节点职责

- **planner**：`IBApp.createPlan(userMessage, stepResults)` 生成计划；简单问候返回 `["synthesize"]`。
- **dispatch**：根据 `plan.get(currentStepIndex)` 派发到 **executor**（retrieval/analysis/advice）或 **synthesize**。
- **executor**：对当前子任务调用 `IBApp.doReActForTask(task, message, chatId, stepResults)`（RAG + 工具，Spring AI 单次 call 内可多轮 tool call）；写 stepResults、stepCount+1、currentStepIndex+1、needReplan。
- **checkResult**：取上一步结果与剩余任务，调用 `IBApp.checkEnvironmentChange(...)`，写 `environmentChanged`。
- **afterExpert**：若 `environmentChanged` 或 `needReplan` → replan；若 stepCount≥maxSteps 或 currentStepIndex≥plan.size() → synthesize；否则 → dispatch。
- **replan**：若 `environmentChanged`：`replanRemaining` 只更新剩余任务，保留已执行部分、currentStepIndex 不变；否则 `replan` 整计划重算、currentStepIndex=0。写 plan、needReplan=0、environmentChanged=0。

## 4. 何时触发 Replan / 更新剩余任务

- **结果不足（needReplan=1）**：`ReplanService.isResultInsufficient(expertOutput)` 为 true（空、过短、“无法”“没有找到”“无相关”“请提供”等）→ 进入 replan，**整计划重算**。
- **环境变化（environmentChanged=1）**：规则或 LLM 检测到“专利已失效/过期/撤回”“无此专利”等 → 进入 replan，**只更新剩余任务列表**（`replanRemaining`），已执行部分与 currentStepIndex 保留。

## 5. 配置与安全

- `app.agent.graph.max-steps`：afterExpert 中 stepCount≥maxSteps 强制 synthesize。
- 图 `recursionLimit(30)` 与 maxSteps 双重保险。
