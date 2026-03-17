# 专利商业化平台 AI 技术文档（面试向完整版）

本文档按 Agent 架构、三层记忆、工具、提示词、RAG、SSE/Advisor 等模块系统讲解，覆盖面试中可能被深挖的细节。与 `TECHNICAL-DOCUMENTATION.md`、`agent-graph-execution-flow.md`、`agent-graph-PE-Replan.md`、`README-memory.md` 等互补，本稿侧重「逐项透彻讲解」与「一问一答」式覆盖。

---

## 一、Agent 架构

### 1.1 Agent 主入口

**位置**：`controller/AiController.java`，统一前缀 `/ai`。

| 入口 | 方法 | 路径 | 作用 |
|------|------|------|------|
| 全能力 Agent（同步） | `fullAgentPost` | POST `/ai/agent` | 执行 P&E 图，返回 JSON `{ success, content, chatId }`。内部 `ibApp.doChatWithMultiAgentOrFull(message, chatId)` → `patentGraphRunner.run()`。当前请求体中的 `stream` 未参与分支，仅同步返回。 |
| P&E 图 SSE | `peStreamGet` / `peStreamPost` | GET/POST `/ai/agent/pe-stream` | 执行**完整 P&E 图**，图跑完后通过 SSE **一次性**推送整段最终回复（非逐 token）。 |

**区别小结**：

- **POST /ai/agent**：当前实现为 P&E 图同步调用，适合「要完整答案再展示」。
- **/ai/agent/pe-stream**：P&E 图 + SSE 传输，图结束后一次性推送整段内容，前端用 EventSource 接收；非逐 token 流式。
- 若需「单轮 RAG+工具逐 token 流式」，需在 Controller 中根据 `stream=true` 分支调用 `doChatWithFullAgentStream()` 并返回 SseEmitter（当前未实现）。

---

### 1.2 Agent 图结构

**配置类**：`ai/graph/PatentAgentGraphConfig.java`。使用 **LangGraph4j** 的 `StateGraph`，编译为 `CompiledGraph<PatentGraphState>`，递归上限 `recursionLimit(30)`。

**节点**（共 7 个）：

| 节点名 | 类 | 作用 |
|--------|-----|------|
| planner | PlannerNode | 根据用户消息生成执行计划（如 `retrieval,analysis,advice,synthesize`），写入 state 的 `plan`、`currentStepIndex=0`。 |
| dispatch | DispatchNode | 根据 `plan[currentStepIndex]` 决定下一节点：`retrieval`/`analysis`/`advice` → `executor`，否则 → `synthesize`。写 `nextNode`。 |
| executor | ExecutorNode | 对当前子任务执行一次「专家 ReAct」：调用 `ibApp.doReActForTask(task, message, chatId, stepResults)`，追加 step 结果、步数+1、needReplan。 |
| checkResult | CheckResultNode | 判断上一步结果是否导致「环境变化」（如专利失效、无数据），写 `environmentChanged`（0/1）。 |
| afterExpert | AfterExpertNode | 根据 needReplan、environmentChanged、是否还有剩余 step 决定下一节点：`replan` \| `synthesize` \| `dispatch`。 |
| replan | ReplanNode | 调用 `ReplanService.replanRemaining` / `replan`；若 environmentChanged 则只重规划剩余步骤，否则整计划重算。更新 `plan`、`currentStepIndex`，清 needReplan/environmentChanged。 |
| synthesize | SynthesizeNode | 根据用户问题 + 所有 stepResults 调用 LLM 生成最终回复，写 `finalAnswer`。 |

**边**：

- `START → planner → dispatch`。
- `dispatch` **条件边**：`nextNode == "executor"` → executor，否则 → synthesize。
- `executor → checkResult → afterExpert`。
- `afterExpert` **条件边**：`nextNode` ∈ { `replan`, `synthesize`, `dispatch` } 分别到对应节点。
- `replan → dispatch`（继续执行新计划）；`synthesize → END`。

**执行入口**：`PatentGraphRunner.run(userMessage, chatId)` 构建初始 state，调用 `patentAgentGraph.invoke(initialState, config)`，从最终状态取 `finalAnswer()`；异常时返回兜底文案或对 503/429 等瞬时错误重新抛出。

---

### 1.3 P&E 结构与实现

**P&E** = Plan（规划）+ Execute（执行），本项目中还包含 **Replan**（重规划），合称 P&E + Replan。

- **Plan（PlannerNode）**  
  - 实现：`IBApp.createPlan(userMessage, stepResults)`。  
  - 逻辑：若为简单问候/自我介绍则直接返回 `["synthesize"]`；否则用 `PLAN_PROMPT` 调 LLM，得到逗号分隔的步骤列表（如 `retrieval,analysis,advice,synthesize`），解析为 `List<String>` 写入 state 的 `plan`，`currentStepIndex=0`。

- **Execute（ExecutorNode）**  
  - 实现：按 `plan.get(currentStepIndex)` 取当前任务（retrieval/analysis/advice），调用 `IBApp.doReActForTask(task, message, chatId, stepResults)`。  
  - 内部：query 改写 → 按 task 选专家 systemPrompt → 若为 retrieval 且上一步结果不足则注入「请用 searchWeb 补足」→ 构建 **GraphTaskAgent**（IBManus 架构：think→act 多步循环，带 RAG/记忆/Trace）→ `agent.run(rewritten)`，返回值追加到 `stepResults`，`currentStepIndex+1`，`needReplan` 由 `ReplanService.isResultInsufficient(out)` 决定。

- **Replan（ReplanNode）**  
  - 触发：AfterExpert 根据 `needReplan` 或 `environmentChanged` 将 nextNode 设为 `replan`。  
  - 实现：**ReplanNode 调用 `ReplanService`**（与 IBApp 解耦）：  
    - **环境变化**：`replanService.replanRemaining(userMessage, stepResults, remainingTasks)`，仅重规划剩余步骤；若 `ReplanService.shouldRetryRetrievalWithWeb(lastResult)` 则强制 `["retrieval","synthesize"]` 以便用 searchWeb 重试。  
    - **结果不足**：`replanService.replan(userMessage, stepResults)` 整计划重算。  
  - 写回：新 `plan`、新 `currentStepIndex`，`needReplan=0`，`environmentChanged=0`。

**CheckResult**：`IBApp.checkEnvironmentChange(lastStepResult, remainingTasks, userMessage)` 用 LLM 判断上一步是否表示环境变化（专利失效、无数据等），返回 yes/no，写入 `environmentChanged`。

**AfterExpert**：根据 `needReplan`、`environmentChanged`、是否还有剩余 step、是否已达 maxSteps 等决定下一节点，保证要么继续执行（dispatch）、要么重规划（replan）、要么结束（synthesize）。

---

### 1.4 每个 Agent 如何实现

图内「单任务专家」统一用 **IBManus 架构** 的 **GraphTaskAgent** 实现（参考 OpenManus 分层）。

**分层**：

- **BaseAgent**：维护 name、systemPrompt、nextStepPrompt、state（IDLE/RUNNING/FINISHED/ERROR）、currentStep、maxSteps、chatClient、messageList；提供 `run(userPrompt)`（同步多步循环）和 `runStream(userPrompt)`（SSE）；每步调用抽象 `step()`。
- **ReActAgent**：继承 BaseAgent，`step()` = 先 `think()` 再按需 `act()`；子类实现 `think()`（是否要调用工具）、`act()`（执行工具）。
- **ToolCallAgent**：继承 ReActAgent，持有 availableTools、ToolCallingManager、ChatOptions（GoogleGenAiChatOptions，`internalToolExecutionEnabled(false)`）；`think()` 中调用 ChatClient.prompt(prompt).system().tools().call()，解析 tool_calls，无工具则返回 false；`act()` 中执行工具、写回 conversationHistory，检测 doTerminate 则置 state=FINISHED。支持可选 `conversationId` 与 `extraAdvisors`（图内 RAG/记忆/Trace）。
- **GraphTaskAgent**：继承 ToolCallAgent，构造函数接收 (allTools, chatModel, systemPrompt, nextStepPrompt, chatId, ragAdvisor, memoryPersistenceAdvisor, agentTraceAdvisor)，构建 ChatClient（MyLoggerAdvisor），设置 conversationId 与 extraAdvisors；`run()` 重写为返回**最后一条助手消息文本**供 Synthesize 使用。

**IBManus**：继承 ToolCallAgent，固定 systemPrompt/nextStepPrompt，用于独立 Manus 入口（如 `/ai/manus/chat`，若启用）；图内不直接使用。

因此：图内每个「个体 agent」（检索/分析/建议专家）都是 **GraphTaskAgent** 的一次运行，即 **同一套 think→act 多步循环**，仅 systemPrompt 和 RAG Advisor（retrieval 用 retrievalExpertRagAdvisor）按任务区分。

---

### 1.5 Agent 之间怎样串联

- **串联方式**：通过 **LangGraph4j 的图** 与 **共享状态 PatentGraphState** 串联。  
- **顺序**：Planner（生成 plan）→ Dispatch（按 plan 与 currentStepIndex 选下一节点）→ 若为 executor 则 Executor（执行当前子任务，写 stepResults）→ CheckResult（写 environmentChanged）→ AfterExpert（写 nextNode）→ 若 replan 则 Replan（更新 plan/currentStepIndex）→ 再回到 Dispatch；若 synthesize 则 Synthesize → END。  
- **条件分支**：Dispatch 与 AfterExpert 使用 `addConditionalEdges`，根据 state 的 `nextNode` 或业务条件决定下一节点，实现「多步执行 + 重规划 + 提前结束」的闭环。

---

### 1.6 Agent 之间的信息传递与状态

**状态定义**：`PatentGraphState`（`ai/graph/PatentGraphState.java`），继承 LangGraph4j 的 `AgentState`，通过 `SCHEMA` 声明各 channel 的类型与默认值。

| Channel | 类型 | 说明 |
|---------|------|------|
| userMessage | base | 用户输入。 |
| chatId | base | 会话 ID，用于记忆与 Trace。 |
| stepResults | appender | 各专家步输出累积列表（如 "[Task:retrieval]\n..."）。 |
| nextNode | base | 下一节点名（dispatch/afterExpert 写入）。 |
| finalAnswer | base | 最终回复（Synthesize 写入）。 |
| stepCount | base | 已执行专家步数。 |
| maxSteps | base | 最大步数上限。 |
| plan | base | 当前执行计划（List<String>）。 |
| currentStepIndex | base | 当前执行到 plan 的第几步。 |
| needReplan | base | 上一步是否结果不足需重规划。 |
| environmentChanged | base | 是否检测到环境变化。 |

**传递方式**：每节点 `apply(state)` 返回 `Map<String, Object>`（只包含本节点要更新的 key），LangGraph4j 按 SCHEMA 合并到共享 state；下一节点读取同一 state（如 `plan()`、`currentStepIndex()`、`stepResults()`）。例如 Executor 写入 `STEP_RESULTS`（appender 追加）、`STEP_COUNT`、`CURRENT_STEP_INDEX`、`NEED_REPLAN`；Synthesize 读取 `stepResults()` 与 `userMessage()` 生成 `FINAL_ANSWER`。

**信息流**：用户消息 → Planner 生成 plan → Dispatch 按 plan 派发 → Executor 每步把「当前任务 + 用户消息 + 已有 stepResults」交给 GraphTaskAgent，专家输出追加到 stepResults → CheckResult 用「最后一条 stepResult」判断环境变化 → AfterExpert 决定继续/重规划/结束 → Replan 时用「userMessage + stepResults + remainingTasks」重算剩余步骤 → Synthesize 用「userMessage + 全部 stepResults」生成最终答案。

### 1.7 为什么重规划逻辑在 ReplanService 而不是 IBApp 或 ReplanNode？

- **历史原因**：早期把「规划 + 重规划 + 综合 + 环境检查」都放在 **IBApp** 里，图节点只做「读 state → 调 IBApp → 写 state」，这样 IBApp 成为图相关 LLM 调用的唯一入口，节点保持薄、避免循环依赖。但重规划（`replan`、`replanRemaining`、`shouldRetryRetrievalWithWeb`、`isResultInsufficient` 等）从职责上属于「重规划」领域，与「应用入口 / 全能力对话」的 IBApp 无关，放在 IBApp 会导致类膨胀、职责混杂。
- **当前设计**：重规划逻辑已抽到 **`ai/app/ReplanService`**：
  - **ReplanNode** 只依赖 `ReplanService`，调用 `replanService.replan()` 与 `replanService.replanRemaining()`，不再依赖 IBApp。
  - **ExecutorNode** 判断 `needReplan` 时调用 `ReplanService.isResultInsufficient(out)`（静态方法）。
  - **IBApp.doReActForTask** 在「检索且上一步结果不足」时调用 `ReplanService.shouldRetryRetrievalWithWeb(lastResult)` 以决定是否注入 searchWeb 补足提示；IBApp 仅保留「执行单任务」与「规划/综合/环境检查」等入口相关逻辑。
- **为何不直接放进 ReplanNode**：若把 LLM 调用、REPLAN 提示词、`parsePlan`、`shouldRetryRetrievalWithWeb` 等全部塞进 ReplanNode，节点会变得臃肿且难以单测；且 `isResultInsufficient` 会被 ExecutorNode 与 doReActForTask 共用，放在 ReplanNode 会导致 ExecutorNode 依赖 ReplanNode（语义奇怪）。抽成 **ReplanService** 后，ReplanNode 与 IBApp 按需注入，职责清晰、便于测试与扩展。

---

## 二、三层记忆架构说明

（以下与你提供的描述一致，并按要求做了更精确的修正。）

### 2.1 Layer 1：Working Memory

**作用**：在内存中做滑动窗口和剪枝，控制上下文规模。

- **滑动窗口**：配置 `window-size`（如 10），超过则从最旧一侧取 `turns-to-compress`（如 5）轮做压缩；压缩结果拼接到 `runningSummary`，再删除这些旧轮，保持窗口大小。  
- **Importance 剪枝**：总字符超过 `max-total-chars`（如 8000）时触发；移除 `importance < prune-threshold`（如 0.1）的轮次。  
- **溢出到 Layer 2**：每次压缩得到的摘要会写入 Experiential Memory。

**实现**：`WorkingMemoryService`，依赖 `ImportanceScorer`、`SummaryCompressor`；配置项 `app.memory.working.*`。

---

### 2.2 Importance 如何计算？

实现见 **PatentDomainImportanceScorer**，基于规则的领域打分（类似 NER/关键词）：

| 维度 | 说明 | 默认权重 |
|------|------|----------|
| 专利号/编号 | 匹配 CN/EP/US/WO/ZL + 数字或 8–15 位数字 | +0.35 |
| 领域关键词 | 命中如：专利、许可、转让、商业化、价值评估、专利号、检索、侵权等 | 每词约 +0.12，上限 0.40 |
| 长度因子 | 总长度 ≥50 字加 0.125，≥150 字再加 0.125 | 最多 +0.25 |

最终得分限制在 `[0, 1]`。`application.yaml` 可配置：`keywords`、`patent-number-weight`、`keyword-weight`、`length-factor` 等（`app.memory.importance.*`）。

---

### 2.3 递归摘要如何实现？

由 **LlmSummaryCompressor.compress(previousSummary, turnsToCompress)** 完成：

1. **输入**：`previousSummary`（当前累积的 runningSummary）、`turnsToCompress`（本次要压缩的若干轮对话）。  
2. **Prompt**：若已有摘要则「已有摘要（可与之合并）：... 新增对话：...」，否则只给「对话内容：...」。  
3. **输出**：一段压缩摘要（约 80–300 字，强调专利号、用户意图、关键结论与承诺）。  
4. **递归累积**：新摘要拼接到 runningSummary：`runningSummary = runningSummary + "\n" + newSegment`。即每次溢出时只压缩「最旧 N 轮 + 已有摘要」，再把结果累加。

---

### 2.4 重要性衰减如何实现？

**当前实现**：

- **ExperientialMemoryService** 为每条摘要写入 `decay_weight`（默认 1.0）；检索时按 `decay_weight` 降序、`created_at` 降序排序取 topK。  
- **写入时**：目前所有条目的 `decay_weight` 均为默认值，**未**根据时间或访问情况做衰减。

更严谨的说法：**Experiential Memory 通过递归摘要沉淀事件摘要与用户意图，带 `decay_weight` 元数据用于检索时排序；当前未实现「未检索衰减」等后台衰减策略。**

---

### 2.5 Semantic Memory 如何筛选？

在 **LongTermMemoryService.storeIfEligible** 中，满足以下条件才写入长期记忆：

| 条件 | 说明 |
|------|------|
| Importance 阈值 | `importance >= importance-threshold`（如 0.25），低价值轮次直接丢弃。 |
| 相似度去重 | 与已有记忆的相似度 ≥ `duplicate-similarity-threshold`（如 0.92）视为重复，不写入。 |
| NLI 冲突检测 | 若与已有记忆冲突，先删除旧事实，再写入新事实。 |

---

### 2.6 NLI 如何实现？

**LlmNliConflictDetector.hasConflict(newMemoryContent, existingMemories)**：

1. **输入**：新记忆文本；已有记忆 top-5（由相似度召回）。  
2. **Prompt**：要求 LLM 判断新记忆与已有记忆在事实或主张上是否矛盾（例如同一专利结论相反、用户意图前后冲突）。  
3. **输出**：仅回答 YES 或 NO；若以 YES 开头则判定为冲突。  
4. **处理逻辑**：冲突 → 删除最相似的那条旧记忆，再写入新记忆（UPDATE）；互补 → 直接追加（MERGE）。

---

### 2.7 分层记忆架构（小结）

| 层级 | 描述 | 实现情况 |
|------|------|----------|
| Working Memory | 滑动窗口 + importance 剪枝 | ✓ 准确 |
| Experiential Memory | 递归摘要 + 检索按 decay_weight 排序 | 递归摘要 ✓；衰减仅有元数据，未实现未检索衰减 |
| Semantic Memory | NLI 控制写入，长期事实与偏好 | ✓ 准确 |

记忆不自动注入 Prompt，由 Agent 通过 **retrieve_history** 工具按需拉取，减少干扰与 token 消耗。**记忆可更新**：通过 NLI 的 UPDATE（删旧写新）与 MERGE（追加）；当前未实现用户主动「修正/删除某条记忆」的 API 与 Experiential 的自动衰减策略。

---

## 三、工具

### 3.1 WebSearch

**类**：`ai/tools/WebSearchTool`。  
**能力**：`searchWeb(query)`，调用 SearchAPI.io（当前配置 engine=baidu），取前 5 条 organic_results 拼接为字符串返回。  
**用途**：当 RAG/知识库无法覆盖或需补充最新信息时，由模型在 ReAct 中调用；检索专家在「无专利号/接口失败」时也会被提示优先使用 searchWeb 补足。  
**配置**：如 `search-api.api-key`（需在配置中提供）。

### 3.2 其余工具（补全）

| 工具类 | 方法/用途 | 说明 |
|--------|------------|------|
| PatentDetailTool | 按专利号查详情 | 依赖业务/Redis 等，失败时可能返回连接错误，触发 Replan 用 searchWeb 重试。 |
| PatentHeatTool | 按专利号查热度/状态 | 同上。 |
| UserIdentityTool | 用户/问卷身份 | 供建议专家做个性化建议。 |
| MemoryRetrievalTool | retrieve_history(conversation_id, query) | 按需检索三层记忆（短期窗口、中期摘要、长期事实），由 Agent 显式调用。 |
| FileOperationTool / WebScrapingTool / ResourceDownloadTool | 文件/网页/资源 | 通用能力，供模型按需调用。 |
| TerminalOperationTool / PDFGenerationTool | 终端/PDF 生成 | 通用或专项能力。 |
| TerminateTool | doTerminate | 图内专家 ReAct 结束时调用，用于 BaseAgent 将 state 置为 FINISHED。 |

**注册**：`ToolRegistration` 将上述工具（含可选 MemoryRetrievalTool）组装为 `ToolCallback[]`，供 ChatClient 与 GraphTaskAgent 使用。可选 `TracingToolCallbackProvider`（`app.agent.trace-tools=true`）包装以打印每次工具调用的 name/input/result。

---

## 四、提示词

### 4.1 全能力单 Agent 与 Manus

- **SYSTEM_PROMPT**（IBApp）：全能力 ChatClient 的 defaultSystem，介绍平台能力、多轮中可调用的工具与 retrieve_history，引导简洁专业回复。  
- **IBManus** 自有 **SYSTEM_PROMPT** 与 **NEXT_STEP_PROMPT**：强调工具优先、多步任务先查再结论、结束时调用 terminate，用于独立 Manus 入口（若启用）。

### 4.2 图内 P&E 与专家

- **PLAN_PROMPT**：Planner 用，输出逗号分隔的步骤列表（retrieval, analysis, advice, synthesize），规则包括简单问候仅 synthesize、多步查询 retrieval,analysis,synthesize 等。  
- **REPLAN_PROMPT**：整计划重算用，说明上一步结果不足时可输出 retrieval,synthesize 或 synthesize。  
- **REPLAN_REMAINING_PROMPT**：仅重规划剩余步骤；明确「若上一步为 API/连接失败则必须输出 retrieval,synthesize」。  
- **CHECK_ENV_PROMPT**：CheckResult 用，判断上一步结果是否表示环境变化（专利失效、无数据等），仅回答 yes/no。  
- **RETRIEVAL_EXPERT_PROMPT / ANALYSIS_EXPERT_PROMPT / ADVICE_EXPERT_PROMPT**：Executor 按 task 选择，分别定义检索专家、分析专家、建议专家的职责与可用工具（含 searchWeb、getPatentDetails、getPatentHeat、retrieve_history 等）。  
- **GRAPH_TASK_NEXT_STEP_PROMPT**：图内专家 think/act 的下一步提示，要求完成当前任务、必要时调用工具、完成后可调用 doTerminate。  
- **Synthesize 用 prompt**：在 `IBApp.synthesizeAnswer` 中，将用户问题与全部 stepResults 拼成上下文，要求 LLM 生成面向用户的最终回复。

### 4.3 其余提示词

- **RAG 空上下文**：ContextualQueryAugmenterFactory 中「无相关内容时」的固定话术；检索专家专用版本在空上下文时要求先调用 searchWeb。  
- **评估**：RagFaithfulnessEvaluator 等使用 JUDGE_PROMPT 做忠实度判断（若项目启用评测）。

---

## 五、RAG

### 5.1 文档清洗

- **来源**：`app.rag.documents.path`（默认 `classpath*:documents/**/*.md`），仅加载 .md。  
- **读取**：`DocumentLoader.loadMarkdownDocuments()` 使用 Spring AI 的 `MarkdownDocumentReader`（含 code block、blockquote 等），每文件一个 Document，metadata 含 `source`（URI/filename）。  
- **清洗**：当前未做额外去噪、去重或格式规范化；若需可在此步扩展（如 strip 无关模板、合并短段等）。

### 5.2 文档向量化

- **分块**：`DocumentLoader.loadDocumentsForRag()` 在加载后经 `ChunkSplitter` 分块；支持递归字符切分（LangChain4jRecursiveSplitter）或语义切分+Token 回退（SemanticChunkSplitter），由 `app.rag.use-semantic` 与 `RagSplitterConfig` 决定。  
- **向量化**：由 **PgVectorVectorStoreConfig** 在启动时或增量同步时，将 chunk 列表交给 `PgVectorStore`（Spring AI）；底层使用配置的 **EmbeddingModel**（如 OpenAI text-embedding-3-small，1536 维）对 chunk 文本做 embedding 后写入 PgVector 表 `vector_store`。

### 5.3 文档入库

- **流程**：`RagDocumentCorpus` 持有 `DocumentLoader.loadDocumentsForRag()` 的结果；`PgVectorVectorStoreConfig` 对该列表做**增量同步**：为每个 chunk 生成 `chunk_key`（sha256(source|text)），已存在则跳过，新增则写入。  
- **BM25**：同一批 chunk 列表同时提供给 `BM25DocumentRetriever` 建内存倒排索引，供混合检索使用。  
- **配置**：PgVector 使用独立数据源（url、dimensions、index-type、distance-type 等），表名 `vector_store`。

### 5.4 文档检索

- **混合检索**：`HybridDocumentRetriever.retrieve(Query)`：  
  - 向量路：`VectorStoreDocumentRetriever`，topK=vectorTopK（如 8），similarityThreshold=0.3。  
  - BM25 路：`BM25DocumentRetriever`，topK=bm25TopK（如 8）。  
  - 两路结果经 **RRF（Reciprocal Rank Fusion）** 融合，再经 **EmbeddingReranker** 用 query-doc 相似度精排，取 finalTopK（如 6）。  
- **配置**：`app.rag.hybrid.vector-top-k`、`bm25-top-k`、`final-top-k`。

### 5.5 检索前与检索后

- **检索前**：  
  - **QueryRewriter**：使用 Spring AI 的 `RewriteQueryTransformer`，对用户 query 做改写以优化向量检索（在 doReActForTask/doExpertChat 等处可选调用）。  
  - **ContextualQueryAugmenter**：在 RAG Advisor 内使用，将「已有摘要/对话」与「当前 query」结合；检索专家专用版本在空上下文时返回要求先 searchWeb 的模板。  
- **检索后**：**EmbeddingReranker** 作为 post-retrieval 精排；无额外重写或过滤步骤在代码中单独命名。

### 5.6 与 Agent 的衔接

- **RAG Advisor**：`RetrievalAugmentationAdvisor` 将 `DocumentRetriever`（本项目中为 hybridDocumentRetriever）与 `QueryAugmenter` 组合，在 ChatClient 的 prompt 链中注入「Context information is below...」+ 检索结果。  
- **图内**：Executor 使用的 GraphTaskAgent 将 `retrievalExpertRagAdvisor`（retrieval 任务）或 `hybridRagAdvisor`（analysis/advice）作为 extraAdvisor，在每次 think() 时随请求注入 RAG 上下文。

---

## 六、其余内容

### 6.1 SSE 流式输出

- **实现**：使用 **SseEmitter**（超时 5 分钟）。  
- **/ai/agent/pe-stream**：跑**完整 P&E 图**，图执行结束后**一次性** `emitter.send(SseEmitter.event().data(content))` 推送整段回复，因此**不是**「边执行边流式」或「逐 token 流式」，而是「图结束后用 SSE 推送整段」；前端可用 EventSource 接收，逻辑上算 SSE 传输，但不是逐 token。  
- 当前 **POST /ai/agent** 未根据 `stream` 参数分支；若需「单轮 RAG+工具逐 token 流式」，需在 Controller 中增加分支并调用 `ibApp.doChatWithFullAgentStream()`（若实现）并返回 SseEmitter。

若要「图内某步或最终回复」逐 token 流式，需在 Synthesize 或最终一步使用 LLM 的 stream 接口并在 SSE 上按 chunk 推送，当前未实现。

### 6.2 Advisor

每个 Advisor 实现 **CallAdvisor**（同步）和/或 **StreamAdvisor**（流式），通过 **getOrder()** 参与排序；**before** 在请求进入链时执行，**after** 在得到响应后执行（同步在 adviseCall 内，流式在 adviseStream 的 aggregate 回调中）。

图内专家链上主要用到的四个 Advisor 如下。

---

#### 6.2.1 hybridRagAdvisor

| 项目 | 说明 |
|------|------|
| **定义位置** | `ai/rag/config/HybridRagConfig.java`，Bean 方法 `hybridRagAdvisor()`。 |
| **类型** | Spring AI 的 `RetrievalAugmentationAdvisor`，非自定义类。 |
| **作用** | 在每次 ChatClient 请求时：先用 `DocumentRetriever`（本项目的 `hybridDocumentRetriever`）做混合检索（向量 + BM25 → RRF → Rerank），再用 `QueryAugmenter`（`ContextualQueryAugmenterFactory.createInstance()`）处理 query；若有检索结果，将「Context information is below...」+ 文档片段注入到发给 LLM 的 prompt 中，实现 RAG 增强。 |
| **空上下文行为** | 使用默认的 `ContextualQueryAugmenter`：`allowEmptyContext(false)`，空上下文时用固定话术模板要求模型直接回复「知识库无相关内容，可输入专利号或描述需求以调用工具」，**不**触发 searchWeb。 |
| **使用场景** | 图内 **analysis**、**advice** 等非检索专家；以及全能力单轮对话（如 `doExpertChat` 中 `.advisors(hybridRagAdvisor)`）。 |

---

#### 6.2.2 retrievalExpertRagAdvisor

| 项目 | 说明 |
|------|------|
| **定义位置** | `ai/rag/config/HybridRagConfig.java`，Bean 方法 `retrievalExpertRagAdvisor()`，`@Qualifier("retrievalExpertRagAdvisor")`。 |
| **类型** | 同上，`RetrievalAugmentationAdvisor`，仅配置不同。 |
| **作用** | 与 hybridRagAdvisor 一样做混合检索并注入上下文，区别在 **QueryAugmenter**：使用 `ContextualQueryAugmenterFactory.createInstanceForRetrievalExpertWithWebFallback()`。 |
| **空上下文行为** | `allowEmptyContext(true)`，空上下文时使用专用模板：保留用户问题，并**明确要求必须先调用 searchWeb** 获取网络信息再回答，从而在知识库无命中时自动引导模型使用网页搜索工具。 |
| **使用场景** | 图内 **retrieval** 专家（`IBApp.doReActForTask` 中当 systemPrompt 包含 "retrieval" 时选用此 Advisor）。 |

---

#### 6.2.3 memoryPersistenceAdvisor

| 项目 | 说明 |
|------|------|
| **定义位置** | 实现类 `ai/memory/MemoryPersistenceAdvisor.java`；Bean 在 `ai/memory/MultiLevelMemoryConfig.java` 中声明，`@Bean("memoryPersistenceAdvisor")`。 |
| **类型** | 自定义类，实现 `CallAdvisor`、`StreamAdvisor`。 |
| **作用** | **仅做持久化，不向 Prompt 注入任何内容**。在每轮对话**结束后**（after 链）：根据 request 中的 conversationId、用户消息、以及 response 中的助手回复，调用 `persistTurn(conversationId, userMessage, assistantMessage)`，依次写入三层记忆——Working Memory（`workingMemoryService.addTurn`，可能触发滑动窗口压缩）、若压缩产生摘要则写入 Experiential（`experientialMemoryService.addSummary`）、再按重要性等条件写入 Long-Term（`longTermMemoryService.storeIfEligible`）。 |
| **Order** | `Ordered.LOWEST_PRECEDENCE + 100`，在链中偏后执行，确保在拿到完整回复后再写记忆。 |
| **流式** | 同步用 `adviseCall` 直接取 response 文本写盘；流式用 `ChatClientMessageAggregator.aggregateChatClientResponse` 在流结束后再取聚合结果写盘。 |
| **记忆检索** | 不由本 Advisor 注入；由 Agent 通过 **MemoryRetrievalTool**（`retrieve_history`）按需拉取，节省 token 并减少干扰。 |

---

#### 6.2.4 agentTraceAdvisor

| 项目 | 说明 |
|------|------|
| **定义位置** | `ai/advisor/AgentTraceAdvisor.java`；Bean 在 `ai/config/AgentTraceConfig.java` 中声明，`@Qualifier("agentTraceAdvisor")`。 |
| **类型** | 自定义类，实现 `CallAdvisor`、`StreamAdvisor`。 |
| **作用** | **可观测**：在请求前（before）打日志：conversationId、用户消息长度与预览（如 120 字）；在响应后（after）打日志：助手回复长度与预览（如 200 字）、本轮若有工具调用则逐条打印 `name` 与 `arguments` 预览（300 字）、以及 metadata 中的 usage、model。便于排查与理解 Agent 的思考与执行过程。 |
| **Order** | `Ordered.LOWEST_PRECEDENCE - 200`，数值较小，在链中**较早**执行，从而能包住整次请求-响应与工具调用。 |
| **流式** | 流式时先 before，再 `nextStream`，用 `ChatClientMessageAggregator.aggregateChatClientResponse` 在流结束后调用 after，因此看到的是「整段回复聚合后」的日志。 |
| **配合** | 可与 `TracingToolCallbackProvider`（`app.agent.trace-tools=true`）配合，同时看到每次工具执行的入参与返回值。 |

---

#### 6.2.5 为什么这些 Advisor 不在同一包（advisor）下？

项目中与 ChatClient 链相关的 Advisor 分布在 **advisor**、**memory**、**rag/config** 等不同包，原因如下：

| Advisor | 所在包/位置 | 原因 |
|---------|-------------|------|
| **AgentTraceAdvisor** | `ai/advisor/` | 通用可观测逻辑，与具体业务域无关，放在统一的 advisor 包更符合「横切关注点」的归类；Bean 在 `ai/config/AgentTraceConfig` 中创建。 |
| **BannedWordsAdvisor、MyLoggerAdvisor、ReReadingAdvisor** | `ai/advisor/` | 同样是通用能力（违禁词、日志、重读），与领域无关，归入 advisor 包。 |
| **memoryPersistenceAdvisor**（实现类 MemoryPersistenceAdvisor） | `ai/memory/` | 强依赖 WorkingMemoryService、LongTermMemoryService、ImportanceScorer、ExperientialMemoryService，是「记忆模块」的**写盘出口**，与记忆的配置、Bean 组装（MultiLevelMemoryConfig）放在同一包，便于内聚；若放到 advisor 包，memory 包还要反向依赖 advisor，边界更乱。 |
| **hybridRagAdvisor、retrievalExpertRagAdvisor** | 无独立类，Bean 在 `ai/rag/config/HybridRagConfig` | 二者是 Spring AI 的 `RetrievalAugmentationAdvisor` 的**两种配置实例**，依赖 VectorStore、RagDocumentCorpus、EmbeddingModel、DocumentRetriever、QueryAugmenter 等，全部在 RAG 模块内；放在 `rag/config` 与 HybridDocumentRetriever、BM25、Reranker、ContextualQueryAugmenterFactory 等一起配置，符合「RAG 能力内聚」。若在 advisor 包里建两个空壳类只为了返回这两个 Bean，反而增加无意义的间接层。 |

**总结**：  
- **有自定义实现、且属于某一领域能力**的（如记忆写盘），放在该领域包（memory）并在此包内声明 Bean，保持高内聚、避免循环依赖。  
- **仅对框架/第三方 Advisor 做参数组装**的（如两种 RAG Advisor），放在对应功能配置包（rag/config），不单独建 advisor 子类。  
- **与领域无关的横切逻辑**（日志、追踪、违禁词）放在 `ai/advisor/`，便于统一查找与扩展。

---

#### 6.2.6 其余 Advisor（简要）

- **BannedWordsAdvisor**（`ai/advisor/`）：请求前检查用户输入是否命中违禁词，命中则抛异常；`Ordered.HIGHEST_PRECEDENCE`，最先执行。  
- **MyLoggerAdvisor**（`ai/advisor/`）：打 Request/Response 日志；在 ChatClient defaultAdvisors 中加入。  
- **ReReadingAdvisor**（`ai/advisor/`）：当前未在图或全能力链中挂载，为可选扩展。

**典型链顺序**（ChatClient defaultAdvisors）：BannedWordsAdvisor（`HIGHEST_PRECEDENCE`）→ MessageChatMemoryAdvisor → MyLoggerAdvisor；全能力/专家链上再追加 ragAdvisor、memoryPersistenceAdvisor（`LOWEST_PRECEDENCE + 100`）、agentTraceAdvisor（`LOWEST_PRECEDENCE - 200`）；GraphTaskAgent 内 extraAdvisors 顺序为 memoryPersistence → agentTrace → rag。

### 6.3 面试向补充（可深挖点）

- **为何 P&E 还要 Replan**：单步结果可能不足（如检索失败、模型回复「I don't know」），需要根据结果动态收缩或重试（如强制 retrieval+synthesize 并用 searchWeb 补足），避免无效多步。  
- **为何图内专家用 IBManus 架构**：统一 think→act 多步循环，便于每个专家独立做多轮推理与工具调用，且可注入 RAG/记忆/Trace，与 OpenManus 分层一致，便于维护与扩展。  
- **为何 RAG 用向量+BM25**：专利场景既有语义问法也有专利号/术语精确匹配，双路召回 + RRF + Rerank 可兼顾 Recall 与 token 上限。  
- **为何记忆按需检索**：不自动注入可减少 token 与噪声，由模型在需要时调用 retrieve_history，更可控。  
- **瞬时错误 503/429**：PatentGraphRunner 检测到异常链中含 503、429、high demand 等时重新抛出，便于测试/调用方识别并重试，而非返回通用兜底文案。

---

以上内容覆盖 Agent 主入口、图结构、P&E 与 Replan、每个 agent 的实现方式、串联与状态传递、三层记忆（含 Importance、递归摘要、衰减、Semantic 筛选、NLI）、工具（含 WebSearch 与其余）、提示词、RAG 全链路（清洗、向量化、入库、检索、检索前后）、SSE 与 Advisor（含 TracerAdvisor 与 order/before/after），并补全了面试向深挖点，可直接作为 AI 方向技术文档与面试准备材料使用。
