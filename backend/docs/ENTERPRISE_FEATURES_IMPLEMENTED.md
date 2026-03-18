# 企业级五项能力实现说明

本文档说明已实现的五项能力及其配置与使用方式。

---

## 1. CRAG 式检索后验证

**位置**：`ai/rag/crag/`

- **CragDecision**：三分支枚举 CORRECT / AMBIGUOUS / INCORRECT。
- **RetrievalQualityEvaluator**：接口，`LlmRetrievalQualityEvaluator` 用 LLM 对「query + 召回文档」打 0～1 分。
- **CragDocumentRetriever**：包装底层检索器，按分数与阈值分支：CORRECT 直接返回文档；AMBIGUOUS 可截断；INCORRECT 返回空，触发现有空上下文策略（如 searchWeb）。

**配置**（`application.yaml`）：

```yaml
app.rag.crag:
  enabled: true
  correct-threshold: 0.7
  ambiguous-threshold: 0.35
  ambiguous-trim-to-half: true
```

**说明**：与现有空上下文策略统一——当 CRAG 判为 INCORRECT 时返回空列表，由 `ContextualQueryAugmenter` 走“知识库无内容”或“先 searchWeb”的模板。

---

## 2. Citation + Trace 审计

**Citation**：在 `IBApp.resolvePrompt()` 中，对 retrieval/analysis/advice 任务追加 `CITATION_INSTRUCTION`，要求回答基于 context 并标注 [1]、[2] 等来源。

**Trace 落库**：

- **TracingDocumentRetriever**：包装 RAG 检索器，将本次 query 与 documents 写入 `RetrievalTraceContext`（ThreadLocal）。
- **PersistingTraceAdvisor**：在每次 Agent 调用结束后，从 context 取 sessionId、从 `RetrievalTraceContext` 取检索结果，与请求/响应/工具调用一并写入 `AgentTraceRepository`。
- **JdbcAgentTraceRepository**：MySQL 表持久化，表结构见 `docs/schema-agent-trace.sql`。
- **GET /api/ai/agent/trace?sessionId=xxx&limit=50**：按会话查询审计记录。

**配置**：

```yaml
app.agent:
  trace-query-limit: 50
  trace-persist: true   # 需先在 MySQL 执行 docs/schema-agent-trace.sql
```

**使用**：启用 `trace-persist` 并建表后，Trace 自动落库；前端或运维通过 `GET /api/ai/agent/trace?sessionId=<chatId>` 查询。

---

## 3. 迭代/多轮 RAG（多查询融合）

**位置**：`ai/rag/preretrieval/MultiQueryExpander`、`LlmMultiQueryExpander`；`ai/rag/retrieval/MultiQueryDocumentRetriever`。

- **MultiQueryExpander**：将一条 query 扩展为多条子查询。
- **LlmMultiQueryExpander**：用 LLM 生成 2～maxQueries 条改写/子问题（含原问）。
- **MultiQueryDocumentRetriever**：对每条子查询调用底层检索，再用 `RrfFusion` 融合结果。

**配置**：

```yaml
app.rag.multi-query:
  enabled: true
  max-queries: 3
```

**说明**：检索链顺序为 多查询(可选) → CRAG(可选) → 租户过滤(可选) → Tracing。

---

## 4. Reflect 与策略更新

**位置**：`ai/reflect/`

- **ReflectiveRuleStore**：存储反思规则；默认实现 **InMemoryReflectiveRuleStore**（按 chatId 保留最近 20 条）。
- **ReflectionService**：根据步骤成败调用 LLM 生成一条简短规则并写入 store；提供 `getRecentRules(chatId, limit)`。
- **ReplanNode**：在“结果不足、整计划重算”时调用 `reflectionService.reflect(chatId, userMessage, stepResults, false)`。
- **IBApp**：`createPlan(..., chatId)` 与 `getPromptForTask()` 注入“Prior lessons”/“Lessons learned”到规划与专家 prompt。

**说明**：无需额外配置；有 `ReflectiveRuleStore`（默认内存实现）即会启用反思与注入。

---

## 5. 文档级访问控制（多租户）

**位置**：`ai/tenant/TenantContext`、`ai/tenant/TenantContextFilter`；`ai/rag/retrieval/TenantAwareDocumentRetriever`。

- **TenantContext**：ThreadLocal 存储当前请求的 tenantId。
- **TenantContextFilter**：当 `app.rag.tenant.enabled=true` 时，从请求头 **X-Tenant-Id** 读取租户 ID 并写入 TenantContext，请求结束清除。
- **TenantAwareDocumentRetriever**：在检索结果上按 `document.metadata.tenant_id` 过滤，仅保留无 tenant_id 或与当前租户一致的文档。
- **DocumentLoader**：可为入库文档设置默认 `tenant_id`（`app.rag.tenant.default-id`），便于单租户或默认租户文档。

**配置**：

```yaml
app.rag.tenant:
  enabled: true
  default-id: "default"   # 当前加载的文档默认归属的租户
```

**使用**：前端或网关在请求头中带 `X-Tenant-Id: <tenantId>`，检索时仅返回该租户可见文档。

---

## 6. HyDE（假设文档嵌入）

**位置**：`ai/rag/hyde/`

- **HyDEDocumentRetriever**：包装向量检索器；检索前用 LLM 根据 query 生成一段「假设性答案」段落，再对该段落做向量检索，使 query 与文档空间对齐，提升召回。

**配置**：

```yaml
app.rag.hyde:
  enabled: true
```

**说明**：仅作用于向量一路；BM25 仍用原始 query。检索链中向量路在 `HybridDocumentRetriever` 内，启用后为「HyDE 向量 + BM25 → RRF → Rerank」。

---

## 7. GraphRAG 风格（实体图检索）

**位置**：`ai/rag/graphrag/`

- **EntityExtractor**：从文本抽取实体（专利号、技术领域、机构、关键概念）；默认 **LlmEntityExtractor**。
- **EntityDocumentIndex**：实体 → 文档索引；**InMemoryEntityDocumentIndex** 从 RAG 语料建索引（启动时对前 `index-max-docs` 篇文档做实体抽取）。
- **GraphRAGDocumentRetriever**：包装主检索器；查询时从 query 抽取实体，用索引召回相关文档，再与主检索结果 RRF 融合。

**配置**：

```yaml
app.rag.graphrag:
  enabled: true
  entity-top-k: 6       # 实体路召回条数
  index-max-docs: 300   # 建索引时最多处理文档数（避免启动过慢）
```

**说明**：轻量级 GraphRAG，无图库与社区摘要；通过实体-文档索引增加一路「实体召回」并与现有混合检索融合。

---

## 配置汇总（application.yaml 片段）

```yaml
app:
  rag:
    hyde:
      enabled: false
    graphrag:
      enabled: false
      entity-top-k: 6
      index-max-docs: 300
    crag:
      enabled: false
      correct-threshold: 0.7
      ambiguous-threshold: 0.35
      ambiguous-trim-to-half: true
    multi-query:
      enabled: false
      max-queries: 3
    tenant:
      enabled: false
      default-id: ""
  agent:
    trace-query-limit: 50
    trace-persist: false
```

按需将上述 `enabled` 设为 `true`，并完成 Trace 建表（见 `docs/schema-agent-trace.sql`）即可使用对应能力。检索链顺序：**多查询 → GraphRAG（实体路）→ CRAG → 租户过滤 → Tracing**；向量路在 Hybrid 内可选 **HyDE**。
