/**
 * 三层分层记忆架构（企业级）：Working → Experiential → Long-Term。
 * <p>
 * Layer1 Working：内存、滑动窗口、动态剪枝（importance &lt; 阈值剔除）、溢出写 Experiential。
 * Layer2 Experiential：事件摘要 + pgvector，decay_weight 重排。
 * Layer3 Long-Term：NLI 三段论（Recall → Conflict → UPDATE/MERGE），可选 Zettelkasten 原子事实。
 * <p>
 * 子包结构：
 * <ul>
 *   <li>config — 配置（MultiLevelMemoryConfig、MemoryVectorStoreConfig）</li>
 *   <li>model — 数据模型（MemoryTurnRecord）</li>
 *   <li>working — Layer1 工作记忆（WorkingMemoryService）</li>
 *   <li>experiential — Layer2 中期事件摘要（ExperientialMemoryService）</li>
 *   <li>longterm — Layer3 长期语义记忆（LongTermMemoryService）</li>
 *   <li>importance — 重要性评分（ImportanceScorer、PatentDomainImportanceScorer）</li>
 *   <li>compression — 摘要压缩（SummaryCompressor、LlmSummaryCompressor）</li>
 *   <li>nli — NLI 冲突检测（NliConflictDetector、LlmNliConflictDetector）</li>
 *   <li>extraction — 原子事实抽取（AtomicFactExtractor、LlmAtomicFactExtractor）</li>
 *   <li>advisor — 持久化 Advisor（MemoryPersistenceAdvisor）</li>
 *   <li>tool — 按需检索工具（MemoryRetrievalTool）</li>
 * </ul>
 * 配置见 application.yaml：app.memory.working / experiential / long-term。
 */
package com.inovationbehavior.backend.ai.memory;
