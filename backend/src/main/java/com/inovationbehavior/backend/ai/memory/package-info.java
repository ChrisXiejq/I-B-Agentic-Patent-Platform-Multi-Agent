/**
 * 分层记忆架构（企业级）
 * <p>
 * 短期记忆：滑动窗口 + 摘要压缩管理上下文；importance 由领域 NER（专利/关键词）加权并随轮次衰减。
 * 长期记忆：按向量（BGE/当前 Embedding）相似度 + importance 阈值筛选入库，结合 NLI 冲突检测控制写入。
 * <p>
 * 主要组件：
 * <ul>
 *   <li>{@link com.inovationbehavior.backend.ai.memory.ShortTermMemoryService} 短期：窗口 + 摘要 + 衰减</li>
 *   <li>{@link com.inovationbehavior.backend.ai.memory.LongTermMemoryService} 长期：检索 + 条件写入（阈值/去重/NLI）</li>
 *   <li>{@link com.inovationbehavior.backend.ai.memory.LayeredMemoryAdvisor} 统一 Advisor，优先使用</li>
 *   <li>{@link com.inovationbehavior.backend.ai.memory.ImportanceScorer} / {@link com.inovationbehavior.backend.ai.memory.PatentDomainImportanceScorer} 重要性评分</li>
 *   <li>{@link com.inovationbehavior.backend.ai.memory.NliConflictDetector} / {@link com.inovationbehavior.backend.ai.memory.LlmNliConflictDetector} NLI 冲突检测</li>
 * </ul>
 * 配置见 application.yaml：app.memory.short-term / long-term / importance。
 */

package com.inovationbehavior.backend.ai.memory;
