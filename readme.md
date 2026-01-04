
# GitCook-Agent 技术架构设计方案 v2.0（完整版 · 含实现细节）

> 本文档在原《GitCook-Agent 技术架构设计方案 v2.0》基础上，**完整整合了实现层面的详细设计说明**，
> 用于直接指导工程落地（不包含代码）。

---

## 1. 项目概述

GitCook-Agent 是一个基于 Anduin2017/HowToCook 开源仓库的智能烹饪助手。

它并不是一个简单的文档搜索器，而是一个具备“认知能力”的 Agent，能够：

- 理解模糊需求（推荐）
- 执行精确指令（教学）
- 回答原理性问题（知识问答）
- 在本地知识不足时，通过互联网进行自我补全

---

## 2. 系统核心架构（System Architecture）

系统基于 **LangGraph** 构建，采用：

**Router → Retriever → Grader → Generator**  
的循环状态流（Stateful Graph）。

### 核心工作流

1. Input：用户输入  
2. Intent Router：识别意图（推荐 / 教学 / 问答）  
3. Retrieval：混合检索（或直接联网）  
4. Grade / Select：
   - 教学 + 多义性 → HITL 澄清选择
   - 问答 + 本地质量低 → Web Search Fallback
5. Context Locking：锁定特定菜谱父文档
6. Generation：生成结构化回复

---

## 3. 数据处理层（ETL & Indexing）

### 3.1 5 类结构化切分（Structural Splitting）

Markdown 文档被切分为 5 类语义子块：

1. Intro：菜品简介、口味
2. Ingredients：必备原料
3. Operation：步骤 / 做法
4. Calculation：用量 / 时间计算
5. Additional：提示 / 备注 / 原理

### 3.2 上下文增强（Contextual Enrichment）

为避免切片后语义失真，每个子块头部强制注入元数据：

```
[菜名: 红烧肉] [类别: 操作] 大火收汁。
```

### 3.3 父子索引（Parent-Child Indexing）

- Vector Store：存储子块向量（用于检索）
- Doc Store：存储完整父文档（用于生成）
- Metadata：Category / Difficulty / Taste / Main_Ingredients

---

## 4. 运行时逻辑（Runtime Logic）

### 4.1 意图识别与路由（Intent Routing）

| 意图 | 示例 | 策略 |
|---|---|---|
| 推荐 | 周末吃点辣的 | Metadata Filter + Intro |
| 教学 | 红烧肉怎么做 | RRF 混合检索 |
| 问答 | 为什么要炒糖色 | Query Rewrite + Additional |

---

## 5. LangGraph 状态机设计（实现细节）

### 5.1 状态（State）结构

#### 会话级（跨多轮）

- is_locked
- locked_parent_id
- locked_doc_md
- locked_doc_struct
- core_profile

#### 本轮级（每次输入）

- user_input
- intent
- slots
- query_rewrite
- retrieval_results
- parent_groups
- need_hitl
- hitl_options
- user_selection
- grader_label
- grader_confidence
- need_web_fallback
- web_context
- final_context
- response_md

### 5.2 核心节点（Nodes）

1. Input
2. Intent Router
3. Retriever
4. Ambiguity Detector
5. HITL Suspend / Resume
6. Selection Resolver
7. Lazy Locking
8. Grader
9. Web Fallback
10. Context Builder
11. Generator

锁定后，LangGraph 继续循环运行，所有代词与问题默认绑定到锁定父文档。

---

## 6. 混合检索与 HITL 策略（实现细节）

### 6.1 RRF 混合检索

- BM25 TopK：20–50
- Vector TopK：20–50
- RRF 合并 TopK：20–40
- k 常量：60–100

### 6.2 Parent 聚合指标

- rrf_sum_score
- max_chunk_score
- coverage_score
- hit_chunks
- evidence_snippets

### 6.3 HITL 触发条件

- 命中 >1 个 parent
- top2 / top1 ≥ 0.92 或分差 < 8%
- Query 为菜名级、非细节级

### 6.4 选项卡片字段

- 菜名 + 版本
- 难度 / 口味 / 主材 / 耗时
- Intro 摘要
- 关键差异点

---

## 7. Lazy-Locking 后的检索与上下文策略

### 7.1 三层上下文优先级

1. 结构化字段直接回答
2. 锁定父文档子块局部检索
3. 全库检索 / 联网兜底

### 7.2 QA 扩展策略

- 先查 locked Additional
- 再查全库 Additional
- 再触发 Web Fallback

### 7.3 Grader 判定规则

- Relevant：直接回答
- Unsure (<0.6)：全库检索
- Irrelevant：提示切换或联网

### 7.4 自动解锁 / 二次 HITL

- 明确切换菜品 → 解锁
- 请求其他做法 → 二次 HITL
- 代词 / 细节问题 → 保持锁定

---

## 8. Web Fallback 与输出规范

### 联网触发

- 本地无结果
- Grader 判 Irrelevant
- 置信度 < 0.6

### 输出要求

- 强制结构化（原料 / 步骤 / 提示）
- 标注来源（本地 / 网络）
- 与锁定版本保持一致性说明

---

## 9. 输出格式规范

- 推荐：Markdown 表格
- 教学：原料 → 步骤 → 避坑
- 问答：结论 → 原理 → 来源

---

## 10. 技术栈

- LangGraph
- LLM：gpt-4o / deepseek-v3 / qwen2.5
- Embedding：bge-m3
- Vector DB：ChromaDB
- Docs：InMemory / Redis
- Search：Tavily API

---

## 11. 方案总结

该方案在以下三点达到工程级稳定性：

1. **高精度**：父子索引 + 上下文增强
2. **高体验**：Lazy-Locking + HITL
3. **高扩展**：软路由 + 联网兜底

> 本文档可直接作为 GitCook-Agent 的 **系统设计说明书 / 技术蓝图** 使用。
