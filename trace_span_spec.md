# GitCook-Agent 可观测性规范（一）：Trace 与 Span 契约规范

> 本文档定义 **GitCook-Agent** 在运行时必须产出的可观测性数据结构，
> 目标是让 **每一次回答都可解释、可回放、可定位问题来源**。
>
> 本规范 **不涉及代码实现**，只约定“必须记录什么”。

---

## 1. 总体设计原则

1. 一次用户输入 = 一条 Trace  
2. 每个 LangGraph 节点 = 一个 Span  
3. 默认记录“摘要级数据”，出问题可升级采样  
4. 任一 Span 的输出，都必须能解释“为什么做出这个决策”

---

## 2. Trace 顶层结构（Per Turn）

| 字段 | 说明 |
|---|---|
| trace_id | 全局唯一 ID |
| session_id | 会话 ID |
| turn_id | 当前轮次 |
| timestamp | 请求时间 |
| user_input | 用户原始输入（可脱敏） |
| is_locked | 是否处于锁定状态 |
| locked_parent_id | 当前锁定的父文档 |
| final_intent | 本轮最终意图 |
| final_latency_ms | 总耗时 |
| final_cost_estimate | token / 调用 / 联网成本估算 |
| error_flag | 是否出现异常 |

---

## 3. Span 通用字段规范（所有节点必须）

| 字段 | 说明 |
|---|---|
| span_name | 节点名称 |
| start_time / end_time | 起止时间 |
| latency_ms | 节点耗时 |
| inputs_summary | 输入摘要 |
| outputs_summary | 输出摘要 |
| decision | 是否触发分支（hitl / lock / web） |
| confidence | 置信度（如有） |
| warnings | 潜在风险 |
| errors | 异常信息 |

---

## 4. 各节点 Span 输出要求

### 4.1 Intent Router

必须记录：
- intent_pred
- confidence
- slots（口味 / 主材 / 菜名候选）
- query_rewrite（如 QA）
- decision_rationale（极简解释）

---

### 4.2 Retriever（BM25 / Vector / RRF）

#### A. 原始召回

- bm25_topk: chunk_id / parent_id / block_type / rank / score / snippet_80  
- vector_topk: 同上  

#### B. RRF 融合

- rrf_topk: rrf_score / 来源(bm25/vec)

#### C. Parent 聚合

- parent_candidates:
  - parent_id
  - dish_name
  - rrf_sum_score
  - coverage_score
  - hit_block_types
  - top_evidence_snippets

---

### 4.3 Ambiguity Detector / HITL

- need_hitl
- trigger_reason（如 top2_ratio=0.94）
- threshold_version
- hitl_options（卡片字段）

---

### 4.4 Lazy-Locking

- lock_action（lock / unlock / keep）
- locked_parent_id
- locked_profile（摘要）
- lock_source（用户选择 / 显式指定）

---

### 4.5 Grader

- grader_label（relevant / irrelevant / unsure）
- confidence
- grader_inputs_refs（chunk_id 列表）
- need_web_fallback
- fallback_reason

---

### 4.6 Web Fallback

- web_query
- web_sources（url_hash / title / domain / snippet_120）
- web_context_tokens
- injection_filtered（true/false）

---

### 4.7 Generator

- response_format
- citations_used（本地 / 网络）
- safety_flags
- hallucination_guard_triggered

---

## 5. 采样与存储策略（建议）

- 默认：100% 记录摘要  
- 深度采样：1–5% 记录完整 prompt/context  
- 错误强制采样：100%  

---

## 6. 成功标准

- 给定 trace_id，可在 **30 秒内解释完整决策路径**
- 给定错误回答，可定位到 **具体节点与原因**
