# src/retrieval/logging_utils.py

## 文件职责
为检索、证据与生成流程记录 JSONL 日志，形成可追踪的运行轨迹。

## 核心函数与参数
- `log_retrieval(query, res, parents, log_path=logs/retriever.log)`：
  - 记录向量检索 topk、父文档候选聚合分数、命中片段摘要。
- `log_parent_lock(query, parent_lock, parents, log_path=DEFAULT_LOCK_LOG)`：
  - 记录锁定状态（locked/pending/none）及候选摘要。
- `log_evidence_built(query, evidence_set, log_path=DEFAULT_EVIDENCE_LOG)`：
  - 记录构建的证据集 `chunk_id` 列表。
- `log_evidence_insufficient(query, turn, parent_lock, evidence_set, payload, log_path=DEFAULT_EVIDENCE_LOG)`：
  - 记录缺证据场景与决策原因。
- `log_evidence_routing(trace_id, turn, query, routing, log_path=DEFAULT_EVIDENCE_LOG)`：
  - 记录意图路由与层级选择策略。
- `log_generation_started(record, log_path=DEFAULT_GENERATION_LOG)`：
  - 标记生成开始，写入 `event` 与 `ts`。
- `log_generation_completed(record, log_path=DEFAULT_GENERATION_LOG)`：
  - 记录生成结果与输出统计（`char_count`、`preview`）。
- `log_generation_mapping(record, log_path=DEFAULT_GENERATION_LOG)`：
  - 记录生成使用的 chunk 映射策略。

## 计算与规则细节
- 所有日志以 JSON Lines 追加写入，便于按行解析。
- `log_retrieval` 对向量结果去重（按 `chunk_id`）。
- `log_parent_lock` 在 pending 状态下写入候选列表与证据摘要。

## 关联关系
- 依赖 `src/retrieval/retrieval_types.py` 的 `ParentHit`/`ParentLock` 及默认日志路径。
- 被 `src/retrieval/parent_retriever.py` 调用记录检索与生成关键节点。

## 外部依赖
- Python 标准库（datetime/json/pathlib/typing）。

