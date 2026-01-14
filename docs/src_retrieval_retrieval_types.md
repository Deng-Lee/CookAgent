# src/retrieval/retrieval_types.py

## 文件职责
定义检索与锁定流程的核心数据结构与默认日志路径常量，作为模块间的共享类型协议。

## 核心数据结构
- `ChunkHit`：
  - 单个向量检索命中块（`score`/`rank`/`text`/`metadata`）。
- `ParentHit`：
  - 同一父文档聚合结果。
  - 计算字段：`rrf_sum`、`max_chunk_score`、`coverage_ratio`、`overall_score`。
  - 标志位：`low_evidence`、`good_but_ambiguous`、`auto_recommend`。
- `ParentLock`：
  - 锁定状态（`locked`/`pending`/`none`）及锁定理由与分值。
- `RetrievalState`：
  - 组合结果：`parents` + `parent_lock` + `evidence_set`。

## 常量
- `DEFAULT_LOCK_LOG`：`logs/parent_locking.log`
- `DEFAULT_EVIDENCE_LOG`：`logs/evidence_driven.log`
- `DEFAULT_GENERATION_LOG`：`logs/generation.log`
- `DEFAULT_SESSION_DIR`：`logs/sessions`

## 关联关系
- 被 `src/retrieval/parent_retriever.py`、`src/retrieval/logging_utils.py`、`src/retrieval/session_utils.py`、`src/retrieval/evidence_utils.py` 引用。
- 用于保证检索、锁定、日志记录的字段口径一致。

## 外部依赖
- Python 标准库（dataclasses/pathlib/typing）。

