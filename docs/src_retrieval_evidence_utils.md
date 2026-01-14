# src/retrieval/evidence_utils.py

## 文件职责
从检索结果或指定父文档构建“证据集”，并提供证据充分性判断与自动化输出模板。

## 核心函数与参数
- `build_evidence_set(parent: ParentHit)`：
  - 从 `ParentHit.hits` 去重组装证据集（以 `chunk_id` 去重）。
  - 输出包含 `parent_id` 与 `chunks` 列表。
- `build_evidence_set_for_parent(db_path, collection_name, parent_id)`：
  - 直接从 Chroma collection 拉取某个 parent 的所有 chunk。
  - 适配会话锁定后的补充检索。
- `evidence_sufficient(evidence_set)`：
  - 判断是否包含 `Ingredients` 与 `Operation` 两类块。
  - 返回 `(bool, missing_block_types)`，缺失类型以原始块名输出。
- `format_auto_answer(evidence_set)`：
  - 将证据集按 block 类型聚合成“原料/步骤/注意事项”三段式 Markdown。
- `extract_first_step(evidence_set)`：
  - 从 `Operation` 块中抽取首条步骤，用于快速摘要。
- `default_clarify_question(query)`：
  - 生成缺证据时的澄清问题。

## 计算与规则细节
- 证据去重使用 `chunk_id`，确保同一块不会重复输出。
- `evidence_sufficient` 使用 `normalize_block_type` 统一类型大小写与别名。
- `format_auto_answer` 按固定顺序输出 section，缺失的 section 会被跳过。

## 关联关系
- 调用 `src/retrieval/model_utils.py` 获取本地模型路径。
- 调用 `src/retrieval/generation_utils.py` 的 `normalize_block_type`、`BLOCK_CANONICAL_TO_ORIG`。
- 被 `src/retrieval/parent_retriever.py` 使用以构建证据与自动回答。

## 外部依赖
- `chromadb`
- `sentence-transformers`

