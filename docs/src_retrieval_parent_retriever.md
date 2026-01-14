# src/retrieval/parent_retriever.py

## 文件职责
父级检索核心入口：负责向量查询、父文档聚合评分、自动锁定/歧义处理、证据集构建、会话续问与规则化回答生成。

## 核心流程概览
1) 向量检索 chunk 级结果。
2) 以 `parent_id` 聚合并计算 `overall_score`。
3) 根据阈值判定 auto-lock 或进入歧义候选。
4) 构建 evidence set 并输出自动回答或澄清引导。
5) 会话模式下支持锁定后的二次问答与证据分层路由。

## 关键函数与参数
- `_distance_to_score(distance)`：
  - 将 Chroma cosine distance 转换为相似度 `score = 1 / (1 + distance)`。
- `aggregate_hits(res, k=60)`：
  - 将 chunk 结果聚合为 `ParentHit`。
  - 计算字段：
    - `rrf_sum = Σ 1/(k + rank)`
    - `max_chunk_score = max(score)`
    - `coverage = 命中 chunk 数`
    - `coverage_ratio = coverage / total_chunks`
  - 对 coverage 过阈值的 parent 进行归一化并计算 `overall_score`：
    - `rrf_sum_norm` 与 `max_chunk_norm` 归一化到 0..1。
    - `overall_score = 0.7 * rrf_sum_norm + 0.3 * max_chunk_norm`。
- `load_parent_doc(parent_id)`：从路径读取父文档内容。
- `_retrieve_state(...)`：
  - 初始化 embedding 与 Chroma client，执行 `collection.query`。
  - 根据阈值判断：
    - `coverage_threshold=0.5` 过滤低覆盖。
    - `t_min=0.40` 为最低候选分阈值。
    - 若 `top2/top1 > 0.92` 标记为 `good_but_ambiguous`。
  - 根据判定设置 `ParentLock`：`locked` / `pending` / `none`。
  - 可记录 `retrieval` / `lock` / `evidence` 日志。
- `retrieve(...)`：只返回父级候选列表。
- `_build_output_from_state(...)`：
  - 若已锁定且证据充分，调用 `format_auto_answer` 返回全菜谱结构化 Markdown。
  - 若证据不足，返回 `LOW_EVIDENCE` 与 `clarify_question`。
  - 若歧义，返回候选列表（`include_candidates` 控制是否提供 `option_id`）。
- `run_once(...)`：单轮检索入口。
- `run_session_once(...)`：
  - 维护会话锁定、候选选择与缓存（解析的 steps/ingredients）。
  - 在锁定态根据 `classify_query` → `route_blocks` 选择证据层级。
  - 若 layer1 不足则升级到 layer2（全证据集）。
  - 记录 evidence routing 与 generation 日志。

## 计算与判定规则
- 低证据判定：
  - 无任何 `coverage_ratio >= 0.5` 的 parent，或 top1 `overall_score < 0.40`。
- 歧义判定：
  - `top2/top1 > 0.92` 且 top1 分数有效。
- 自动锁定：
  - 通过上述判定且未触发歧义时，锁定 top1。
- 生成层级：
  - `layer1` 只用意图映射的 block；不满足时升级 `layer2`。

## CLI 参数
- `--query`：用户问题（空则进入交互模式）。
- `--db-path`：Chroma 路径。
- `--collection`：collection 名称。
- `--top-k`：chunk 级检索条数。
- `--top-parents`：父级候选数量。
- `--log-path` / `--lock-log-path` / `--evidence-log-path`：日志路径。
- `--turn`：对话轮次。
- `--trace-id` / `--session-id`：追踪与会话标识。

## 关联关系
- 依赖：
  - `src/retrieval/model_utils.py`：模型路径解析。
  - `src/retrieval/evidence_utils.py`：证据构建与自动回答。
  - `src/retrieval/generation_utils.py`：意图识别与局部回答。
  - `src/retrieval/logging_utils.py`：检索/生成日志。
  - `src/retrieval/session_utils.py`：会话锁定与候选选择。
  - `src/retrieval/retrieval_types.py`：结构体与默认日志路径。
- 上游依赖：
  - `src/data_process/build_index.py` 生成的 Chroma collection。

## 外部依赖
- `chromadb`
- `sentence-transformers`

