# src/retrieval/generation_utils.py

## 文件职责
提供基于规则的意图识别、证据路由、解析与答案生成逻辑，服务于锁定态的多轮问答。

## 核心函数与参数
- `classify_query(query)`：
  - 通过正则规则判断意图（如 `ASK_STEPS`、`ASK_INGREDIENTS`）。
  - 输出 `intent`/`confidence`/`slots`（如 `step_n`）。
- `route_blocks(intent)`：
  - 将意图映射到候选 block 类型（如 steps -> operation）。
- `normalize_block_type(block_type)`：
  - 统一块类型别名（Intro/Ingredients/Operation/Additional 等）。
- `build_layer_evidence(evidence_set, block_types)`：
  - 从证据集中选出指定 block 类型的子集。
- `parse_steps(texts)`：
  - 从步骤文本中提取条目；支持编号、列表符号或按句号拆分。
- `parse_ingredients(texts)`：
  - 将原料文本解析为条目列表。
- `extract_sentences(texts, keywords)`：
  - 根据关键词抽取相关句子（时间、火候等）。
- `generate_answer(intent, slots, evidence_set, parsed_steps=None, parsed_ingredients=None)`：
  - 基于意图返回结构化回答或缺失字段列表。
- `output_intent_for(intent, slots)`：
  - 将内部意图映射为输出类型（如 `steps_overview`/`qa`）。
- `build_generation_mapping(output_intent, evidence_set)`：
  - 按 block 类型构建输出 section 与使用的 `chunk_id`。

## 计算与规则细节
- 意图识别是基于关键词的“最高分命中”。
- 步骤解析支持 `1.`、`1)`、`1、`、`-`、`*` 等格式。
- `generate_answer` 会优先从 `operation/ingredients/tips` 中抽取，缺失时返回 `missing_slots`。

## 关联关系
- 被 `src/retrieval/parent_retriever.py` 用于路由、解析与回答生成。
- `normalize_block_type` 与 `BLOCK_CANONICAL_TO_ORIG` 被 `src/retrieval/evidence_utils.py` 复用。

## 外部依赖
- Python 标准库（re/typing）。

