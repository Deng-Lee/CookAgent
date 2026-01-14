# src/data_process/splitter.py

## 文件职责
负责将菜谱 Markdown 按结构化语义切分为 5 类块，并在每个块前注入统一的元数据前缀，保证后续检索与生成的语义一致性。

## 关键数据结构
- `ChunkCategory`：枚举 5 类结构块（Intro/Ingredients/Operation/Calculation/Additional）。
- `Chunk`：切分后块的载体，包含 `parent_id`、`category`、`content`、`enriched`、`meta`。

## 核心函数与参数
- `_normalize_heading(text)`：去空白并小写，用于提升标题匹配鲁棒性。
- `_match_category(heading)`：用 `CATEGORY_PATTERNS` 规则命中结构分类。
- `_parse_title(lines)`：解析 H1 标题作为菜名，若不存在则回退到首个非空行。
- `_split_sections(lines, start_idx, strict=False, source_path=None)`：
  - 识别 H2 标题并按类别切分段落。
  - `strict=True` 时遇到未知标题直接抛错；否则降级为 Additional 并输出 warn。
- `_build_chunk(dish_name, parent_id, category, lines, source_path)`：
  - 计算 `content`（去空行拼接）。
  - 计算 `enriched`：`[菜名: <name>] [类别: <category>] <content>`。
  - 生成 `meta`（`dish_name`/`category`/`source_path`）。
- `split_markdown(md_text, source_path=None, parent_id=None, strict=False)`：
  - 组织切分总流程：标题解析 → 章节切分 → 生成 `Chunk` 列表。
  - Intro 为首个无类别段落（若存在）。
- `split_file(path, strict=False)`：读取文件后调用 `split_markdown`。

## 计算与规则细节
- 分类命中是“包含式匹配”，例如 `"步骤"` 命中 `Operation`。
- Intro 只取“第一个有内容的未分类段落”，其他未命中段落归入 Additional。
- `enriched` 字段是后续向量嵌入的主文本来源。

## 入口与使用方式
命令行入口：
```
python src/data_process/splitter.py <path> [--strict]
```
输出为 JSON 格式的 `Chunk` 列表。

## 关联关系
- 被 `src/data_process/build_index.py` 调用，用于批量切分并构建向量索引。
- 产出的 `Chunk`/`enriched` 字段是后续 `src/retrieval` 模块检索与生成的上游数据基础。

## 外部依赖
- Python 标准库（argparse/json/re/dataclasses/enum/pathlib）。

