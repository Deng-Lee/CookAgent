# src/data_process/build_index.py

## 文件职责
批量遍历菜谱 Markdown，调用结构化切分器，将富化文本嵌入向量后写入 Chroma 向量库。

## 核心函数与参数
- `resolve_local_model_path()`：
  - 使用 `LOCAL_BGE_M3_PATH` 环境变量优先定位本地模型。
  - 否则从 HuggingFace 缓存目录中选择最新快照。
  - 找不到则抛出 `FileNotFoundError`。
- `iter_markdown_files(root)`：递归枚举 `root` 下所有 `.md` 文件。
- `chunk_records(path, strict)`：
  - 调用 `split_file` 获取结构化 `Chunk`。
  - **插入 Title chunk**（菜名标题），提升菜名召回。
  - 生成 `records` 列表：`id` / `text` / `metadata`。
- `build_collection(root, db_path, collection_name, strict=False)`：
  - 初始化 `SentenceTransformerEmbeddingFunction`（`local_files_only=True`）。
  - 使用 `chromadb.PersistentClient` 创建/获取 collection。
  - 批量 upsert `records`。
- CLI 参数：
  - `--root`：菜谱 Markdown 根目录（默认 `data/raw/cook/dishes`）。
  - `--db-path`：Chroma 持久化路径（默认 `data/chroma`）。
  - `--collection`：collection 名称（默认 `cook_chunks`）。
  - `--strict`：严格模式（未知标题直接失败）。

## 计算与规则细节
- Title chunk 的 `chunk_index` 设为 `-1`，并扩展 `total_chunks` 计数。
- 其他 chunk 的 `id` 使用 `path.stem` + 索引 + `uuid4` 后缀生成。
- 记录的 `text` 使用 `Chunk.enriched`，用于降低切分导致的语义漂移。

## 入口与使用方式
```
python src/data_process/build_index.py --root <recipes> --db-path <chroma> --collection <name> [--strict]
```

## 关联关系
- 依赖 `src/data_process/splitter.py` 的 `split_file` 和 `Chunk` 数据结构。
- 产出的向量库由 `src/retrieval/parent_retriever.py` 与 `src/retrieval/evidence_utils.py` 查询。

## 外部依赖
- `chromadb`
- `sentence-transformers`

