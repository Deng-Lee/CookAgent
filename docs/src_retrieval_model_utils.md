# src/retrieval/model_utils.py

## 文件职责
统一解析本地嵌入模型路径，避免联网下载，供检索与证据构建模块复用。

## 核心函数与参数
- `resolve_local_model_path()`：
  - 优先读取环境变量 `LOCAL_BGE_M3_PATH`。
  - 若无，则检查 HuggingFace 缓存目录 `~/.cache/huggingface/hub/models--BAAI--bge-m3`。
  - 在 `snapshots/` 中选择最近修改且包含 `config.json` 的快照。
  - 最后尝试根目录是否含 `config.json`。

## 计算与规则细节
- 快照选择按 `stat().st_mtime` 降序排列取第一个。
- 若无法定位模型，抛出 `FileNotFoundError` 提示配置环境变量。

## 关联关系
- 被 `src/retrieval/parent_retriever.py`、`src/retrieval/evidence_utils.py` 调用。
- 与 `src/data_process/build_index.py` 中的同名逻辑一致，但位于检索侧。

## 外部依赖
- Python 标准库（os/pathlib）。

