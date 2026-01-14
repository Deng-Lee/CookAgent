# src/retrieval/session_utils.py

## 文件职责
提供会话态读写与超时清理逻辑，用于多轮对话中的父文档锁定与候选选择。

## 核心函数与参数
- `session_path(session_id)`：
  - 将 session_id 规整为安全文件名，位于 `logs/sessions/`。
- `load_session(session_id)`：
  - 若无文件则返回默认会话结构（含 `parent_lock`/`pending_candidates`）。
- `save_session(session)`：
  - 以 JSON 格式写入 session 文件（`ensure_ascii=False`）。
- `purge_expired_pending(session)`：
  - 如果 pending 状态超过 `ttl_seconds`，清空锁定与候选列表。
- `parse_option_id(text, candidates)`：
  - 解析用户输入中的数字选项，校验是否在候选集合中。

## 计算与规则细节
- `session_id` 仅允许字母/数字/`_.-`，其余替换为 `_`。
- 过期判断使用 `time.time()` 与 `updated_at` 差值。

## 关联关系
- 依赖 `src/retrieval/retrieval_types.py` 的 `DEFAULT_SESSION_DIR`。
- 被 `src/retrieval/parent_retriever.py` 调用，实现锁定会话的读写与候选选择。

## 外部依赖
- Python 标准库（json/re/time/pathlib/typing）。

