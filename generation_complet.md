下面我按你现有架构（`run_once` / `run_session_once`、`parent_locking.log`、`evidence_driven.log`、`generation.md` 两层路由）给你一份**可直接照着实现的 generation_started / generation_completed + mapping 规范**：包含

1. **日志 schema（字段级）**
2. **在流程里的插入点（在哪些函数入口/出口打点）**
3. **mapping 的 MVP 实现方式（按 block_type 映射，不做复杂 citation）**
4. **验收清单**

你可以直接落成 `docs/generation_logging_spec.md`，或者按此写代码。

---

# 1) 新增日志文件与事件类型

建议新建独立文件：`generation.log`（JSONL，一行一个事件）。
理由：retriever / evidence / lock 已经分开了，generation 再独立会更清晰。

## 事件类型（最小集）

* `generation_started`
* `generation_completed`
* `generation_mapping`（可选；也可合并进 completed）
* `generation_refused`（可选；也可用 completed.finish_reason=refused 代替）

我建议 **mapping 独立一条**，因为 mapping 可能比较长，避免 completed 过重；但合并也 OK。

---

# 2) generation_started schema（字段级）

> 触发条件：只有在 **本轮会进入“生成”** 时才写（AUTO 且 locked + evidence_check 通过）。
> AMBIG/LOW 不写 generation_started。

```json
{
  "event": "generation_started",
  "ts": "2026-01-09T17:10:23.123+08:00",
  "trace_id": "sess_cli-12",
  "session_id": "cli_default",
  "turn": 12,

  "mode": "single_turn | session_followup",
  "query": "第一步是什么？",
  "output_intent": "full_recipe | step_n | steps_overview | ingredients_only | tips_only | qa",

  "decision": {
    "state": "AUTO_RECOMMEND",
    "layer_used": 1,
    "intent": "ASK_STEP_N",
    "intent_conf": 0.83,
    "upgraded_to_layer2": false,
    "upgrade_reason": null
  },

  "lock": {
    "status": "locked",
    "parent_id": "data/recipes/红烧肉/简易红烧肉.md",
    "lock_reason": "auto | user_select",
    "lock_score": 0.78,
    "locked_at_turn": 11
  },

  "evidence": {
    "parent_id": "data/recipes/红烧肉/简易红烧肉.md",
    "chunk_ids": ["c_003", "c_004"],
    "block_types": ["operation"],
    "size": 2
  },

  "scoring": {
    "top1_overall_score": 0.78,
    "top2_overall_score": 0.76,
    "ratio12": 0.97
  }
}
```

### 字段说明（你实现时的要点）

* `mode`：区分是 run_once 生成，还是 session follow-up 生成（便于统计）
* `output_intent`：你后面做 routing/模板会需要这个；先用枚举占位即可
* `decision.layer_used`：两层策略里最后使用的层（1 或 2）
* `evidence.block_types`：直接从 evidence_set 汇总去重即可
* `scoring.*`：如果是 follow-up（locked 继承）没有 top2，可允许为 null；单轮 run_once 则尽量写上（方便评估）

---

# 3) generation_completed schema（字段级）

> 触发条件：写在生成结束（成功/拒答/异常）时。
> 必须包含 finish_reason，并记录 latency。

```json
{
  "event": "generation_completed",
  "ts": "2026-01-09T17:10:24.801+08:00",
  "trace_id": "sess_cli-12",
  "session_id": "cli_default",
  "turn": 12,

  "status": "ok | refused | error",
  "finish_reason": "ok | evidence_insufficient | low_evidence | pending | exception",
  "latency_ms": 1678,

  "output": {
    "format": "markdown",
    "sections": ["step_1"],
    "char_count": 128,
    "preview": "第一步：冷水下锅放入五花肉..."
  },

  "evidence": {
    "parent_id": "data/recipes/红烧肉/简易红烧肉.md",
    "chunk_ids": ["c_003", "c_004"]
  },

  "error": {
    "type": null,
    "message": null
  }
}
```

### 说明

* `finish_reason`：把所有“不生成”的原因标准化，后面做统计很方便

  * `pending`：AMBIG 的时候（通常不会写 started，但如果你未来某些场景 started 后发现 pending，也能统一）
  * `evidence_insufficient`：生成前/中证据检查失败
* `output.preview`：只截 100–200 字，避免日志爆炸

---

# 4) generation_mapping schema（字段级，MVP）

> MVP 只做 block_type 级别映射：每个 section 对应哪些 chunk_id。
> 不要求逐句引用。

```json
{
  "event": "generation_mapping",
  "ts": "2026-01-09T17:10:24.802+08:00",
  "trace_id": "sess_cli-12",
  "session_id": "cli_default",
  "turn": 12,

  "mapping_strategy": "by_block_type_v1",
  "sections": [
    { "section": "ingredients", "used_chunk_ids": ["c_001", "c_002"] },
    { "section": "steps",       "used_chunk_ids": ["c_003"] },
    { "section": "tips",        "used_chunk_ids": ["c_004"] }
  ]
}
```

### MVP 映射规则（按你现有结构最容易落地）

* 如果输出包含 `ingredients` 段：映射到 evidence 中所有 `block_type=ingredients` 的 chunk_ids
* `steps` / `step_n`：映射到 `operation`
* `tips`：映射到 `tips`

> 如果某轮 routing 只选了 operation，那么 mapping 里只会有 steps/step_n，这也是合理的。

---

# 5) 打点插入位置（非常具体）

你现在的流程大致是：

* `run_once(query, trace_id)`：检索→聚合→评分→状态→锁定→证据→输出
* `run_session_once(input, session, trace_id)`：处理 pending 选择、locked follow-up、或 new query 调用 run_once

## 5.1 在哪里写 generation_started？

**只在“确定要生成”时写**，也就是：

* state == `AUTO_RECOMMEND`
* lock.status == `locked`
* evidence_set 构建完成
* evidence_check 通过（至少 blocks 不缺）

建议插入点：

* `AnswerComposer.generate(...)` 或你调用“生成器”的入口处
  （比在 run_once 里写更稳，因为 follow-up 也会走同一个生成器）

伪顺序：

1. build evidence_set
2. evidence_check
3. **log generation_started**
4. generate answer
5. mapping
6. **log mapping**
7. **log generation_completed**

## 5.2 在哪里写 generation_completed？

* 在生成器返回后（正常）
* 在 `except` 里（异常）
* 在 evidence_check 失败时（refused / evidence_insufficient）

## 5.3 如何保证 trace_id/turn 一致？

* `run_session_once` 负责生成 `turn` 自增与 `trace_id = session_id + "-" + turn`
* `run_once` 接收 trace_id 透传到底层 logger

---

# 6) 和 evidence_insufficient 的关系（不要重复记，但要关联）

你已经在 `evidence_driven.log` 里有：

* `evidence_built`
* `evidence_insufficient`

建议做法：

* 证据不足时：

  * `evidence_driven.log` 记录 `evidence_insufficient`
  * `generation.log` 记录 `generation_completed(status="refused", finish_reason="evidence_insufficient")`
* 两边都带同一个 `trace_id`，就能串起来

不要在 generation.log 里重复写大量 missing_slots 细节（除非你觉得方便），细节放在 evidence_driven.log 足够。

---

# 7) 验收清单（你加完日志后必须满足）

1. **AUTO_RECOMMEND 成功生成**

* 同一个 trace_id 至少有：

  * `evidence_built`
  * `generation_started`
  * `generation_mapping`（或 completed 内 mapping）
  * `generation_completed(status=ok)`

2. **AMBIGUOUS 不生成**

* 不应出现 `generation_started`
* 可以出现（可选）`generation_completed(status=refused, finish_reason=pending)`
  但我更建议：AMBIG 不写 generation.log，只写 lock/evidence/routing 与候选输出日志

3. **evidence_insufficient**

* 必须出现 `evidence_insufficient`
* 且 generation_completed 的 `finish_reason=evidence_insufficient`

4. **能回放**
   给一个 trace_id，你能回答：

* 用哪个 parent 生成的？
* 用了哪些 chunk？
* 输出有哪些 section？各 section 对应哪些 chunk？
