下面给你一套**可直接落地的“两层策略（默认按需取证据 + 兜底整篇）”实现步骤**，严格贴合你现在已有的模块：`parent_lock`、`evidence_set`、`evidence_insufficient`、`run_once`、交互模式（locked 继承）。

我会按**数据结构 → 决策流程 → 日志 → 边界情况**写，保证你能照着做。

---

# 两层策略实现规范（v1）

## 目标

在 `locked parent` 前提下处理追问：

* **Layer 1（默认）**：按问题类型路由到必要 block（ingredients/operation/tips），只喂相关证据
* **Layer 2（兜底）**：当 Layer 1 不确定或证据不足时，升级为“整篇（locked parent 全块）”再回答
* 若仍不足：触发 `evidence_insufficient` → 澄清/换版本

---

# 0) 前置：你需要的最小数据与标注

## 0.1 chunk 元数据必须有

每个 chunk 至少带：

* `chunk_id`
* `parent_id`
* `block_type`：`title | ingredients | operation | tips | other`
* `text`

> 你现在已经能构建 evidence_set，说明这些基本具备；block_type 如果还没有统一枚举，先补齐。

## 0.2 锁定后建议缓存一个 “parent_index”

锁定 parent 后，把该 parent 的 chunks 按 block_type 分组，缓存到 session（或内存）：

```json
{
  "parent_cache": {
    "parent_id": ".../简易红烧肉.md",
    "blocks": {
      "ingredients": ["c1","c2"],
      "operation": ["c3"],
      "tips": ["c4"]
    }
  }
}
```

好处：追问时不需要再查向量库就能快速取证据。

---

# 1) 新增一个入口：`run_followup(query, session, trace_id)`

> 在交互模式下，如果 `parent_lock.status == locked` 且用户不是“换菜/新任务”，就走这个函数。

输出结构建议与你 run_once 一致：

* `state`（通常保持 AUTO_RECOMMEND）
* `lock_status` = locked
* `answer`
* `evidence_set`（本轮实际用到的）
* `routing_info`（新增）

---

# 2) Layer 1：问题路由（Routing）

## 2.1 先做轻量问题分类（不用 LLM）

实现一个函数：

### `classify_query(query) -> {intent, confidence, slots}`

* `intent`：`ASK_INGREDIENTS | ASK_STEPS | ASK_STEP_N | ASK_TIPS | ASK_TIME | ASK_HEAT | ASK_SUBSTITUTION | UNKNOWN`
* `confidence`：0~1
* `slots`：可选（如 step_n=1）

### 2.2 规则示例（MVP 够用）

你可以用关键词/正则，示例：

* **步骤相关**

  * 包含：`怎么做|步骤|流程|做法` → `ASK_STEPS`
  * `第(\d+)步|第一步|下一步|然后` → `ASK_STEP_N` / `ASK_STEPS`
* **原料相关**

  * 包含：`原料|材料|食材|需要什么|用什么` → `ASK_INGREDIENTS`
  * `多少|几克|几勺|用量` → `ASK_INGREDIENTS`（slot=quantity）
* **时间火候**

  * `多久|几分钟|多长时间|炖多久` → `ASK_TIME`
  * `大火|小火|中火|火候` → `ASK_HEAT`
* **替代/可选**

  * `可以不放|能换|替代|没有.*怎么办` → `ASK_SUBSTITUTION`
* **技巧**

  * `注意什么|技巧|为什么|怎么更好吃|避免` → `ASK_TIPS`
* 否则 → `UNKNOWN`

给每条规则一个分值，最高分作为 intent，若最高分 < 阈值（比如 0.4）则 `UNKNOWN`。

---

# 3) Layer 1：按 intent 选择证据块（Evidence Routing）

实现：

### `route_blocks(intent) -> block_plan`

推荐映射（你可直接用）：

| intent                 | 默认选择 blocks                             |
| ---------------------- | --------------------------------------- |
| ASK_INGREDIENTS        | `ingredients` (+ title 可选)              |
| ASK_STEPS / ASK_STEP_N | `operation` (+ tips 可选)                 |
| ASK_TIME / ASK_HEAT    | `operation` (+ tips)                    |
| ASK_SUBSTITUTION       | `ingredients` + `tips` (+ operation 可选) |
| ASK_TIPS               | `tips` (+ operation)                    |
| UNKNOWN                | 先走 Layer 2（直接升级）                        |

### 3.1 构建 Layer1 evidence_set

从缓存的 `parent_cache.blocks` 里取对应 chunk_id，并去重：

```text
evidence_layer1 = chunks in selected blocks
```

### 3.2 Layer1 的 “证据检查”

在进入生成前做一次检查：

* 如果 `selected_blocks` 里某个 block 在该 parent 中不存在
  → `insufficient_reason = missing_block_type`
* 或 evidence_layer1 为空
  → `insufficient_reason = empty_evidence`

此时不要生成，直接进入 Layer 2 升级。

---

# 4) Layer 1：回答生成（建议两段式）

你有两种实现路径，推荐先用**规则抽取 + 可选 LLM 改写**。

## 4.1 规则抽取（强推荐，尤其是 “第N步”）

* 对 `operation` 文本做 `parse_steps()` 得到 `steps[]`
* 对 `ingredients` 文本做 `parse_ingredients()` 得到 `items[]`

然后：

* `ASK_STEP_N`：直接返回 steps[n-1]
* `ASK_STEPS`：返回 steps 的前 3 步 + “继续问下一步”
* `ASK_INGREDIENTS`：列出 items
* `ASK_TIME/ASK_HEAT`：在 operation/tips 中做关键词匹配抽句子（找“分钟/火”）

## 4.2 可选：LLM 仅做“改写”

把抽取到的内容作为“唯一事实”，让 LLM润色成更自然的中文。

> 关键：LLM 输入只包含你抽取的短文本，而不是整篇。

---

# 5) Layer 2：兜底升级（全块证据）

触发条件（任一即可）：

1. `intent == UNKNOWN` 或 `confidence < intent_min`（比如 0.5）
2. Layer1 evidence 为空 / 缺块
3. Layer1 回答后仍不足（比如 ASK_TIME 没找到任何“分钟/时间”句子）

升级动作：

### 5.1 选择证据范围：locked parent 全块

```text
evidence_layer2 = all chunks of locked parent (dedup by chunk_id)
```

### 5.2 再做一次 evidence_check

* 对于特定 intent（如 ASK_STEPS），仍要求 operation 存在
* 若全块仍缺关键 block → `evidence_insufficient`（真正触发）

### 5.3 Layer2 生成策略

这里你可以允许 LLM 看到更多上下文（整篇），但要加两条硬约束：

* 只能基于提供文本回答
* 没写就说没写/建议换版本

并且你仍然记录：

* `upgrade_to_full_parent = true`

---

# 6) Layer 2 仍不足：evidence_insufficient 的动作（MVP）

当 Layer2 仍不足（如菜谱完全没写“焯水几分钟”）：

1. 写 `evidence_insufficient` 日志
2. 输出：

   * “该版本菜谱未说明 X”
   * 给出选项：

     * “要不要换一个版本（Top2/Top3）？”
     * 或提一个澄清问题（你更关心时间还是火候？）

> 注意：不要“凭常识给分钟数”，除非你明确标注“非菜谱内容/常见经验”，否则会破坏 Evidence-Driven 的可信性。

---

# 7) 必须加的日志字段（用于观测与调优）

在 `evidence_driven.log` 或新 `routing.log` 中记录：

### `evidence_routing`

* trace_id, turn
* intent, confidence
* selected_blocks_layer1
* evidence_chunk_ids_layer1
* upgraded_to_layer2: true/false
* evidence_chunk_ids_layer2（如果升级）
* final_evidence_chunk_ids（实际用于生成的）

### `generation_started/completed` 里也带上：

* `routing.intent`
* `routing.layer_used`（1 or 2）

这样你以后可以统计：

* 有多少 query 走了升级
* 升级是否真的提升了回答命中率
* 哪些 intent 分类经常失败（该补规则）

