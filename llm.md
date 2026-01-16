好，我已经把你刚才确认的 **Step 1（LLM 只做润色）+ Step 2（LLM 受控抽取）** 整理成一份**可以直接放进项目仓库的 Markdown 设计文档**。
文档以 **“执行顺序为主线”** 写清楚了**第一步做什么、第二步做什么、什么时候回退、回退到哪里**，不会和你现在的流程/状态机产生冲突。

下面是**完整文档正文**（你可以直接保存为 `llm_integration_spec.md` 或类似名称）。

---

# LLM Integration Spec

**（Step 1：润色 / Step 2：受控抽取 + 完整回退机制）**

## 1. 设计目标

本规范用于在现有 **Evidence-Driven / Parent-Locking / Intent-Aware** 架构中，引入 LLM 作为**增强组件**，而非决策核心。

LLM 的职责被严格限制为两类：

1. **Step 1：润色（Polish）**

   * 仅改善表达质量
   * 不新增事实
   * 不改变系统决策

2. **Step 2：受控抽取（Controlled Extraction）**

   * 从限定的 `evidence_set` 中抽取结构化信息
   * 必须返回可校验的引用
   * 抽取失败可自动回退

**LLM 永远不能：**

* 改变 parent_lock 状态
* 改变 AUTO / AMBIGUOUS / LOW_EVIDENCE 判定
* 越过 evidence_set 编造信息

---

## 2. 总体执行顺序（核心）

所有调用 LLM 的路径，**必须严格遵循以下顺序**：

```
Step 2：受控抽取（JSON + 引用）
        ↓
校验（schema / citation / 越界）
        ↓
构建 answer_draft（系统侧）
        ↓
Step 1：润色（可选）
        ↓
返回最终 answer
```

⚠️ **禁止先润色再抽取**（会破坏引用校验）。

---

## 3. 输入数据规范

### 3.1 evidence_set（统一输入）

```json
{
  "parent_id": "recipes/hongshaorou.md",
  "dish_name": "红烧肉",
  "chunks": [
    {
      "chunk_id": "c_01",
      "block_type": "operation",
      "text": "冷水下锅焯水 5 分钟..."
    }
  ]
}
```

约束：

* `chunk_id` 必须稳定
* `block_type` 必须 canonical（ingredients / operation / tips）
* Step 2 / Step 1 均只能使用此 evidence_set

---

## 4. Step 2：受控抽取（Controlled Extraction）

### 4.1 什么时候执行 Step 2

#### 场景 A：首轮 FULL_RECIPE（parent 已锁定）

* 输入：**完整 evidence_set**
* intent = `FULL_RECIPE`

#### 场景 B：follow-up（parent 已锁定）

* intent 高置信：

  * 输入：Layer1 blocks（按 intent）
* intent 低置信 / Layer1 不足：

  * 输入：完整 evidence_set（Layer2）

---

### 4.2 Step 2 输出 JSON Schema（强制）

```json
{
  "intent": "ASK_TIME",
  "fields": {
    "time_info": [
      {
        "text": "炖 40 分钟",
        "citations": [
          {
            "chunk_id": "c_01",
            "quote": "小火慢炖 40 分钟"
          }
        ]
      }
    ]
  },
  "missing": []
}
```

**强约束：**

* `citations[].chunk_id` ∈ evidence_set
* `quote` 必须能在对应 chunk.text 中直接匹配
* 如果无法抽取，必须填 `missing`

---

### 4.3 Step 2 校验规则（不通过即回退）

以下任一条件成立 → **Step 2 失败**：

1. JSON 解析失败
2. schema 关键字段缺失
3. 引用的 `chunk_id` 不存在
4. `quote` 在 chunk.text 中找不到
5. 明显越界（出现证据中不存在的数字/时间/用量）
6. intent 与输出字段不匹配

---

## 5. Step 2 回退策略（完整）

### 5.1 FULL_RECIPE 回退链

```
Step 2 抽取
  ├─ 成功 → 构建结构化菜谱
  └─ 失败 →
        规则抽取（parse_steps / parse_ingredients）
            ├─ 成功 → 结构化输出
            └─ 失败 →
                  format_auto_answer（原文拼接）
```

如果任务是 FULL_RECIPE 且：

* ingredients 或 operation 缺失
  → 返回 `EVIDENCE_INSUFFICIENT`（而不是 LOW_EVIDENCE）

---

### 5.2 follow-up 回退链（intent-aware）

```
Layer1 Step2
  ├─ 成功 → answer_draft
  └─ 失败 →
        generate_answer (Layer1)
            ├─ 成功 → answer_draft
            └─ 失败 →
                  Layer2 Step2
                      ├─ 成功 → answer_draft
                      └─ 失败 →
                            generate_answer (Layer2)
                                ├─ 成功 → answer_draft
                                └─ 失败 → EVIDENCE_INSUFFICIENT
```

---

## 6. Step 1：润色（Polish）

### 6.1 什么时候执行

* **仅在已有 answer_draft 的情况下**
* FULL_RECIPE / follow-up 均可
* 永远是可选步骤

---

### 6.2 Step 1 输入 / 输出

* 输入：系统生成的 `answer_draft`
* 输出：润色后的自然语言文本

**约束：**

* 不得新增任何事实
* 不得新增数字、时间、用量
* 不得改变步骤顺序或内容含义

---

### 6.3 Step 1 回退策略

触发条件：

* LLM 超时 / 失败
* 输出为空或明显跑题
* 检测到新增事实（简单 regex 即可）

回退行为：

* **直接返回未润色的 answer_draft**

---

## 7. Evidence Sufficiency（与 LLM 的关系）

LLM **不参与** evidence_sufficient 判定。

系统侧规则：

```text
evidence_sufficient(intent) :=
required_blocks(intent) ⊆ available_blocks(evidence_set)
```

* FULL_RECIPE → ingredients + operation
* ASK_TIME → operation 或 tips
* ASK_INGREDIENTS → ingredients
* ASK_STEP_N → operation

LLM 只在 evidence_sufficient 为 true 的路径中工作。

---

## 8. 日志与可观测性（必须）

每次 LLM 调用记录：

```json
{
  "trace_id": "...",
  "stage": "extract | polish",
  "intent": "ASK_TIME",
  "evidence_scope": "layer1 | layer2 | full",
  "llm_called": true,
  "llm_success": false,
  "fallback_used": true,
  "fallback_reason": "bad_citation",
  "fallback_target": "rule_generate"
}
```

这些字段是 eval_runner 和回归测试的基础。

---

## 9. 设计底线（写给未来维护者）

> LLM 是增强模块，不是决策模块。
>
> 抽取失败是正常情况，回退是设计的一部分，而不是异常。

---

## 10. 最小实现建议

新增两个封装函数即可：

* `answer_with_extraction_then_polish(...)`
* `validate_extraction(json_obj, evidence_set)`

不需要改你现有的：

* 检索
* parent locking
* intent routing
* generate_answer / format_auto_answer

