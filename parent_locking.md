下面是一页**可以直接放进你 repo 的工程规范文档**。
它是 **Parent Locking 的“最小但不简陋”的标准定义**，覆盖：

* 数据结构（你该存什么）
* 约束规则（系统必须遵守什么）
* 常见坑（90% 的人会踩的）

你可以直接复制为 `docs/parent_locking_spec.md`。

---

# Parent Locking 规范（v1）

> **目的**
> 在一次对话阶段中，明确并强制系统只围绕一个 `parent`（父文档/菜谱版本）进行推理、生成与追问，
> 防止答案漂移、证据混用和不可回放的问题。

---

## 1. 核心定义（先统一概念）

### 1.1 什么是 Parent

* `parent`：一份完整的知识单元

  * 在本项目中 = **一份完整菜谱的一个版本**
* 由多个 `chunk` 组成（5–6 个）

### 1.2 什么是 Parent Locking

> **Parent Locking = 在 session/context 中，
> 显式记录并强制“当前有效 parent_id”，
> 后续所有检索、生成、追问都必须以它为唯一知识来源。**

---

## 2. 数据结构规范（必须）

### 2.1 Session 中的 Lock 结构（最小可用）

```json
{
  "parent_lock": {
    "status": "locked | pending | none",
    "parent_id": "hongshaorou_v3",
    "lock_reason": "auto | user_select",
    "lock_score": 0.78,
    "locked_at_turn": 1
  }
}
```

### 2.2 字段说明

| 字段               | 含义                | 备注                                         |
| ---------------- | ----------------- | ------------------------------------------ |
| `status`         | 锁定状态              | `locked`=已锁；`pending`=等待用户选；`none`=未进入锁定流程 |
| `parent_id`      | 当前锁定的 parent      | `pending/none` 时可为空                        |
| `lock_reason`    | 锁定原因              | `auto`=系统自动；`user_select`=用户选择             |
| `lock_score`     | 锁定时的 OverallScore | 用于 debug / 回放                              |
| `locked_at_turn` | 锁定发生在第几轮          | 非常重要，便于回放                                  |

---

## 3. Lock 状态流转规则（严格）

### 3.1 Lock 状态机

```text
none
 ↓
pending  ←（GOOD_BUT_AMBIGUOUS）
 ↓
locked   ←（AUTO_RECOMMEND / 用户选择）
```

### 3.2 各状态含义

#### `none`

* 尚未进行 parent 决策
* 允许全库检索

#### `pending`

* 已识别多个同样合理的 parent
* **禁止生成最终答案**
* 只允许：

  * 展示候选
  * 等待用户选择

#### `locked`

* parent 已确定
* **后续所有行为必须受约束**

---

## 4. 约束规则（最重要部分）

### 4.1 生成约束（强制）

> **当 `status = locked`：**

* ❌ 禁止使用其他 parent 的 chunk
* ❌ 禁止重新跑全库检索
* ✅ 只允许：

  * 使用 `locked_parent_id` 下的 chunks
  * 或在该 parent 内做局部检索

**否则即为 Bug。**

---

### 4.2 追问约束（强制）

用户追问如：

* “下一步呢？”
* “火要多大？”
* “可以不放冰糖吗？”

**规则：**

* 默认继承当前 `locked_parent`
* 不重新做 parent 决策
* 不重新跑 top-k 检索

---

### 4.3 什么时候可以解除 Lock（必须显式）

只允许在以下情况解除：

1. 用户明确换菜 / 换版本

   * “换个做法”
   * “我想看上海甜口的”
2. 用户问的是**跨 parent 的元问题**

   * “红烧肉和糖醋肉有什么区别？”
3. 系统显式进入新一轮检索流程（新任务）

解除时必须：

```json
{
  "parent_lock": {
    "status": "none",
    "parent_id": null
  }
}
```

---

## 5. Parent Locking 与 HITL 的关系（澄清）

### 5.1 正确关系

* **HITL 只是输入**
* **Parent Locking 才是状态**

```text
HITL 选择结果
   ↓
写入 parent_lock.parent_id
   ↓
状态变更为 locked
```

### 5.2 错误做法（禁止）

❌ 用户选了，但系统没更新 lock
❌ lock 只是 UI 层变量，不参与后续逻辑
❌ lock 记录了，但生成时不校验

---

## 6. 日志与可观测性（最低要求）

### 6.1 必须记录的日志点

* `lock_created`

  * parent_id
  * reason
  * score
* `lock_released`

  * reason
* `generation_parent_id`

  * 每次生成使用的 parent_id

### 6.2 Debug 时必须能回答的问题

* 这句话是基于哪个 parent？
* 用户选了什么？系统有没有真的用？
* 是否发生过 parent 漂移？

---

## 7. 常见坑（非常重要）

### ❌ 坑 1：只在 UI 里“看起来锁了”

* 实际生成仍然用全库检索
* 用户追问时答案突然变味

### ❌ 坑 2：HITL 先做，Lock 后补

* 用户选了，但系统没真正记住
* 选项变成“装饰品”

### ❌ 坑 3：锁了 parent，但 chunk 仍然混用

* 尤其在生成时调用了“全局 top-k”
* 这是最隐蔽、也最危险的错误

### ❌ 坑 4：每一轮对话都重新判定 parent

* 导致锁反复被覆盖
* 用户体验极差

---

## 8. 验收标准（你做完了吗？）

你可以用这 3 条自检：

1. 用户连续追问 5 轮，**答案不换菜、不换版本**
2. 你能在日志里指出：
   “第 3 轮生成基于 parent = X”
3. 用户选了版本后，你**可以 100% 确认系统尊重了这个选择**

---

## 9. 一句话总结（写给未来的你）

> **Parent Locking 不是功能，而是约束；
> 不是为了“记住选择”，
> 而是为了“防止系统背叛选择”。**

---

如果你愿意，下一步我可以帮你把 **Evidence-Driven Generation 的规范**也按同样粒度写出来，那两份文档放在一起，你这个 Agent 的“可信度骨架”就齐了。
