下面是一页**与 Parent Locking 同粒度、同工程风格**的规范文档，你可以直接放进 repo，命名例如：

```
docs/evidence_driven_generation_spec.md
```

这份规范的目标只有一个：

> **确保 Agent 的每一句“生成内容”，都能被追溯到明确、受控的证据来源。**

---

# Evidence-Driven Generation 规范（v1）

> **目的**
> 在 Parent Locking 已生效的前提下，
> 强制生成过程“只基于证据、不越权发挥、可回放可审计”，
> 将系统从“像在回答”升级为“知道自己为什么这样回答”。

---

## 1. 核心定义（先统一语言）

### 1.1 什么是 Evidence

* **Evidence = 被允许用于生成的 chunk 集合**
* 在本项目中：

  * Evidence **只能来自 `locked_parent_id`**
  * 每条 Evidence 必须能定位到 `chunk_id`

### 1.2 什么是 Evidence-Driven Generation

> **Evidence-Driven Generation =
> 在生成前显式构造证据集，
> 在生成时严格约束使用范围，
> 在生成后记录“输出 ↔ 证据”的映射关系。**

---

## 2. 前置条件（强依赖）

Evidence-Driven Generation **必须依赖 Parent Locking**。

```text
if parent_lock.status != "locked":
    禁止进入最终生成
```

否则：

* 你无法定义“证据范围”
* 所有证据约束都会失效

---

## 3. 数据结构规范（必须）

### 3.1 Evidence Set（生成前构建）

```json
{
  "evidence_set": {
    "parent_id": "hongshaorou_v3",
    "chunks": [
      {
        "chunk_id": "c_001",
        "block_type": "ingredients",
        "text": "...",
        "score": 0.92
      },
      {
        "chunk_id": "c_004",
        "block_type": "operation",
        "text": "..."
      }
    ]
  }
}
```

### 3.2 证据来源说明

* `chunks` 必须 **全部属于 `locked_parent_id`**
* `score` 可选，仅用于排序/权重，不影响合法性

---

## 4. Evidence 构建规则（生成前）

### 4.1 基本规则（不可违反）

1. **只允许 locked_parent 的 chunk**
2. **chunk_id 去重**
3. **明确 block_type**
4. **证据不足时禁止“编”**

---

### 4.2 常见 Evidence 选择策略（推荐）

#### 教学型输出（菜谱步骤）

* 必须包含：

  * `ingredients`
  * `operation`
* 可选包含：

  * `tips`
  * `intro`

#### 追问型输出（局部）

* 根据问题类型缩小 Evidence：

  * 问原料 → 只取 `ingredients`
  * 问步骤 → 只取 `operation`

> **原则：Evidence 越小，生成越稳。**

---

### 4.3 Evidence 不足的处理（非常重要）

如果：

* 证据集中 **缺失关键 block**
* 或证据文本不包含用户所问信息

**禁止直接生成答案。**

允许的动作：

* 明确告知“不在该菜谱中”
* 建议用户切换版本 / 解除 Lock
* 请求澄清

---

## 5. 生成约束规则（生成时）

### 5.1 强约束（必须）

* ❌ 禁止引用 Evidence Set 之外的信息
* ❌ 禁止跨 parent 组合步骤
* ❌ 禁止“补全常识步骤”而无证据支持

---

### 5.2 推荐的 Prompt 约束方式（概念层）

生成时应向模型明确传达：

```text
你只能使用以下证据内容回答问题；
如果证据中没有明确说明，请直接说明“该菜谱未提及”。
```

> 你不需要一开始就实现复杂的 citation，只要逻辑上强约束即可。

---

## 6. 输出结构规范（生成后）

### 6.1 输出内容的最小结构要求

每次最终生成应至少包含：

* 明确的结构（如：原料 / 步骤 / 注意事项）
* 避免自由散文式输出
* 每一段落 **可被映射到 1～N 个 chunk**

---

### 6.2 Output ↔ Evidence 映射（必须记录）

即使不展示给用户，也必须在日志中记录：

```json
{
  "generation_map": [
    {
      "output_section": "ingredients",
      "used_chunks": ["c_001", "c_002"]
    },
    {
      "output_section": "step_1",
      "used_chunks": ["c_004"]
    }
  ]
}
```

---

## 7. 多轮对话中的 Evidence 继承规则

### 7.1 默认继承

在 `parent_lock.status = locked` 时：

* 后续追问：

  * 不重建 Evidence 范围
  * 在已有 parent 内 **局部重选 chunk**

---

### 7.2 何时需要重建 Evidence Set

仅当：

1. 用户问题类型变化（步骤 → 原理）
2. 用户显式要求补充说明
3. 当前 Evidence 明显不足

---

## 8. 常见坑（高频 & 高危）

### ❌ 坑 1：生成时偷偷“查全库”

* 非常常见
* 直接破坏 Evidence 体系

### ❌ 坑 2：Evidence 构建正确，生成却没用

* Prompt 没约束
* 模型“自由发挥”

### ❌ 坑 3：证据不足还硬答

* 这是幻觉的主要来源
* 正确做法是：承认不知道

### ❌ 坑 4：混合不同 parent 的 chunk

* 最隐蔽
* 也是最致命的错误

---

## 9. 日志与可观测性（最低要求）

### 9.1 必须有的日志

* `evidence_built`

  * parent_id
  * chunk_ids
* `generation_started`

  * evidence_size
* `generation_completed`

  * output_sections
  * evidence_mapping

---

## 10. 验收标准（你是否真正实现了）

你可以用下面 4 条来自测：

1. 删除一个关键 chunk，生成结果会明显变化或拒答
2. 任意一句生成内容，都能指回至少一个 chunk
3. 用户连续追问，答案始终不跨 parent
4. 你可以回放一次对话，解释**每一步答案来自哪里**

---

## 11. Parent Locking × Evidence-Driven Generation 的关系总结

```text
Parent Locking   → 定义“允许用哪一份文档”
Evidence Set     → 定义“具体用哪些段落”
Generation       → 只能在 Evidence Set 内说话
```

---

## 12. 一句话总结（写给未来维护者）

> **Evidence-Driven Generation 的目标不是“引用看起来很严谨”，
> 而是：
> 当系统犯错时，你能明确说出“它是怎么犯错的”。**

---

如果你愿意，下一步我可以帮你把这两份规范（Parent Locking + Evidence-Driven Generation）**合并成一个完整的 Agent Runtime Flow 图 + 状态表**，这样你这个项目在架构层面就非常完整了。
