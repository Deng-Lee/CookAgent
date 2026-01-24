**直接给结论（统一方案，可一次性解决你提到的两类 case）：**

---

## 统一方案：**“多解意图优先 + 单解证据抑制”双门槛机制**

在 **状态判定阶段（AUTO / AMBIG / LOW）** 增加 **一个意图级 gate + 一个证据级 gate**，不改检索、不改聚合。

---

## 一、意图级 Gate（先判“是否允许单解”）

### ✅ 规则 1：**显式多解意图 → 禁止 AUTO_RECOMMEND**

如果 query 命中任一 **多解触发词**，则：

```text
force_allow_multiple = True
```

**多解触发词（直接 hardcode，MVP 即可）：**

* 推荐 / 几个 / 几道 / 几种 / 一些 / 多种 / 有哪些

**效果：**

* 即便 top1 爆高分
* 也 **禁止进入 AUTO_RECOMMEND**
* **直接进入 GOOD_BUT_AMBIGUOUS**

👉 适用于你列的所有：

* 推荐几个素菜
* 推荐几道荤菜
* 推荐几种家常汤
* 推荐一些主食
* 红烧肉怎么做（多种做法）
* 炒饭有哪些做法

---

## 二、证据级 Gate（防“覆盖率劫持”）

### ✅ 规则 2：**单 parent 过度集中 → 强制多解**

如果满足以下全部条件：

```text
top1.coverage_ratio == 1.0
AND top1.coverage_raw >= COVERAGE_FULL_MIN   # 建议 >=5
AND top2.overall_score >= MULTI_SCORE_MIN    # 建议 >=0.45
```

则：

```text
force_allow_multiple = True
```

**推荐起始值：**

```text
COVERAGE_FULL_MIN = 5
MULTI_SCORE_MIN   = 0.45
```

**效果：**

* 防止“一个菜谱 chunk 太全 → 碾压其它同类做法”
* 红烧肉 / 炒饭 / 面食 等多做法菜 → 自动进入 AMBIG

---

## 三、最终状态决策（统一出口）

```python
if force_allow_multiple:
    state = GOOD_BUT_AMBIGUOUS
elif low_evidence:
    state = LOW_EVIDENCE
else:
    state = AUTO_RECOMMEND
```
