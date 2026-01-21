{
  "id": "EVAL_001",
  "query": "推荐几个素菜",
  "expected": {
    "state": "AUTO_RECOMMEND",
    "category": "素菜",
    "parent_ids": ["dishes/素菜/xxx.md"],
    "allow_multiple": true
  },
  "notes": "明确单类目，多解允许"
}

| 字段                  | 含义                       |
| ------------------- | ------------------------ |
| query               | 用户输入                     |
| expected.state      | 期望状态（AUTO / AMBIG / LOW） |
| expected.category   | 若命中类目，期望类目               |
| expected.parent_ids | 允许的 parent 集合            |
| allow_multiple      | 是否允许返回多个 parent          |
| notes               | 人类备注，不参与计算               |

好，下面是一份**可以直接落盘使用的「40 条 Eval 样本完整模板清单」**。
我严格**只基于你当前定版架构**，不引入任何历史方案或未来扩展。

你可以把这份当成 `eval_set_v1.jsonl / yaml` 的**蓝本**。

---

# CookAgent · Eval Set v1（40 条完整模板清单）

> 设计目标：
> **覆盖所有状态分支 + 所有“不可接受错误” + 阈值边界**

---

## A 类：明确单解（AUTO_RECOMMEND）— 16 条

### A1–A8：经典菜名（强单解）

```text
A01  红烧肉怎么做
A02  宫保鸡丁的做法
A03  鱼香肉丝怎么做
A04  西红柿炒鸡蛋步骤
A05  清蒸鲈鱼怎么做
A06  麻婆豆腐的做法
A07  回锅肉步骤
A08  可乐鸡翅怎么做
```

**期望**

* state = AUTO_RECOMMEND
* allow_multiple = false
* 锁定唯一 parent

---

### A9–A16：明确类目 + 明确内容

```text
A09  素菜 麻婆豆腐 怎么做
A10  荤菜 红烧肉 做法
A11  水产 清蒸鲈鱼 怎么做
A12  汤 番茄鸡蛋汤 怎么做
A13  早餐 鸡蛋饼 怎么做
A14  主食 炒饭 做法
A15  甜点 红豆沙 怎么做
A16  饮料 酸梅汤 怎么做
```

**期望**

* state = AUTO_RECOMMEND
* category 硬过滤生效
* 不混类目

---

## B 类：多解歧义（GOOD_BUT_AMBIGUOUS）— 10 条

### B1–B6：泛化推荐（同类多菜）

```text
B01  推荐几个素菜
B02  推荐几道荤菜
B03  推荐几种家常汤
B04  推荐几道早餐
B05  推荐一些主食
B06  推荐几个甜点
```

**期望**

* state = GOOD_BUT_AMBIGUOUS
* allow_multiple = true
* 返回候选 parent 列表
* 不自动锁定

---

### B7–B10：同菜名多版本

```text
B07  红烧肉怎么做（多种做法）
B08  麻婆豆腐的不同做法
B09  炒饭有哪些做法
B10  酸辣汤的家常做法
```

**期望**

* state = GOOD_BUT_AMBIGUOUS
* 触发 score2/score1 分支
* 进入 pending

---

## C 类：类目冲突（必须澄清）— 6 条

> **核心验证点：目录类目硬过滤 + 多命中澄清**

```text
C01  推荐点素汤
C02  早餐 主食 有哪些
C03  素菜 荤菜 推荐几个
C04  汤 饮料 推荐点
C05  水产 荤菜 怎么做
C06  甜点 饮料 有什么推荐
```

**期望**

* state = GOOD_BUT_AMBIGUOUS
* 不返回 parent
* pending_categories ⊇ 命中类目集合
* 不自动组合、不猜

---

## D 类：证据不足（LOW_EVIDENCE）— 8 条

### D1–D4：语料外 / 明显无匹配

```text
D01  推荐几道非洲传统菜
D02  做一道火星料理
D03  适合宇航员吃的菜谱
D04  古罗马宫廷菜怎么做
```

**期望**

* state = LOW_EVIDENCE
* 无 parent_lock
* 返回澄清或拒答

---

### D5–D8：过度抽象 / 无法定位

```text
D05  做点好吃的
D06  给我点灵感
D07  随便推荐一个
D08  今天吃什么好
```

**期望**

* state = LOW_EVIDENCE
* 不 AUTO
* 不进入 pending

---

## E 类：边界 & 回归测试（8 条）

### E1–E4：默认排除类目验证

```text
E01  推荐几个示例菜
E02  调料怎么做
E03  半成品菜谱
E04  有哪些调料
```

**期望**

* 若未显式允许：

  * state = LOW_EVIDENCE
* 不返回正常菜谱 parent

---

### E5–E8：类目词映射回归

```text
E05  肉菜推荐几个        → 荤菜
E06  无肉的菜有哪些      → 素菜
E07  海鲜怎么做          → 水产
E08  喝的有什么推荐      → 饮料
```

**期望**

* 类目映射正确
* 不混目录
* state = GOOD_BUT_AMBIGUOUS 或 AUTO（取决于候选数）

---

## 总量核对

| 类型      | 数量       |
| ------- | -------- |
| A AUTO  | 16       |
| B AMBIG | 10       |
| C 冲突    | 6        |
| D LOW   | 8        |
| **总计**  | **40 条** |
