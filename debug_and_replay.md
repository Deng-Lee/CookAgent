# GitCook-Agent 可观测性规范（二）：Debug Playbook 与回放报告模板

> 本文档定义 **如何使用 Trace 数据进行调试、复盘与归因**，
> 并给出一条请求的 **标准回放报告结构**。

---

## 1. 调试目标

回答三个问题：
1. 系统为什么这样答？
2. 是哪一环做错了？
3. 如何验证修复是否有效？

---

## 2. 故障分类（先定位类型）

1. 意图错误（Router）
2. 召回错误（BM25 / Vector / RRF）
3. Parent 聚合错误
4. HITL / 锁定决策错误
5. Grader 误判
6. Web 注入 / 噪音
7. 生成幻觉 / 引用错误

---

## 3. 标准 Debug 流程

1. 输入 trace_id  
2. 回放完整 Trace  
3. 判断错误类别  
4. 对比“期望路径 vs 实际路径”  
5. 标注根因  
6. 修复 → 回归验证  

---

## 4. 回放报告模板（Markdown）

```markdown
# Trace Replay Report

## 基本信息
- trace_id:
- session_id:
- turn_id:
- user_input:
- intent:

## Router 决策
- intent_pred:
- slots:
- rationale:

## 检索结果
### BM25 TopK
- ...

### Vector TopK
- ...

### RRF + Parent 聚合
- top parents:
- 覆盖情况:

## HITL / Locking
- 是否触发 HITL:
- 锁定状态:

## Grader / Web
- grader_label:
- 是否联网:
- web sources:

## Generator
- response_format:
- citations_used:

## 问题分析
- 错误类型:
- 根因节点:
- 直接原因:

## 修复建议
- 参数 / 策略 / 数据:
- 验证方式:
```

---

## 5. 自动归因规则（示例）

- intent=instruction 但只命中 intro → 检索异常  
- is_locked=true 但引用不同 parent → 锁定污染  
- grader=irrelevant 但命中多个高分 chunk → grader 偏保守  

---

## 6. 回归验证清单

- 同类 query 是否改善  
- 是否引入新回归  
- 成本 / 延迟变化  

---

## 7. 成功标准

- 每个线上问题都有 trace + 回放报告  
- 修复前后有可量化对比  
