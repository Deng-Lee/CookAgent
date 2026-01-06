# GitCook-Agent 决策规范：文档整体性评分（OverallScore）与检索置信（Retrieval Confidence）

> 目的：将检索阶段已有的聚合特征（`rrf_sum / max_score / coverage / total_chunks / coverage_ratio`）
> 组合成一个**可解释、可校准、可版本化**的“文档整体性评分”，并据此输出“检索置信等级”，
> 用于后续 **HITL（多义性澄清）** 与 **联网兜底（Web Fallback）** 的决策输入。  
>
> 本规范不包含代码，仅定义字段、计算步骤、阈值策略与输出契约。

---

## 1. 术语与范围

### 1.1 范围
- 适用于：向量/混合检索后，对候选 **父文档（Parent）** 的聚合评分。
- 不用于：对 chunk 级别重新排序（chunk 排序仍由 BM25/Vector/RRF 等负责）。

### 1.2 输入特征（按 Parent 聚合）
对每个候选父文档 `p`，已知：

| 字段 | 含义 |
|---|---|
| rrf_sum | 该 parent 在 RRF 结果中的累计支持强度（多 chunk 支撑） |
| max_score | 该 parent 内最高的单个 chunk 匹配分（单点强证据） |
| coverage | 命中的块类型/章节覆盖度（如 intro/ingredients/operation/additional） |
| total_chunks | 该 parent 的 chunk 总数（文档规模） |
| coverage_ratio | 命中 chunk 数 / total_chunks（证据在文档内的稠密度） |

> 注：若你的 score 是“距离”（越小越好），请在进入本规范前转换为“相似度方向”（越大越好）。

---

## 2. 设计目标与约束

### 2.1 目标
OverallScore 应满足：
1. **单调性**：输入信号变好，OverallScore 不应下降。
2. **可解释**：可拆分为强度/覆盖/稳定性分项。
3. **可校准**：阈值与权重可通过日志/标注集调整。
4. **可版本化**：每次调整参数必须带版本号，便于回归对比。

### 2.2 不追求的目标
- 不追求行业统一“唯一公式”（不存在通用标准）。
- 不使用 LLM 在线计算 OverallScore（成本/稳定性不合适）。

---

## 3. 分项定义（三个正交维度）

OverallScore 由三项构成：

1. **匹配强度（Strength）**：该 parent 是否“像” query
2. **覆盖完整性（Coverage）**：该 parent 是否系统覆盖问题所需信息
3. **稳定性/抗噪（Stability）**：该 parent 是否由少数偶然 chunk “抬上来”

---

## 4. 归一化（Normalization）规范

### 4.1 为什么需要归一化
`rrf_sum/max_score` 的绝对尺度随：
- embedding 模型、距离度量、向量库参数、索引版本
而变化。必须先将其归一到 0–1，才能稳定组合。

### 4.2 推荐归一化方式（在线可用）
采用 **同一轮候选集的分位数归一化**（per-query normalization），对每个 query 的候选 parents 集合计算：

- `norm(x) = clamp((x - P10) / (P90 - P10 + eps), 0, 1)`

其中 `P10/P90` 为该轮候选集的 10/90 分位数。

> 说明：  
> - 比 min-max 更抗异常值  
> - 比 z-score 更稳定易解释

### 4.3 需要归一化的字段
- `rrf_sum_norm = norm(rrf_sum)`
- `max_score_norm = norm(max_score)`
- `coverage_norm = norm(coverage)`（若 coverage 本身是小整数，也可直接除以最大可取值）

### 4.4 不建议直接归一化的字段
- `coverage_ratio` 本身 ∈ [0,1]，无需归一化（可选做平滑）。
- `total_chunks` 建议 log 缩放后再归一化：`log_chunks = log(total_chunks + 1)`

---

## 5. 分项计算公式（建议默认）

### 5.1 Strength（匹配强度）
结合“多证据支持”（rrf_sum）与“强单点证据”（max_score）：

- `S_strength = α * rrf_sum_norm + (1-α) * max_score_norm`

建议默认：`α = 0.6`（偏向多证据更稳）

### 5.2 Coverage（覆盖完整性）
覆盖类型数量 + 命中密度：

- `S_coverage = β * coverage_norm + (1-β) * coverage_ratio`

建议默认：`β = 0.5`

### 5.3 Stability（稳定性/抗噪）
目标：惩罚“超大文档只命中极少 chunk”的偶然抬升，同时避免过度惩罚大文档。

- `log_chunks = log(total_chunks + 1)`
- `log_chunks_norm = norm(log_chunks)`（同轮归一化）
- `S_stability = clamp( coverage_ratio * (0.5 + 0.5 * log_chunks_norm), 0, 1 )`

解释：
- coverage_ratio 是核心
- log_chunks_norm 作为“规模调节项”，让大文档在同等 coverage_ratio 下略更可信（命中不太可能是巧合），但权重不超过 0.5

---

## 6. OverallScore 组合方式（推荐：加权几何平均）

### 6.1 组合公式
使用加权几何平均，避免“单项很差但总分仍被抬高”：

- `OverallScore = (S_strength^w1) * (S_coverage^w2) * (S_stability^w3)`

建议默认权重：
- `w1 = 0.5`（强度最重要）
- `w2 = 0.3`（覆盖其次）
- `w3 = 0.2`（稳定性）

> 若你的场景更偏“教学”（必须步骤/原料齐全），可提高 `w2`。

### 6.2 约束与保护
- 若任一分项为 0，则 OverallScore 为 0：代表“缺少关键证据”
- 可选平滑：`S_* = max(S_*, 0.01)` 防止数值下溢（仅实现层考虑）

---

## 7. 检索置信输出（Retrieval Confidence）

### 7.1 输出对象
对本轮候选 parents（通常 top-N parents）输出：
- `best_parent_id`
- `best_overall_score`
- `confidence_level`
- `risk_flags`
- `thresholds_used`（便于回放与版本化）

### 7.2 置信等级判定（默认策略）
设两个阈值：`T_low`、`T_high`（由校准流程确定，见第 9 节）

- `best_overall_score < T_low` → `confidence_level = "low"`
- `T_low ≤ best_overall_score < T_high` → `confidence_level = "medium"`
- `best_overall_score ≥ T_high` → `confidence_level = "high"`

### 7.3 风险标记（Risk Flags）
用于后续决策与调试解释：

| 规则 | risk_flag |
|---|---|
| S_strength 高但 S_coverage 低 | low_coverage |
| max_score_norm 高但 rrf_sum_norm 低 | single_spike |
| coverage_ratio 极低 | sparse_evidence |
| total_chunks 极大且 coverage_ratio 低 | huge_doc_sparse |
| top2 与 top1 OverallScore 非常接近 | ambiguous_candidate |

> 注：risk_flags 不是错误，仅提示“当前证据形态”

---

## 8. 与后续决策的衔接规范（HITL / Web Fallback）

### 8.1 HITL（多义性澄清）使用“相对置信”
HITL 关注：是否存在多个同样合理的 parent。建议基于 OverallScore 做相对比较：

- 计算：`ratio_p = score2 / score1`（score1 为 top1 parent OverallScore）
- 若 `ratio_p ≥ R_hitl` 且 `top1_parent_id != top2_parent_id` → `need_hitl = true`

建议默认：`R_hitl = 0.92`（需校准）

### 8.2 Web Fallback（联网兜底）使用“绝对置信”
联网触发建议：
- `confidence_level = low` 且无法通过澄清解决（例如 query 很明确但本地无证据）
- 或者检索结果为空

> 重要：低置信 ≠ 必然联网。低置信有一部分应走“澄清”而不是“联网”。

---

## 9. 阈值与权重的校准流程（必须版本化）

### 9.1 最小校准集
- 50–200 条 query（覆盖推荐/教学/问答、宽泛/细节）
- 人工标注：`good / ambiguous / bad`（只标 parent 是否正确即可）

### 9.2 阈值确定（推荐分位数法）
- `T_low`：取 `bad` 样本 best_overall_score 的 P90（让 90% bad 在阈值以下）
- `T_high`：取 `good` 样本 best_overall_score 的 P10（让 90% good 在阈值以上）
- `R_hitl`：取 `ambiguous` 样本 `score2/score1` 的 P50~P70（控制打断频率）

### 9.3 权重调整（消融法）
固定阈值后，依次调整：
- α（Strength 内部权重）
- w1/w2/w3（Overall 组合权重）
观察：
- good/bad 分离度（AUC/简单命中率）
- HITL 触发率与命中率
- 线上成本（如果已上线）

### 9.4 版本化要求
每次调整必须记录：
- `score_policy_version`（例如 `overall_v1`）
- `{α, β, w1, w2, w3, T_low, T_high, R_hitl}`

---

## 10. 输出契约（建议 JSON Schema）

每轮检索输出（供日志/后续节点使用）：

```json
{
  "score_policy_version": "overall_v1",
  "best_parent_id": "P123",
  "best_overall_score": 0.73,
  "confidence_level": "high",
  "thresholds_used": {
    "T_low": 0.35,
    "T_high": 0.68,
    "R_hitl": 0.92
  },
  "top_parents": [
    {
      "parent_id": "P123",
      "overall_score": 0.73,
      "strength": 0.82,
      "coverage": 0.65,
      "stability": 0.71,
      "risk_flags": ["low_coverage"]
    },
    {
      "parent_id": "P456",
      "overall_score": 0.70,
      "strength": 0.79,
      "coverage": 0.67,
      "stability": 0.69,
      "risk_flags": ["ambiguous_candidate"]
    }
  ]
}
```

---

## 11. 与 RAGAS 等离线评估工具的关系（定位）

- 本规范的 OverallScore：**在线决策信号**（快、可解释、低成本）
- RAGAS：**离线评估框架**（LLM-as-judge，评估 context precision/recall、faithfulness 等）

推荐实践：
- 用 OverallScore 做在线 HITL/联网触发
- 用 RAGAS 对不同 `score_policy_version` 的策略进行离线回归验证：
  - high confidence 的样本是否更 faithful
  - low confidence 的样本是否更 likely 需要联网/澄清

---

## 12. 最小落地建议（你当前阶段）

你目前已经有：`rrf_sum/max_score/coverage/total_chunks/coverage_ratio`，因此：

1. 先按本规范计算 `strength/coverage/stability/overall_score`
2. 记录每轮的分项与 risk_flags 到 trace
3. 先不急着拍具体阈值，先收集 1–2 天日志分布，再用第 9 节校准

---

## 13. 附录：常见现象与解释（快速诊断）

- `single_spike`：max_score 很高但 rrf_sum 很低  
  → 单个 chunk 偶然命中，建议提高证据数量门槛或触发澄清

- `low_coverage`：强度高但覆盖低  
  → 对教学风险高（可能只命中 intro），应倾向补检索或澄清

- `huge_doc_sparse`：大文档命中稀疏  
  → 文档结构可能不适合当前切分策略，或需要父文档锁定/二次检索
