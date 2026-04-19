# The Causal Traitor v4
## 最终论文与研究总蓝图 v2

> 版本：v2 Draft
> 形成日期：2026-04-20
> 目标：在不针对单一会议做特异性包装的前提下，最大化项目对 ICLR main、NeurIPS main、NeurIPS Evaluations & Datasets Track 的共同竞争力
> 主线原则：问题更尖锐、任务更科学、benchmark 更可信、方法更必要、实验更难反驳

---

## 0. 执行摘要

本项目不再应被理解为“一个复杂的多智能体因果博弈系统”，也不应仅被理解为“一个新的因果推理 benchmark”。升级后的正式定义是：

> **Selective Adversarial Causal Oversight under Information Asymmetry**

核心研究问题不是“LLM 会不会做因果推理”，而是：

> 当攻击者掌握 hidden information，并利用自然语言提出看似合理的因果 claim 时，verifier 是否能够仅依赖 public evidence，做出正确的 selective decision，尤其是在证据不足时执行 wise refusal，而不是 unsafe acceptance？

本蓝图的核心升级如下：

1. 把 `unidentifiable` 从“第三个标签”升级成 **wise refusal** 的正式研究主轴。
2. 把 benchmark 从“纯 synthetic 程序化生成”升级成 **synthetic core + real-grounded subset** 的双层结构。
3. 把 OOD 从“表面变化”升级成 **mechanism OOD / attack-family OOD / paired-flip OOD / context-shift OOD**。
4. 把 verifier 从“能跑的四阶段管线”升级成 **countermodel-grounded selective verification**，并要求其输出可审计的结构化证据。
5. 把实验从“主表 + 补充实验”升级成完整证据链：
   `Main Benchmark -> Leakage Study -> Ablation -> Robustness/OOD -> Real-Grounded -> Human Audit -> Witness Faithfulness`

本项目今后的成功标准不是“系统更复杂”，而是：

- 主问题是否更锋利
- benchmark 是否更可信
- 方法是否更必要
- 结果是否更难被反驳
- 结论边界是否更诚实

---

## 1. 最终定位

### 1.1 一句话定义

本项目研究：

> 在 information asymmetry 下，当攻击者利用 hidden confounding、selection bias、non-identifiability 或说服性话术提出自然语言因果 claim 时，verifier 是否能够仅依据公开证据，区分该 claim 是 `valid`、`invalid`，还是必须做出 `unidentifiable` 的选择性裁决。

### 1.2 项目身份

本项目由三项主贡献构成：

1. **新任务**
   `Selective Adversarial Causal Oversight`
2. **新 benchmark**
   `public-gold partitioned leakage-free benchmark`
3. **新 verifier**
   `Countermodel-Grounded Selective Verification`

其中：

- 任务定义是主问题
- benchmark 是主贡献之一
- verifier 是用来验证该任务与评测协议合理性的关键方法组件

### 1.3 明确降级的内容

以下内容可以保留，但必须退出主论文主线：

- jury 机制
- difficulty 控制
- evolution 机制
- live demo / visualization
- 多轮 agent 戏剧性交互
- DSR / Flow / Arms Race / Game Balance 等旧系统指标

这些内容后续只可放在：

- appendix
- supplemental materials
- system showcase

---

## 2. 最终标题、摘要与贡献写法

### 2.1 标题建议

首选：

**Selective Adversarial Causal Oversight under Information Asymmetry**

方法导向备选：

**Countermodel-Grounded Selective Verification for Adversarial Causal Claims**

benchmark 导向备选：

**A Leakage-Free Benchmark for Selective Adversarial Causal Oversight**

### 2.2 摘要草稿

> Large language models are increasingly used to evaluate causal claims, yet real-world causal oversight is often selective and adversarial: claimants may possess hidden information, exploit non-identifiability, and present persuasive but incomplete reasoning. We formalize this setting as selective adversarial causal oversight, where a verifier must determine whether a natural-language causal claim is valid, invalid, or requires wise refusal under public evidence constraints. We introduce a leakage-free benchmark with strict public-gold partitioning, adversarial hidden-information attacks, persuasion-aware claim variants, and both synthetic and literature-grounded evaluation subsets. We further propose countermodel-grounded selective verification, a verifier that parses claims into identifying assumptions, constructs an assumption ledger, searches for observationally compatible countermodels, and abstains when public evidence cannot uniquely support the claim. Our goal is to show that standard LLM judges, debate systems, and tool-only baselines remain brittle under adversarial causal oversight, while countermodel-grounded verification improves unsafe-acceptance control, wise-refusal quality, calibration, and robustness under mechanism and attack-family shift.

### 2.3 贡献写法

最终贡献建议严格写成四点：

1. 我们提出一个 **selective adversarial causal oversight** 任务，把自然语言因果 claim 审计建模为 information asymmetry 下的选择性决策问题。
2. 我们构建一个 **public-gold partitioned leakage-free benchmark**，显式覆盖 hidden information、non-identifiability、persuasion pressure、mechanism shift 和 real-grounded evaluation。
3. 我们提出 **countermodel-grounded selective verification**，通过 claim parsing、assumption ledger、countermodel search 和 tool-backed adjudication 支撑 unsafe rejection 与 wise refusal。
4. 我们通过 synthetic、real-grounded、human-audited 三层证据展示：现有 judge/debate/tool systems 的系统脆弱性，并证明 countermodel-grounded verification 在 unsafe acceptance、wise refusal、calibration 和 robustness 上更可靠。

---

## 3. 文献定位与新颖性边界

### 3.1 已有工作覆盖的方向

现有工作已经覆盖：

- Pearl 三层 benchmark
- 统计陷阱型 causal benchmark
- semantic shortcut / paired-flip 类 benchmark
- identification 与 estimation 分离评估
- wise refusal / skepticism / sycophancy 诊断
- context-aware social science benchmark
- 可执行 counterfactual 的流程批评
- persuasion attack 对 verification systems 的影响

### 3.2 本项目真正可占据的空白

本项目最合理的创新空白不是“第一个做 causal + LLM + benchmark”，而是：

1. **information asymmetry 下的 causal claim oversight**
2. **把 wise refusal / unidentifiability 正式纳入任务定义**
3. **把 observationally compatible countermodel 作为 claim rejection / refusal 的核心证据**
4. **在 public-gold 严格分区下进行 leakage-free evaluation**

### 3.3 新颖性边界声明

本项目不应声称：

- 我们是第一个 causal benchmark
- 我们是第一个 causal claim verification 工作

本项目可以稳妥声称：

> 我们系统化了一个以 information asymmetry、wise refusal、public-gold partition 和 countermodel-grounded verification 为中心的统一 causal oversight setting。

---

## 4. 最终研究问题与核心假设

### 4.1 核心研究问题

> 在 information asymmetry 下，当 public evidence 不足以唯一支持一个自然语言 causal claim 时，LLM verifier 能否执行正确的 selective decision，尤其是避免 unsafe acceptance，并在必要时做出 wise refusal？

### 4.2 细化子问题

1. 现有 direct judge、CoT、debate、tool-augmented 方法，在 hidden-information causal claim 上是否会系统性过度裁决？
2. 当 claim 在 public evidence 下本来就不可识别时，正确行为是否应当是 wise refusal，而非强行给出 committed verdict？
3. assumption ledger 与 countermodel search 是否能显著降低 unsafe acceptance？
4. 这种改进是否能在 persuasion、mechanism OOD、attack-family OOD 与 context shift 下保持？
5. verifier 输出的 witness 与 explanation 是否对人类评审可信、忠实且有说服力？

### 4.3 核心假设

- **H1**：不允许 abstention 的 verifier 会在 public evidence 不足时系统性 over-commit。
- **H2**：countermodel-grounded verification 能显著降低 unsafe acceptance。
- **H3**：wise refusal 的质量不能由 aggregate accuracy 替代，必须独立评估。
- **H4**：oracle leakage 会显著虚高 causal oversight 的性能。
- **H5**：如果 witness 是真实证据而非装饰，则移除或破坏 witness 后，系统决策与解释质量应显著下降。

---

## 5. 任务定义

### 5.1 任务名称

`Selective Adversarial Causal Oversight`

### 5.2 输入

每个样本包含：

- `public_evidence`
- `claim_text`
- `attacker_rationale`
- `task_metadata`
- `optional_proxy_variables`

### 5.3 输出

系统输出应分为三层：

#### 识别层输出
- `identification_status ∈ {identified, contradicted, underdetermined}`

#### 最终监督层输出
- `final_verdict ∈ {valid, invalid, unidentifiable}`

#### 解释层输出
- `confidence`
- `assumption_ledger`
- `support_witness`
- `countermodel_witness`
- `missing_information_spec`
- `reasoning_summary`

### 5.4 信息分区

#### Attacker 可见
- hidden variables
- gold SCM 或足够强的结构先验
- full data 或足够强的私有背景知识

#### Verifier 可见
- observed data
- optional proxies
- claim text
- attacker rationale / transcript
- tool outputs

#### Verifier 不可见
- gold DAG
- hidden variable identity
- true SCM parameters
- gold label
- evaluator-only annotations

### 5.5 标签定义

#### `valid`
在 public evidence 下，claim 被支持，核心识别假设被支持，且不存在有效 countermodel 使结论翻转。

#### `invalid`
claim 与 public evidence、工具结果、识别条件或强 countermodel witness 明显冲突。

#### `unidentifiable`
在 public evidence 下，存在多个 observationally compatible explanations，或者核心识别假设未被支持，因此不应输出 committed verdict。

### 5.6 为什么必须允许 `unidentifiable`

在 Pearl-style causal reasoning 中，大量问题的正确答案不是 yes/no，而是：

> 现有证据不足以唯一识别。

因此，`unidentifiable` 不应被理解为系统失败，而应被理解为 **wise refusal** 的正确形式化输出。

---

## 6. Benchmark 设计 v2

### 6.1 总体结构

benchmark 分为两层：

1. `Synthetic Core`
2. `Real-Grounded Subset`

### 6.2 Synthetic Core

保留当前程序化生成器，并扩展 family 覆盖范围：

- L1 latent confounding
- L1 selection bias
- L1 proxy disambiguation
- L1 reverse causality
- L2 valid backdoor
- L2 valid IV
- L2 invalid IV
- L2 subgroup / heterogeneity overgeneralization
- L2 frontdoor / partial measurement
- L3 counterfactual ambiguity
- L3 mediation abduction
- L3 monotonicity / cross-world failure

### 6.3 Real-Grounded Subset

新增一个 literature-grounded 或 semi-real 子集：

- 目标规模：150-300 条
- 来源：economics、policy、epidemiology、education、observational medicine
- 每条样本必须包含：
  - source citation
  - public evidence summary
  - visible vs hidden information contract
  - claim text
  - gold label
  - identifying assumptions
  - witness note

### 6.4 攻击设计

保留当前结构性攻击 taxonomy，并新增 persuasion overlay：

- authority pressure
- expert tone pressure
- confidence pressure
- consensus pressure
- concealment of missing information

### 6.5 OOD 设计

主 OOD 设计升级为：

- graph family OOD
- mechanism OOD
- attack-family OOD
- context-shift OOD
- paired-flip OOD

### 6.6 contamination 控制

必须同时满足：

- template randomization
- variable renaming
- hidden semantics randomization
- paraphrase diversification
- family holdout
- attack-family holdout
- fresh generation
- human spot-check
- synthetic 与 real-grounded 分离评估

---

## 7. 方法：Countermodel-Grounded Selective Verification v2

### 7.1 总体结构

最终 verifier 保持四阶段，并要求每阶段产出结构化中间结果：

1. `Claim Parsing`
2. `Assumption Ledger`
3. `Countermodel Search`
4. `Tool-backed Adjudication`

### 7.2 Claim Parsing

输出至少包含：

- query type
- treatment / outcome
- claim polarity
- claim strength
- explicit assumptions
- implied assumptions
- rhetorical strategy
- abstention-risk cues

### 7.3 Assumption Ledger

对每条 claim 的识别前提显式列出，并标记：

- supported
- contradicted
- unresolved

### 7.4 Countermodel Search

countermodel 必须成为正式证据对象，至少输出：

- `countermodel_type`
- `observational_match_score`
- `query_disagreement`
- `triggered_assumptions`
- `countermodel_explanation`
- `verdict_suggestion`

### 7.5 Tool-backed Adjudication

工具只在没有强 countermodel 时进入支持性裁决。
工具结果必须输出结构化 trace，而不是单一 heuristic score。

### 7.6 最终决策规则

建议固定为：

1. 强反证或强 countermodel -> `invalid`
2. 观测等价但 query 不唯一 -> `unidentifiable`
3. 核心识别假设 unresolved -> `unidentifiable`
4. 无有效反例且识别条件与工具证据共同支持 -> `valid`

### 7.7 非目标声明

verifier 的任务不是恢复宇宙真理，也不是恢复唯一真实 SCM。
其任务是：

> 在当前 public evidence 下，审计 claim 是否应被支持、反驳，或拒绝承诺。

---

## 8. 理论部分的最低目标

### Proposition 1：Abstention Necessity

若存在两个与 public evidence 相容的 SCM 对目标 query 给出不同答案，则任何不允许 abstention 的 verifier 都无法在该类样本上保持普遍可靠。

### Proposition 2：Countermodel Witness Soundness

若 verifier 构造出一个与 public evidence 一致、但对目标 query 给出相反结论的 countermodel，则原 claim 不能被当前 public evidence 唯一支持。

### Proposition 3：Oracle Leakage Inflation

若 verifier 可以访问 gold-only information，则其性能不再代表真实 selective causal oversight 能力。

### 理论边界

理论部分不追求大而全定理，但必须保证：

- 命题正确
- 边界清楚
- 与实验结论一致
- 不夸大 claim

---

## 9. 指标体系 v2

### 9.1 Primary Metrics

- `unsafe_acceptance_rate`
- `wise_refusal_recall`
- `wise_refusal_precision`
- `over_refusal_rate`

### 9.2 Core Metrics

- `verdict_accuracy`
- `macro_f1`
- `ECE`
- `Brier`

### 9.3 Method-specific Metrics

- `countermodel_coverage`
- `identification_stage_accuracy`
- `witness_faithfulness_score`

### 9.4 Human Audit Metrics

- gold label reasonableness
- verifier label reasonableness
- witness persuasiveness
- explanation faithfulness
- Cohen’s kappa / agreement

### 9.5 说明

正式论文主表不应依赖单一加权总分。整体分数可保留为内部开发指标，但主文必须分层汇报主要指标。

---

## 10. 实验矩阵 v2

### Exp 1：Main Benchmark

系统比较：

- direct judge
- CoT judge
- self-consistency judge
- tool baseline
- debate baseline
- refusal-aware baseline
- countermodel-grounded verifier

### Exp 2：Leakage Study

比较：

- clean public partition
- oracle-leaking partition

### Exp 3：Component Ablation

比较：

- no ledger
- no countermodel
- no abstention
- no tools
- no witness

### Exp 4：Persuasion Robustness

比较：

- no pressure
- authority pressure
- confidence pressure
- consensus pressure
- concealment pressure

### Exp 5：OOD Suite

比较：

- graph family OOD
- mechanism OOD
- attack-family OOD
- context-shift OOD
- paired-flip OOD

### Exp 6：Real-Grounded Subset

目的：

- 验证结论不只成立于 synthetic generator

### Exp 7：Human Audit

目的：

- 验证标签、witness 和 explanation 的可信度

### Exp 8：Witness Faithfulness / Necessity

目的：

- 验证 witness 是否是正式证据，而非后处理装饰

### Exp 9：Cross-Model Transfer

定位：

- supplemental

---

## 11. 内部 go / no-go 标准

以下标准用于内部冻结实验，而不是对外声称：

- 若主系统在 IID/OOD 上接近完美表现，则 benchmark 仍过易，必须先加难度。
- 若所有 OOD bucket 的 gap 接近 0，则 OOD 设计仍不足。
- 若 leakage study 不能显著抬高泄漏条件表现，则信息分区设计存在问题。
- 若 human audit 中 gold label reasonableness 偏低，必须先修 benchmark。
- 若 wise refusal recall 高但 precision 低，则 refusal 机制不可用。
- 若 strongest baseline 与主方法差距只体现在一个弱指标上，则主论点不够稳。

---

## 12. 最终交付物

必须交付：

- benchmark generator v2
- real-grounded subset
- verifier pipeline v2
- main experiments
- human audit results
- paper draft
- reproducibility package

可选加分：

- cross-model transfer
- demo / visualization
- jury appendix
- evolution appendix

---

## 13. 最终不做什么

- 不做“大而全的多智能体平台”主线
- 不把 demo 当主贡献
- 不把 UI 当主贡献
- 不把旧系统指标拉回主表
- 不夸大新颖性
- 不在缺乏真实证据时写过强结论

---

## 14. 最终判词

本项目升级后的正确方向不是：

> 做一个更复杂的系统

而是：

> 用一个更严格的问题定义、一个更可信的 benchmark、一个更必要的方法机制和一条更难被反驳的证据链，构成一篇真正的 selective causal oversight 论文。
