# The Causal Traitor v3
## 最终论文与施工总蓝图

> **Superseded for new work as of 2026-04-20.**
> Use [FINAL_CONSTRUCTION_BLUEPRINT_V2.md](C:/Users/njb18/Desktop/causal-traitor/docs/FINAL_CONSTRUCTION_BLUEPRINT_V2.md),
> [ENGINEERING_EXECUTION_PLAN_V2.md](C:/Users/njb18/Desktop/causal-traitor/docs/ENGINEERING_EXECUTION_PLAN_V2.md),
> and [PHASE_TASK_CARDS_V2.md](C:/Users/njb18/Desktop/causal-traitor/docs/PHASE_TASK_CARDS_V2.md)
> as the active source of truth for all new planning, implementation, and paper writing.

> 版本：v3 Final  
> 形成日期：2026-04-16  
> 面向目标：ICML / NeurIPS / ICLR 主会级研究项目  
> 适用范围：论文主线重构、benchmark 设计、方法实现、实验施工、资产复用

---

## 0. 执行摘要

基于当前仓库状态、现有设计文档、已完成实现、以及截至 **2026 年 4 月 16 日** 的相关顶会与前沿 arXiv 文献核查，本项目的最优发展路线已经不应再是：

> 做一个“大而全的多智能体因果欺骗系统”。

而应收缩为：

> **信息不对称下的对抗性因果监督**  
> **Adversarial Causal Oversight under Information Asymmetry**

本文档给出的最终结论是：

1. **当前项目方向仍然足够前沿，也仍然有研究价值。**
2. **原始版本的问题定义过宽，新颖性边界不够安全，必须收缩。**
3. **最值得继续深挖的核心空白，是对抗性因果 claim 的 `valid / invalid / unidentifiable` 三分类验证。**
4. **最值得押注的方法主线，是 `Countermodel-Grounded Verification`，而不是继续扩展 jury / difficulty / evolution。**
5. **现有仓库中有相当多资产可以保留，尤其是 SCM 数据生成、工具执行框架、实验骨架和可视化系统。**
6. **必须立即修复 oracle leakage、标签空间错误、理论定位不稳、统计实验不足等结构性问题。**

本文档的目标不是继续给出分散建议，而是直接提供一份可执行的最终总方案，供后续团队按阶段实施。

---

## 1. 最终定位

### 1.1 最终项目一句话定义

本项目研究：

> 当攻击者掌握隐藏因子、结构先验或不可观测信息，并用自然语言构造看似合理的因果论证时，LLM verifier 是否能够仅依赖可观测证据、代理变量和工具验证，正确判断该 claim 是 **valid、invalid 还是 unidentifiable**，并给出可验证的 witness 或 countermodel。

### 1.2 最终论文定位

本项目最终不再以“游戏系统”或“多角色平台”作为主贡献，而以如下三项贡献构成论文主线：

1. **新任务**
   
   提出 `Adversarial Causal Oversight`：面向对抗性因果 claim 的监督任务。

2. **新 benchmark**
   
   提出一个 leakage-free、fresh、支持 `valid / invalid / unidentifiable` 标签与 witness 标注的 benchmark。

3. **新 verifier**
   
   提出 `Countermodel-Grounded Verification`：通过假设账本、反例模型搜索和工具验证来做因果 claim 审计。

### 1.3 明确不再作为主贡献的内容

以下内容可以保留，但不应进入论文标题、摘要核心贡献和主实验主线：

- 多智能体陪审团机制
- 动态难度控制
- 多轮进化博弈
- 可视化前端
- DSR / Flow / Arms Race / Game Balance 这类系统性指标

这些内容后续应转为：

- appendix
- supplemental demo
- system showcase
- ablation 的次级结论

---

## 2. 最终标题、摘要与贡献写法

### 2.1 最终标题建议

建议主标题采用以下版本之一：

#### 首选

**Adversarial Causal Oversight under Information Asymmetry**

#### 方法导向备选

**Countermodel-Grounded Verification for Adversarial Causal Claims**

#### benchmark 导向备选

**A Leakage-Free Benchmark for Adversarial Causal Claim Verification**

`The Causal Traitor` 建议保留为项目名、系统名或仓库名，而不是主论文标题。

### 2.2 论文摘要草稿

> Large language models are increasingly used to reason about causal claims, yet real-world causal oversight is often adversarial: a claimant may possess hidden information, selectively present assumptions, or exploit non-identifiability while sounding persuasive. We formalize this setting as **Adversarial Causal Oversight**, where a verifier must judge whether a natural-language causal claim is **valid**, **invalid**, or **unidentifiable** using only observable evidence, optional proxy variables, and tool-backed analysis. We introduce a new leakage-free benchmark spanning Pearl's three causal levels, with hidden confounding, selection bias, proxy-based identification, and counterfactual ambiguity. We further propose **Countermodel-Grounded Verification**, a verifier that parses claims into structured assumptions, constructs an assumption ledger, searches for observationally consistent countermodels, and only endorses claims when identification survives explicit adversarial scrutiny. Our empirical goal is to show that standard LLM judges and debate-based systems remain brittle under hidden-information attacks, while countermodel-grounded verification improves robustness, calibration, and unidentifiability awareness.

### 2.3 贡献写法

最终论文中的贡献建议严格压缩为三点：

1. 我们提出 **Adversarial Causal Oversight**，将自然语言因果 claim 审计建模为一个在信息不对称下的三分类监督任务。
2. 我们构建一个 **leakage-free causal oversight benchmark**，覆盖 Pearl 三层、hidden confounding、selection bias、proxy-assisted identification 与 counterfactual ambiguity，并提供 witness / countermodel 标注。
3. 我们提出 **Countermodel-Grounded Verification**，通过 claim parsing、assumption ledger、countermodel search 与 tool-backed adjudication 提升对错误 claim 和不可识别 claim 的鲁棒性。

---

## 3. 文献定位与新颖性边界

### 3.1 已有工作已经覆盖了什么

截至 2026 年 4 月，已有工作大致覆盖了以下方向：

- **Pearl 三层因果 benchmark**：如 CLadder。
- **LLM 因果推理真实性质检验**：如 NeurIPS 2024 对 causal reasoning realism 的分析。
- **multi-agent debate 改进推理**：如 MAD。
- **weak judge / strong agent / oversight**：如 scalable oversight 系列。
- **对抗式 causal debate 初步尝试**：如 CRAwDAD。
- **统计陷阱式因果 benchmark**：如 CausalPitfalls、CausalFlip。
- **latent confounder / proxy variable / selection bias 方向的识别研究**：如 ICML 2024 / ICLR 2025 / ICML 2025 的相关工作。
- **LLM 生成或推断 latent variable**：如 VIGOR+、TLVD。

### 3.2 本项目仍然可能占据的真正空白

根据当前检索结果，本项目最有希望占据的空白不是泛泛的“因果 + LLM + debate”，而是下列更具体的问题：

#### 空白 A：对抗性因果 claim verification

不是做 causal QA，不是做 graph recovery，而是：

> 对一个自然语言因果 claim 进行审计，并在 hidden information 存在的情况下判定其是否成立。

#### 空白 B：显式支持 `unidentifiable`

现有很多 benchmark 仍然隐含要求模型给出单一答案，但真正的因果识别问题中，正确输出经常应该是：

> 当前信息下不可识别。

#### 空白 C：countermodel-grounded verifier

即：

> 用“与观测分布一致但因果答案相反”的 countermodel 作为拒绝 claim 或降级 claim 的核心证据。

#### 空白 D：leakage-free causal oversight benchmark

强调：

- 信息不对称
- strict access control
- hidden confounding
- selection bias
- proxy availability
- fresh split
- witness annotation
- abstention-aware evaluation

### 3.3 新颖性边界声明

这里必须保持严格诚实：

> 本文档中的“新颖性判断”是基于截至 **2026-04-16** 的公开文献检索结果形成的**研究决策结论**，而不是对“绝无同构未公开工作”的数学保证。

因此，本项目后续在正式投稿前，仍需进行一次更系统的 related work 补检与引言压稿。

---

## 4. 最终研究问题

### 4.1 核心研究问题

> Can LLM-based verifiers reliably audit adversarial causal claims under information asymmetry, especially when hidden confounding, selection bias, and non-identifiability make persuasive explanations easy but correct adjudication hard?

### 4.2 细化子问题

1. 现有 direct-judge、CoT、debate-based 和 tool-augmented LLM 方法，在对抗性 hidden-information 因果 claim 上到底有多脆弱？
2. 当 claim 本身不可识别时，现有系统是否会系统性过度裁决？
3. 将 verifier 核心改为 assumption-ledger + countermodel search，是否能显著提高：
   - invalid claim rejection
   - unidentifiable awareness
   - confidence calibration
4. 这些改进是否能在：
   - 新图结构
   - 新语言模板
   - 新模型家族
   - 更强攻击强度
   
   下保持稳定？

### 4.3 核心假设

本文档建议以以下工作假设推进：

> 在对抗性因果监督任务中，仅靠自然语言辩论不足以可靠地区分 valid、invalid 与 unidentifiable claims；必须把显式识别假设与 observationally consistent countermodel search 引入 verifier 才能显著提高鲁棒性。

---

## 5. 任务定义

### 5.1 任务名称

`Adversarial Causal Oversight`

### 5.2 输入

每个样本输入由下列部分组成：

- `D_obs`
  仅可观测数据
- `P`
  可选代理变量 / 额外观测线索
- `c`
  攻击者给出的自然语言 causal claim
- `r_adv`
  攻击者的论证文本或辩论记录
- `m`
  可选元信息，如变量说明、测量语义、任务层级

### 5.3 输出

verifier 必须输出：

- `label ∈ {valid, invalid, unidentifiable}`
- `confidence ∈ [0,1]`
- `assumption_ledger`
- `witness`
  可为 supporting witness 或 countermodel witness
- `reasoning_trace`

### 5.4 信息分区

这是整个项目必须严格落实的关键约束。

#### Attacker 可见

- 完整 gold SCM 或至少足够多的隐藏结构信息
- hidden variables
- full data 或足够强的私有结构先验

#### Verifier / Judge 可见

- observed data
- optional proxies
- claim 文本
- optional debate transcript
- 工具执行结果

#### Verifier / Judge 不可见

- gold true DAG
- hidden variable identities
- true SCM parameters
- full data
- gold label

### 5.5 标签定义

#### `valid`

在当前观测信息与允许使用的辅助信息下，claim 可以被支持，并且不存在已知反例模型使其结论翻转。

#### `invalid`

claim 与数据、识别条件或工具验证明显冲突，或者能构造出明确反证。

#### `unidentifiable`

在当前信息下，存在多个 observationally compatible explanations，且它们对目标因果 query 给出不同答案，因此 claim 不能被唯一支持。

### 5.6 为什么必须允许 `unidentifiable`

因为在 Pearl-style causal reasoning 中，大量问题的正确答案不是 yes/no，而是：

> 当前证据不足以唯一识别。

如果系统不给这个输出空间，就会把问题错误地变成二元分类，导致系统性 over-claiming。

---

## 6. Benchmark 设计

### 6.1 benchmark 的最终目标

不是做“几个好看的场景”，而是做一个：

- fresh
- leakage-free
- identifiable / non-identifiable aware
- adversarial
- tool-compatible
- 可复现实验

的 benchmark。

### 6.2 benchmark 样本 schema

每个样本建议包含以下字段：

```json
{
  "instance_id": "...",
  "causal_level": "L1/L2/L3",
  "graph_family": "...",
  "language_template_id": "...",
  "observed_variables": [...],
  "proxy_variables": [...],
  "selection_mechanism": "...",
  "observed_data_path": "...",
  "claim_text": "...",
  "attacker_rationale": "...",
  "query_type": "...",
  "target_variables": {"treatment": "...", "outcome": "..."},
  "gold_label": "valid/invalid/unidentifiable",
  "gold_answer": "...",
  "gold_assumptions": [...],
  "support_witness": {...},
  "countermodel_witness": {...},
  "meta": {
    "difficulty_family": "...",
    "ood_split": "...",
    "seed": 0
  }
}
```

### 6.3 Pearl 三层任务家族

#### L1：Association Layer

目标：识别“相关是否足以支持因果 claim”

建议覆盖：

- latent confounding
- reverse causality ambiguity
- selection bias / collider bias
- proxy-assisted disambiguation
- Simpson-like reversal

#### L2：Intervention Layer

目标：识别 `P(Y|do(X))` 是否被当前证据支持

建议覆盖：

- valid backdoor identification
- invalid backdoor adjustment
- weak IV
- invalid IV
- proxy-assisted identification
- subgroup / heterogeneity overclaim

#### L3：Counterfactual Layer

目标：识别反事实结论是否真正可支持

建议覆盖：

- same observational fit, different counterfactual answers
- monotonicity failure
- mediator misspecification
- cross-world assumption sensitivity
- ETT / PN / PS overclaim

### 6.4 攻击类型 taxonomy

Attacker 不再只是“随机说谎”，而是程序化生成如下类型：

1. `association_overclaim`
2. `hidden_confounder_denial`
3. `selection_bias_obfuscation`
4. `invalid_adjustment_claim`
5. `weak_iv_as_valid_iv`
6. `invalid_iv_exclusion_claim`
7. `heterogeneity_overgeneralization`
8. `counterfactual_overclaim`
9. `function_form_manipulation`
10. `unidentifiable_claim_disguised_as_valid`

### 6.5 benchmark 生成原则

1. **程序化生成，不依赖固定 3 个故事模板。**
2. **图结构随机化。**
3. **参数随机化。**
4. **语言模板随机化。**
5. **变量名随机化，不用固定 `smoking/education/drug`。**
6. **保留少量易解释 showcase 场景供 demo 使用。**
7. **主实验必须用 fresh split，不允许复用 showcase 叙事当 test。**

### 6.6 数据集切分方案

建议采用四层切分：

- `train`
  只用于 prompt 设计、parser 调试、rule 开发
- `dev`
  用于方法调参与错误分析
- `test-iid`
  同家族新实例
- `test-ood`
  新图结构、新语言模板、新变量命名、新攻击强度

### 6.7 benchmark 污染控制

为了避免 benchmark contamination，建议同时执行：

1. 模板随机化
2. 变量命名随机化
3. 隐变量语义随机化
4. 语义 paraphrase
5. 图家族 holdout
6. 时间上 fresh generation
7. 人工 spot-check

---

## 7. 方法：Countermodel-Grounded Verification

### 7.1 方法总览

最终 verifier 不再是“综合打分器”，而是下面四步：

1. `Claim Parsing`
2. `Assumption Ledger`
3. `Countermodel Search`
4. `Tool-backed Adjudication`

### 7.2 模块 1：Claim Parsing

输入：
- 自然语言 claim
- 攻击者文本 / transcript

输出结构：

```json
{
  "query_type": "association/intervention/counterfactual",
  "treatment": "...",
  "outcome": "...",
  "claim_polarity": "positive/negative/null",
  "claim_strength": "tentative/strong/absolute",
  "mentioned_assumptions": [...],
  "implied_assumptions": [...],
  "rhetorical_strategy": "...",
  "needs_abstention_check": true
}
```

### 7.3 模块 2：Assumption Ledger

这是方法中的关键创新之一。

其目标不是总结语言，而是把对方论证背后的识别前提全部显式列出来，例如：

- no unobserved confounding
- valid adjustment set
- exclusion restriction
- instrument relevance
- positivity
- no selection bias
- monotonicity
- cross-world consistency
- correct functional form

ledger 输出应包含：

- assumption 名称
- assumption 来源
  由 claim 明示 / 隐含 / 工具要求
- assumption 当前状态
  supported / contradicted / unresolved

### 7.4 模块 3：Countermodel Search

这是整个 verifier 的核心。

其目标是尝试回答：

> 是否存在一个与当前观测分布一致、但对目标 causal query 给出相反结论的模型？

若答案是 yes，则：

- 原 claim 至少不能被唯一支持
- 若反例强，则直接判 invalid
- 若只是说明多解，则判 unidentifiable

#### Countermodel 搜索策略

建议按层级执行：

##### L1

- latent confounder injection
- direction flip candidate
- selection mechanism candidate

##### L2

- hidden confounder compatible models
- weak / invalid IV alternatives
- alternative proxy explanations

##### L3

- same observational fit but different PN / PS / ETT
- alternative SCM family
- functional form flip

#### 输出

```json
{
  "found_countermodel": true,
  "countermodel_type": "...",
  "observational_match_score": 0.94,
  "query_disagreement": true,
  "countermodel_explanation": "...",
  "verdict_suggestion": "invalid/unidentifiable"
}
```

### 7.5 模块 4：Tool-backed Adjudication

只有在 countermodel search 不能击穿 claim 时，才进入支持性判断。

支持性工具分层如下：

#### L1

- correlation analysis
- conditional independence test
- partial correlation
- Simpson / selection pattern checks

#### L2

- backdoor validity analysis
- IV estimation
- IV validity diagnostics
- sensitivity analysis
- proxy-based checks

#### L3

- counterfactual estimation
- SCM distinguishability analysis
- alternative-model sensitivity
- ETT / PN / PS only as carefully labeled approximations or identified quantities

### 7.6 最终决策规则

建议采用以下顺序：

1. 若存在强 countermodel witness，且与 claim 明显冲突：
   - 输出 `invalid`
2. 若存在多个 observationally compatible models，且 query answer 不一致：
   - 输出 `unidentifiable`
3. 若核心识别假设未被支持且无法排除替代解释：
   - 输出 `unidentifiable`
4. 若没有有效反例，且识别条件和工具证据共同支持：
   - 输出 `valid`

### 7.7 最终输出格式

```json
{
  "label": "valid/invalid/unidentifiable",
  "confidence": 0.73,
  "assumption_ledger": [...],
  "support_witness": {...},
  "countermodel_witness": {...},
  "tool_trace": [...],
  "reasoning_summary": "..."
}
```

---

## 8. 理论部分的最终写法

### 8.1 理论目标

本项目不需要一开始就追求非常强的大定理，但必须至少具备：

- 明确定义
- 正确命题
- 合理证明思路
- 清晰边界条件

### 8.2 建议写入论文的三个命题

#### 命题 1：Abstention Necessity

若存在两个 observationally equivalent models 对目标 query 给出不同结论，则任何仅依赖观测证据的 verifier 都不应被要求输出唯一有效判定。

论文作用：

- 为 `unidentifiable` 标签提供理论必要性

#### 命题 2：Countermodel Witness Soundness

若 verifier 构造出一个与观测分布一致、且对目标 query 给出相反结论的 countermodel，则原 claim 不能被当前观测证据唯一支持。

论文作用：

- 为 countermodel-based rejection 提供方法合理性

#### 命题 3：Oracle Leakage Inflation

若 verifier 可访问 gold graph / hidden variable / true SCM 级别信息，则其性能不再代表真实因果监督能力。

论文作用：

- 为 leakage-free benchmark 的必要性提供理论和实验支撑

### 8.3 理论边界

必须明确写出：

- verifier 不是在“恢复真实世界唯一真模型”
- verifier 的任务是“在给定证据下审计 claim 是否可被支持”
- `valid` 表示“当前证据支持且未发现有效反例”，而不是“宇宙真理”

---

## 9. 实验设计

### 9.1 主指标

建议最终主表只保留以下指标：

1. `Verdict Accuracy`
2. `Macro F1`
3. `Invalid Claim Acceptance Rate`
4. `Unidentifiable Awareness`
5. `ECE / Brier Score`
6. `Countermodel Coverage`

### 9.2 次级指标

放进 appendix：

- DSR
- Arms Race Index
- Strategy Diversity
- Jury Agreement
- Difficulty Stability
- Evolution Trend

### 9.3 baseline 设计

#### Judge 类

- direct LLM judge
- CoT judge
- self-consistency judge
- CLadder-style prompt judge

#### Debate 类

- single-turn rebuttal
- multi-agent debate
- current repo A/B/C reduced debate baseline

#### Tool 类

- tool-only verifier
- tool + parser verifier

#### 你们的方法

- parser + assumption ledger
- parser + ledger + countermodel
- full countermodel-grounded verifier

### 9.4 主实验矩阵

#### Exp 1：Core Benchmark

目标：
- 比较不同 verifier 在 main benchmark 上的整体表现

#### Exp 2：Adversarial Robustness

按攻击强度分层：
- weak
- medium
- strong
- hidden-information-aware

#### Exp 3：Identifiability Ablation

移除以下组件：
- no assumption ledger
- no countermodel search
- no abstention
- no tools

#### Exp 4：Leakage Study

比较：
- clean information partition
- oracle-leaking partition

目标：
- 证明泄露会夸大能力

#### Exp 5：OOD Generalization

比较：
- graph family OOD
- lexical OOD
- variable naming OOD

#### Exp 6：Cross-Model Transfer

攻击者 / verifier 使用不同模型家族，测试泛化。

#### Exp 7：Human Audit

抽样人工评审：
- label correctness
- witness quality
- explanation faithfulness

### 9.5 统计协议

所有正式实验必须满足：

1. 至少 `3-5` 个 seeds
2. 汇报 mean ± std
3. bootstrap 95% CI
4. paired significance test
   - 推荐 McNemar 或 bootstrap paired test
5. 多重比较校正
   - 推荐 Holm-Bonferroni

### 9.6 人工评审协议

建议：

- 样本量：150-300 条
- 双人标注 + 冲突仲裁
- 标注项：
  - gold label 是否合理
  - verifier label 是否合理
  - countermodel witness 是否有说服力
  - explanation 是否忠实于工具证据

---

## 10. 现有仓库资产复用方案

### 10.1 资产分层

#### A 类：直接保留

- `game/data_generator.py`
- `causal_tools/`
- `agents/tool_executor.py`
- `evaluation/tracker.py`
- `experiments/` 框架
- `visualization/`

#### B 类：重写后保留

- `agents/agent_a.py`
  改为 attacker / claim generator
- `agents/agent_c.py`
  改为 verifier-first
- `agents/agent_b.py`
  改为 proposer / debate baseline

#### C 类：降级为 appendix / demo

- `agents/jury.py`
- `game/difficulty.py`
- `game/evolution.py`
- `run_live_game.py`

### 10.2 旧模块到新角色的映射

| 旧模块 | 新角色 | 处理方式 |
|---|---|---|
| Agent A | Attacker | 保留思想，重写为程序化 claim generator |
| Agent B | Baseline debater / proposer | 降级为 baseline |
| Agent C | Main verifier | 重点重构 |
| Jury | Appendix ablation | 不删，但退出主线 |
| Difficulty | Demo control | 退出主论文主线 |
| Evolution | Demo / appendix | 退出主论文主线 |

### 10.3 现有设计中必须立刻废弃的部分

1. verifier 读取真 DAG
2. verifier 读取 true SCM
3. verifier 读取 hidden variable identity
4. benchmark 只保留 3 个 showcase 场景当 test
5. 用 `winner == agent_b` 作为 detection proxy 写进主实验

---

## 11. 文件级施工规划

### 11.1 第一阶段：不改 UI，只改研究内核

#### 新增建议

- `benchmark/`
  - `schema.py`
  - `generator.py`
  - `splits.py`
  - `attacks.py`
  - `witness.py`

- `verifier/`
  - `claim_parser.py`
  - `assumption_ledger.py`
  - `countermodel_search.py`
  - `decision.py`

- `experiments/`
  - `exp_main_benchmark/`
  - `exp_adversarial_robustness/`
  - `exp_leakage_study/`
  - `exp_ood_generalization/`
  - `exp_transfer/`
  - `exp_human_audit/`

#### 复用建议

- 保留 `causal_tools/`
- 保留 `evaluation/tracker.py`
- 保留 `visualization/`
- 保留 `main.py`，但未来改为 benchmark runner 外壳

### 11.2 第二阶段：把 demo 层重新挂回去

在主实验系统稳定之后，再把：

- live game
- jury
- difficulty
- evolution

接回去做 supplemental demo。

---

## 12. 当前实现中的具体整改清单

### 12.1 P0：必须先修的结构性问题

1. 重新设计 `CausalScenario` 的 access control
2. 切断 oracle fields 对 verifier 的暴露
3. 新建 `gold_only` 与 `public_view` 两套对象
4. 统一 label space 为 `valid / invalid / unidentifiable`
5. verifier 输出必须允许 abstain

### 12.2 P1：必须修的实现错位

1. `AgentA.adapt_strategy()` 参数错位
2. `configs/default.yaml` 与 `DifficultyController` 默认值不一致
3. `Scorer` 中 `MetricResult` 构造不完整
4. 若继续保留 PN / PS 命名，必须区分：
   - identified quantity
   - heuristic proxy

### 12.3 P2：研究主线相关改造

1. 重写 `Agent C` 为 verifier-first
2. 增加 claim parser
3. 增加 assumption ledger
4. 增加 countermodel search
5. benchmark 由固定故事改为程序化 family generator

### 12.4 P3：实验层改造

1. 增加 seed sweep
2. 增加 CI 和显著性
3. 增加 OOD split
4. 增加 leakage study
5. 增加人工评审脚本

---

## 13. 施工阶段划分

### 阶段 0：冻结论文主线

输出物：

- 本文档确认版
- 标题确认
- 三点贡献确认
- 不再继续加功能

### 阶段 1：benchmark 核心重构

目标：

- 完成 schema
- 完成 public/gold 分区
- 完成新 generator
- 完成 train/dev/test/ood split

输出物：

- benchmark v1

### 阶段 2：verifier 核心实现

目标：

- parser
- assumption ledger
- countermodel search
- decision rule

输出物：

- verifier v1

### 阶段 3：主实验跑通

说明：
- 若与 `docs/AGENT_EXECUTION_MANUAL.md` 的 `Phase` 编号发生冲突，以后者为准。
- 本节对应执行手册中的 `Phase 4` 主实验实现范围。

目标：

- main benchmark
- robustness
- leakage study
- ablation

输出物：

- 正式实验表格 v1

### 阶段 4：人工评审与写作

说明：
- 本节是主实验完成后的人工评审与论文写作阶段，不作为执行手册 `Phase 4` 的编号依据。

目标：

- human audit
- 论文正文初稿
- 图表与案例分析

输出物：

- paper draft v1

### 阶段 5：补充 demo 与 appendix

目标：

- 把 jury / evolution / visualization 作为系统加分项补回

输出物：

- supplemental materials

---

## 14. 论文写作总提纲

### 14.1 正文结构建议

1. Introduction
2. Related Work
3. Adversarial Causal Oversight Task
4. Leakage-Free Benchmark
5. Countermodel-Grounded Verification
6. Theoretical Motivation
7. Experiments
8. Limitations
9. Broader Impact

### 14.2 引言的主叙事

建议引言只讲下面这条故事线：

1. LLM 因果推理正在被越来越多地使用。
2. 但现实中的因果 claim 经常是对抗性的，并且存在隐藏信息。
3. 现有 benchmark 往往忽略 information asymmetry 与 unidentifiability。
4. 现有 judge / debate 方法在 persuasive but wrong causal claims 上仍然不稳。
5. 为此，我们提出 adversarial causal oversight、构建新 benchmark，并提出 countermodel-grounded verifier。

### 14.3 不要在引言里讲的内容

以下内容不要再占引言主线：

- UI
- WebSocket
- 多角色设定的戏剧性
- 动态难度
- 游戏平衡
- 军备竞赛

这些最多在 appendix 或 system demo 里出现。

---

## 15. 风险与应对

### 风险 A：新颖性仍被审稿人认为不足

应对：

- 明确不与 CLadder/MAD/CRAwDAD 做同类叙事
- 强调 `unidentifiable`
- 强调 leakage-free
- 强调 countermodel witness

### 风险 B：countermodel search 做不稳

应对：

- 分层实现，从 rule-based family 开始
- 先在 L1/L2 做强，再扩展到 L3
- 不强求一开始就做极强通用搜索

### 风险 C：实验跑不出显著优势

应对：

- 先做 leakage study，确保至少能证明“oracle leakage 会虚高”
- 再做 unidentifiable awareness，这往往比总体 accuracy 更容易拉开差距

### 风险 D：开发量过大

应对：

- 严格冻结非主线模块
- 先做 benchmark 和 verifier
- UI / jury / evolution 全部后置

---

## 16. 最终交付物定义

### 必须交付

1. benchmark 数据与生成器
2. verifier 实现
3. main experiments
4. paper draft
5. reproducibility package

### 可选加分交付

1. live demo
2. interactive visualization
3. jury appendix
4. evolution appendix

---

## 17. 最终决策

### 17.1 最终做什么

本项目最终应做：

> 一个面向对抗性因果 claim 验证的新任务、一个严格控制信息泄露的新 benchmark、以及一个以 countermodel search 为核心的 verifier。

### 17.2 最终不做什么

本项目最终不应再继续把以下内容作为主方向：

- 更复杂的游戏玩法
- 更多陪审团花样
- 更复杂的动态难度
- 更花哨的 UI
- 更长的辩论流程

### 17.3 一句话总施工路线

> 先把项目从“课程系统作品”收缩成“任务 + benchmark + verifier”的论文核心，再把现有系统资产作为 appendix 和 demo 重新挂回去。

---

## 18. 参考脉络（写作时优先使用）

### 顶会与核心基线

- CLadder, NeurIPS 2023  
  https://papers.nips.cc/paper_files/paper/2023/hash/631bb9434d718ea309af82566347d607-Abstract-Conference.html
- Unveiling Causal Reasoning in Large Language Models: Reality or Mirage?, NeurIPS 2024  
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/af2bb2b2280d36f8842e440b4e275152-Abstract-Conference.html
- Improving Factuality and Reasoning in Language Models through Multiagent Debate, ICML 2024  
  https://proceedings.mlr.press/v235/du24e.html
- On Scalable Oversight with Weak LLMs Judging Strong LLMs, NeurIPS 2024  
  https://arxiv.org/abs/2407.04622
- Causal Discovery via Conditional Independence Testing with Proxy Variables, ICML 2024  
  https://proceedings.mlr.press/v235/liu24bc.html
- Efficient and Trustworthy Causal Discovery with Latent Variables and Complex Relations, ICLR 2025  
  https://openreview.net/forum?id=BZYIEw4mcY
- Latent Variable Causal Discovery under Selection Bias, ICML 2025  
  https://proceedings.mlr.press/v267/dai25k.html
- Unbiased Evaluation of LLMs from a Causal Perspective, ICML 2025  
  https://proceedings.mlr.press/v267/chen25bi.html

### 前沿 arXiv

- CRAwDAD, arXiv:2511.22854  
  https://arxiv.org/abs/2511.22854
- CausalPitfalls, arXiv:2505.13770  
  https://arxiv.org/abs/2505.13770
- CausalFlip, arXiv:2602.20094  
  https://arxiv.org/abs/2602.20094
- LLM Cannot Discover Causality..., arXiv:2506.00844  
  https://arxiv.org/abs/2506.00844
- VIGOR+, arXiv:2512.19349  
  https://arxiv.org/abs/2512.19349
- TLVD, arXiv:2602.14456  
  https://arxiv.org/abs/2602.14456

---

## 19. 最终判词

这就是我给你们的最终版本结论：

1. **方向可以继续做，而且值得做。**
2. **但只能以收缩后的版本继续做。**
3. **最优版本不是“更复杂的系统”，而是“更严格的任务定义、更干净的 benchmark、更强的 verifier”。**
4. **现有工作成果可以保留很多，但必须重新分层。**
5. **如果后续严格按本文档推进，本项目会从“优秀课程项目”显著更接近“可投稿研究项目”。**

这份文档即为后续的总施工图。
