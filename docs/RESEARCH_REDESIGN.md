# Causal Traitor 论文级重构方案

## 1. 当前项目的真实定位

以 2026 年 4 月的仓库状态看，`causal-traitor` 已经是一个完成度很高的课程项目与系统 Demo，但距离 ICML / NeurIPS 主会论文仍有明显距离。核心问题不是“做得不够多”，而是“做得太散”：系统里有 Agent、Jury、Difficulty、Evolution、Visualization、14 个指标和 4 个实验，但缺少一条足够尖锐、足够可验证、足够可发表的主线。

如果继续沿着“做一个复杂平台”推进，最可能的结果是展示效果很好，但论文贡献被审稿人拆成多个都“不够新、不够深、不够严”的子点。顶会更喜欢的是：一个清晰的问题定义，一个真正站得住的核心方法，一个严密的实验闭环。

## 2. 顶会视角下的核心问题

### 2.1 现在最大的四个硬伤

1. 问题定义过宽

当前项目同时想讲：
- 因果欺骗
- 隐变量发现
- 多智能体辩论
- 陪审团投票
- 动态难度
- 多轮进化
- 可视化系统

这会导致论文主张发散，审稿人很难回答“你们到底解决了什么核心科学问题”。

2. 信息不对称设定在实现上并不干净

当前 `CausalScenario` 直接存储 `true_dag / hidden_variables / full_data / true_scm`，而 `Agent B / Agent C / ToolExecutor` 都会读取这些 oracle 字段。这样一来，“Agent A 独占隐藏信息”的论文设定在实现层实际上被破坏，任何基于该设定的实验结论都会被质疑。

3. 实验更像平台 sanity check，不像论文证据

当前实验大多在回答：
- 系统能不能跑
- 不同模块开关后数值怎么变

但顶会真正关心的是：
- 你定义的任务是否严谨
- 你的方法是否比强基线更可靠
- 改进来自什么机制
- 结论是否在新分布、强攻击、不同模型家族下仍成立

4. 评价目标还不够“科学问题导向”

诸如 DSR、Flow、Game Balance、Arms Race 这类指标适合系统调参与博弈展示，但还不够像论文核心指标。顶会更关注：
- 在有效 / 无效 / 不可识别 claim 上的判决正确率
- 是否能正确 abstain
- 是否校准
- 是否能输出可验证的 counterexample 或 identifying witness

### 2.2 当前项目最值得保留的资产

- 你们已经有了一个很好的“问题直觉”：因果推理在信息不对称和对抗说服下会崩。
- 你们已经有了 Pearl ladder 结构，这是天然的论文主线骨架。
- 你们已经搭好了 SCM 数据生成、工具调用、回合式交互和实验脚本，工程起点很好。
- 你们的系统天然适合转向“benchmark + verifier”型论文，这比继续堆系统功能更有发表潜力。

## 3. 建议的论文重定位

### 3.1 不要再把论文主线写成“复杂多智能体系统”

建议把论文主线改成：

**Adversarial Causal Oversight under Information Asymmetry**

中文可表述为：

**信息不对称下的对抗性因果监督**

一句话版本：

> 当攻击者掌握未观测因子或结构先验，并用自然语言辩论包装错误的因果结论时，LLM 审计器能否仅基于可观测证据做出可靠判断？

### 3.2 建议的核心研究问题

> 在 Pearl 三层因果推理任务中，面对拥有额外隐藏信息的对手，现有 LLM judge / debate / tool-augmented 方法在区分 valid, invalid, and unidentifiable causal claims 时到底有多可靠？如何通过“反例模型搜索 + 识别性监督”显著提高鲁棒性？

### 3.3 建议的论文贡献重新压缩为三点

1. **任务与 benchmark**
   
形式化一个新的任务：`Adversarial Causal Oversight`。攻击者知道更多隐藏信息或可构造更强叙事；审计者只能看到观测数据、可选代理变量、辩论记录与工具输出。

2. **方法**
   
提出一个围绕“识别性”而不是“说服力”的 verifier：
`Claim Parsing -> Assumption Ledger -> Countermodel Search -> Tool-backed Verdict`

3. **发现**
   
系统性证明：
- vanilla LLM judge 在新鲜因果题上不稳定
- 纯 debate 不足以解决隐藏混杂与不可识别性
- 只有把“可观测一致但因果答案相反的 countermodel”显式拉进验证流程，才能显著提升鲁棒性与校准

## 4. 建议的方法主线

### 4.1 从 “A/B/C/Jury/Difficulty/Evolution” 收束到一个真正的核心

建议把方法主体重写成三个角色：

- `Attacker`
  负责构造 persuasive but wrong / underidentified causal claim
- `Verifier`
  负责将 claim 结构化、调用工具、搜索 countermodel、输出证据
- `Judge`
  负责基于 verifier 产物做最终三分类：`valid / invalid / unidentifiable`

其中：
- 现在的 Agent A 可以保留，作为 attacker
- 现在的 Agent C 可以保留部分能力，但需要被重写成 verifier-first
- 现在的 Agent B 和 Jury 不再作为主贡献，只作为 baseline 或 ablation
- Difficulty、Evolution、Visualization 全部降为 appendix / demo assets，而不是 main paper contribution

### 4.2 建议的新方法核心：Countermodel-Grounded Verification

建议把真正的创新点落到下面这个机制上：

#### Step 1. Claim parsing

把自然语言 claim 解析成结构化对象：
- query type: association / intervention / counterfactual
- treatment / outcome
- claimed sign or effect
- explicit assumptions
- implicit assumptions

#### Step 2. Assumption ledger

把对方论证隐含依赖的假设显式列出来：
- no unobserved confounding
- exclusion restriction
- monotonicity
- cross-world consistency
- correct functional form
- no selection bias

这个 ledger 是论文里很关键的一步，因为它把“语言辩论”转成“因果识别假设对账”。

#### Step 3. Countermodel search

尝试构造一个与观测分布一致，但对目标 query 给出相反结论的 SCM / semi-SCM / graph variant。

如果找到了 countermodel，则说明：
- 该 claim 可能是错误的
- 或至少在当前观测信息下不可识别

#### Step 4. Tool-backed adjudication

只有在 countermodel search 失败、且识别条件可被支持时，才用 backdoor / IV / frontdoor / counterfactual tools 给出 positive verdict。

### 4.3 这条主线比现在强在哪里

- 它把“审计”从印象打分变成了识别性验证
- 它能自然引入理论命题
- 它能解释为什么普通 debate 不够
- 它能把 Pearl ladder 真正融进方法，而不是只融进场景包装

## 5. benchmark 设计必须重做

### 5.1 当前 benchmark 的问题

- 只有 3 个固定故事模板，太容易被模型记住
- 场景家族太少，统计结论不稳
- 隐变量存在但缺少更细的 identifiability 设计
- 没有把 `invalid` 和 `unidentifiable` 分开
- 自然语言模板可能泄露答案风格

### 5.2 建议的新 benchmark 结构

每个样本由如下元素组成：
- gold SCM
- observed variables
- optional proxies
- optional selection mechanism
- natural-language claim
- attacker rationale
- gold label: `valid / invalid / unidentifiable`
- gold query answer
- gold assumption set
- optional countermodel witness

### 5.3 建议的任务家族

#### L1: 关联层

- latent confounding
- selection bias / collider bias
- reverse causality ambiguity
- proxy-supported disambiguation

#### L2: 干预层

- valid backdoor
- invalid backdoor
- weak IV / invalid IV
- proxy-assisted identification
- heterogeneous treatment effect overclaim

#### L3: 反事实层

- same observational distribution, different counterfactual answers
- monotonicity violation
- mediator misspecification
- alternative SCM with equal fit but opposite PN / PS / ETT

### 5.4 benchmark 生成原则

1. **程序化生成，不要只靠手写 3 个故事**
2. **图结构、参数、语言模板全部随机化**
3. **按 graph family 划分 train / dev / test**
4. **按 lexical template 做额外 OOD split**
5. **必须加入 freshness 机制，避免 benchmark contamination**

## 6. 理论部分怎么补

当前项目最大短板之一是“有工程，没有理论抓手”。建议补两类命题：

### 6.1 不可识别性命题

如果存在两个在观测分布上不可区分、但在目标 query 上结论相反的模型，则任何只基于观测证据的 judge 都不应被迫做二元裁决。

这会直接推出一个论文结论：

> 在 adversarial causal oversight 中，`abstain / unidentifiable` 不是可选项，而是理论上必要的输出空间。

### 6.2 Countermodel witness 的正确性命题

证明或半形式化论证：
- 若 verifier 找到满足观测一致性的 countermodel，则原 claim 至少不可被当前证据唯一支持
- 若 verifier 找不到 countermodel，且满足对应识别条件，则可输出“provisionally valid”并给出 identifying witness

即便做不到大定理，也必须把定义、命题、证明思路和边界条件写清楚。

## 7. 实验必须从“系统展示”升级为“论文验证”

### 7.1 核心指标建议重构

建议把主指标收束到 5 个：

- `Verdict Accuracy`
  在 `valid / invalid / unidentifiable` 三类上的总体准确率
- `Invalid Claim Acceptance Rate`
  把错误 claim 判成 valid 的比例
- `Unidentifiable Awareness`
  对 underidentified 实例正确 abstain 的能力
- `Calibration`
  ECE / Brier score
- `Countermodel Coverage`
  verifier 能给出显式 counterexample 的比例

DSR、Arms Race、Flow、Game Balance 等指标可以保留到 appendix，不再做 main table。

### 7.2 建议的主实验

#### Exp 1. Core benchmark

比较：
- direct LLM judge
- CoT / self-consistency
- CausalCoT / CLadder-style prompting
- multi-agent debate
- tool-only verifier
- 你们的方法

按 L1/L2/L3、场景家族、语言 OOD 分层汇报。

#### Exp 2. Adversarial robustness

控制攻击者强度：
- no attack
- mild persuasive attack
- strong causal attack
- hidden-variable-aware attack

看 judge 的性能如何退化。

#### Exp 3. Identifiability ablation

比较：
- no assumption ledger
- no countermodel search
- no tools
- no abstain option

证明你们的方法不是“多调用模型就更强”，而是因为识别性机制真的起作用。

#### Exp 4. Leakage-free vs oracle-leaking

这是你们非常应该做的一组实验。

你们要主动展示：
- 在 oracle leakage 版本里，分数虚高
- 在 clean information partition 版本里，任务更难但更真实

这会让论文显得更诚实，也更有研究价值。

#### Exp 5. Cross-model transfer

用不同模型家族当 attacker / judge：
- Qwen attack -> GPT judge
- DeepSeek attack -> Qwen judge
- Claude attack -> open model judge

证明方法不是某个模型组合上的特例。

#### Exp 6. Human or expert audit

抽取一小部分样本做人工标注：
- label 是否正确
- countermodel 是否合理
- verifier explanation 是否可信

顶会很看重这一点，尤其是涉及 judge 和 benchmark 的论文。

## 8. 当前代码库中哪些要保留，哪些要降级

### 8.1 应保留并优先重构的部分

- `game/data_generator.py`
  这是最接近 benchmark 生成器的资产
- `agents/tool_executor.py`
  这是 future verifier 的工具骨架
- `causal_tools/`
  是方法可解释性的基础
- `experiments/`
  可以作为新实验骨架

### 8.2 应降级为 appendix / demo 的部分

- `jury`
- `difficulty`
- `evolution`
- `visualization`

这些东西不是不能有，而是不能再作为 main contribution 写进摘要和标题。

### 8.3 必须先修掉的结构性问题

1. 重新设计 information partition
   
attack / verify / judge 各自能看到什么，必须严格定义并在代码层 enforce。

2. 拆分 oracle fields
   
`CausalScenario` 里不应直接把所有真值都暴露给所有角色。

3. 统一 label space
   
不要只输出 “A 赢 / B 赢”，而要输出 `valid / invalid / unidentifiable`。

4. 让 benchmark 先于 debate
   
论文首先是一个严谨任务，其次才是交互协议。不要把 protocol 当作 benchmark 本身。

## 9. 建议的论文标题与贡献表述

### 9.1 论文标题建议

比起 `The Causal Traitor`，主论文更建议使用正式标题：

- `Adversarial Causal Oversight under Information Asymmetry`
- `Countermodel-Grounded Verification for Adversarial Causal Reasoning`
- `Benchmarking LLM Judges for Hidden-Confounder Causal Claims`

`Causal Traitor` 可以保留为 repo / demo 名称。

### 9.2 摘要里建议只讲这三点

1. 提出新任务：adversarial causal oversight
2. 提出新方法：countermodel-grounded verifier
3. 发现现有 LLM judge / debate 在 hidden-confounder 与 counterfactual setting 下显著失稳，而你们的方法显著更稳健

## 10. 两周内最值得做的事

### P0. 先冻结论文主线

不要再增加系统功能。先确定 paper question、task definition、label space、main metrics。

### P1. 先做 leakage-free benchmark

这是第一优先级，比改 prompt、改 jury、改 UI 都重要。

### P2. 重写 verifier 方案

从“综合打分器”改成“假设账本 + countermodel search + tool-backed verdict”。

### P3. 重做实验矩阵

把现在 4 个实验重组为：
- benchmark
- robustness
- identifiability ablation
- leakage study
- transfer
- human audit

### P4. 最后再考虑 demo 美化

UI 和 live game 对课程展示有价值，但对顶会主线帮助有限，应后置。

## 11. 关键参考脉络

建议你们后续写作时以这些工作为主干，而不是只引用最新 arXiv：

- CLadder, NeurIPS 2023
- Unveiling Causal Reasoning in Large Language Models: Reality or Mirage?, NeurIPS 2024
- Improving Factuality and Reasoning in Language Models through Multiagent Debate, ICML 2024
- Causal Discovery from Proxy Variables via Conditional Independence Test, ICML 2024
- Efficient and Trustworthy Causal Discovery with Latent Variables and Complex Relations, ICLR 2025
- Latent Variable Causal Discovery under Selection Bias, ICML 2025

CRAwDAD 与 TLVD 可以作为“最接近你们方向的最新 preprint”，但不应独自承担论文立论的理论基础。

## 12. 最终建议

如果你们真想冲击顶会，那么这项工作最值得走的路线不是“把系统做得更花”，而是：

> 从一个复杂的课程系统，收缩成一个严谨的新任务 + 一个真正围绕因果识别性的 verifier + 一组能说服审稿人的 benchmark 实验。

这条路会明显更难，但也明显更像论文。
