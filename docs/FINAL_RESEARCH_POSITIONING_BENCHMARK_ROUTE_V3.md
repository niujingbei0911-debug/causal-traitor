# 论文方案最终修订版 V3：信息不对称下的因果声明审计 Benchmark

## 0. 顶会标准预审结论

本方案的核心路线可以继续推进，但必须按 **benchmark-first + validity-package + real-grounded audit** 的标准执行。仅凭当前旧实现或 V2 方案，仍不足以支撑 AAAI 2027 / ICLR 2027 级别投稿；但如果 V3 中定义的证据链全部完成，并且实验结果满足投稿门槛，则可以形成一篇有真实投稿可能性的 benchmark 论文。

这里的“可以投稿”不等于“保证接收”。顶会审稿人仍会重点质疑四件事：

1. 这个任务是否真实重要，而不是人为拼装出来的任务。
2. 这个 benchmark 是否测到了因果识别失败，而不是语言模板、拒答倾向或事实检索能力。
3. 标签是否可信，尤其是 `unidentifiable` 是否有可审计依据。
4. 强 LLM baseline 是否确实会在该任务上出现系统性 unsafe acceptance。

因此，V3 方案把“能否投稿”的判断从口头论证改成硬性验收：

- 若最强真实 LLM baseline 没有显著 unsafe acceptance，则 benchmark 诊断价值不足，不能投稿。
- 若 real-grounded subset 的 human audit agreement 不达标，则标签可信度不足，不能投稿。
- 若 leakage / paired-flip / OOD 检查显示模型可依赖模板或泄漏信息解题，则 benchmark 可靠性不足，不能投稿。
- 若 reference verifier 不能在 unsafe acceptance 上稳定优于强 baseline，则 verifier 必须降级为分析工具，不能作为方法贡献。

这意味着 V3 的目标不是承诺“已经达到顶会标准”，而是定义一条能被顶会审稿人认真评估的证据闭环。

## 1. 一句话定位

本论文不再以“多智能体因果博弈系统”为主线，也不以“提出一个全新的 verifier 算法”为主贡献，而是提出：

> 一个面向信息不对称高风险因果声明审计的、带 public/gold 信息切分、可验证标签、跨领域真实 grounding 和泄漏控制的 benchmark，用于评估 LLM verifier 是否会在公开证据不足以识别因果效应时 unsafe accept，以及是否能够做出因果识别意义上的 wise refusal。

论文主贡献应固定为三点：

1. **新任务**：Selective Causal Claim Auditing under Information Asymmetry。
2. **新 benchmark**：一个包含 synthetic core 与 real-grounded subset 的 public/gold-partitioned benchmark。
3. **诊断发现与参考方法**：系统揭示强 LLM verifier 的 unsafe acceptance，并提供 Countermodel-Grounded Selective Verification 作为 reference verifier。

## 2. 真实问题是什么

现实中，高风险因果声明经常由信息更多的一方提出，而外部审核者只能看到有限公开证据。例如：

- 医疗机构声称某治疗方案导致康复率提升。
- 政策部门声称某培训项目提高就业率。
- 教育机构声称某教学干预提升学习成绩。
- 平台公司声称某推荐策略降低有害内容暴露或提升用户福祉。
- 广告系统声称某投放策略带来销售转化增长。
- 企业或研究报告声称某经济政策、环境政策或社会干预产生了正向效果。
- 科学论文声称某基因扰动、药物处理或实验条件导致特定生物响应。

这些场景的共同结构是：

> claimant 掌握或隐含更多内部信息；verifier 只能看到公开证据、摘要、部分统计结果、可观察变量和自然语言论证。

在这种设置下，问题的根源不是普通事实错误，而是：

> 公开证据可能不足以识别目标因果效应，但 LLM verifier 会被自然语言因果叙述、权威表述、统计相关性或局部证据说服，从而错误输出“该因果声明成立”。

我们称这一失败为 **unsafe acceptance**。

## 3. 现有工作不足在哪里

已有工作与本项目相邻，但没有覆盖我们的核心任务。

1. **LLM 因果推理 benchmark**  
   CLadder、CausalBench 等工作评估模型回答因果题、识别因果图或执行因果推理的能力，但通常不是以“信息不对称审计”作为任务结构，也不系统测试 public evidence 不足时的拒绝能力。

2. **因果事实验证**  
   CHECKWHY 提出了 causal fact verification，并包含大量 claim-evidence-argument triplets 与 support/refute/not-enough-info 标签。但它的 NEI 并不等价于因果识别理论中的 `unidentifiable`，也不强调 public/gold 信息切分。

3. **因果识别与估计 benchmark**  
   CausalReasoningBenchmark 很接近因果识别任务，包含 173 个 query、138 个真实数据集、85 篇论文和多本教材案例，要求系统给出识别规格与估计结果。但它不是自然语言因果声明审计任务，也不以 LLM 在公开证据不足时的 unsafe acceptance 为核心指标。

4. **拒答与不确定性 benchmark**  
   AbstentionBench、UA-Bench 等关注模型是否知道自己不知道，但多数“不知道”来自知识缺失、证据不足或上下文矛盾，而不是“即使有很多观测数据，因果效应仍不可识别”的结构性问题。

5. **LLM-as-a-judge / debate / persuasion attack**  
   这些工作说明 LLM judge 会受表达方式和说服性话术影响，但没有把失败机制落到因果识别、隐藏混杂、选择偏差、无效工具变量等因果结构上。

因此，本文的空白应表述为：

> 现有 benchmark 缺少一个专门评估 LLM 在信息不对称因果声明审计中，能否区分 `valid / invalid / unidentifiable`，并避免在因果不可识别时 unsafe accept 的标准化评测。

## 4. 为什么必须用因果，而不是普通拒答或事实核查

普通事实核查关心“证据是否支持 claim”。普通拒答关心“模型是否知道答案”。但本任务中的关键困难是：

> 一个 claim 可以拥有大量相关证据、合理叙事和显著统计结果，但仍然无法从公开信息中识别出因果效应。

例如，公开证据显示“参加培训者就业率更高”，这只说明 association。若培训参与由求职动机、能力、家庭资源或地区机会共同影响，则仅凭公开观测数据不能推出“培训导致就业率提高”。

因此，本任务中的正确拒绝不是“模型知识不足”，而是 **causal identification failure**。这使得因果理论不是装饰，而是定义标签的基础：

- 若 public evidence 足以识别且支持 claim，标签才可能是 `valid`。
- 若 public evidence 或 gold evidence 直接反驳 claim，标签是 `invalid`。
- 若存在至少两个与 public evidence 相容、但对目标因果查询给出不同结论的 causal model，则 claim 在当前信息下是 `unidentifiable`。

这也是为什么仅靠 RAG、CoT、普通校准、普通 debate 或事实检索不能替代本任务。

## 5. 为什么主贡献必须是 benchmark，而不是 verifier

当前项目中已有 Traitor / Scientist / Auditor / Jury / difficulty / evolution 等复杂系统，但这些不应成为论文主线。原因是：

- 多智能体博弈系统本身无法直接回答“解决了什么现实问题”。
- 当前 verifier 更像工程组合和规则化诊断工具，不足以单独作为 AAAI / ICLR 级方法论文。
- 领域真正缺的是一个能稳定暴露并量化 unsafe acceptance 的任务与 benchmark。

因此，论文应采用：

> Benchmark-first, reference-verifier-assisted.

也就是说：

- benchmark 是主贡献；
- verifier 是 reference method；
- multi-agent 机制最多作为样本生成、攻击生成或诊断分析组件；
- DSR、game balance、jury、evolution 不能作为主实验指标。

## 6. Benchmark 定义

### 6.1 输入

每个样本包含：

```text
claim_text
public_evidence
attacker_or_claimant_rationale
observable_variables
public_assumptions_if_any
domain_context
```

评测时模型只能看到 public view，不能看到 gold view。

### 6.2 evaluator-only 信息

每个样本另有 gold view：

```text
source_dataset_or_paper
source_citation
hidden_variables_or_design_details
gold_scm_or_design_specification
identifying_assumptions
label_derivation
countermodel_witness_if_unidentifiable
human_audit_record
```

### 6.3 输出标签

模型必须输出：

```text
valid
invalid
unidentifiable
```

并给出结构化理由：

```text
decision
causal_query
identified_assumptions
evidence_used
missing_information
refusal_reason_if_any
confidence
```

### 6.4 核心指标

主指标不应是普通 accuracy，而应是：

- `unsafe_acceptance_rate`: 对 `unidentifiable` 或 `invalid` 样本错误输出 `valid` 的比例。
- `wise_refusal_recall`: 对 `unidentifiable` 样本正确输出 `unidentifiable` 的比例。
- `wise_refusal_precision`: 输出 `unidentifiable` 时真实为 `unidentifiable` 的比例。
- `over_refusal_rate`: 对 `valid` 样本错误拒绝的比例。
- `macro-F1`: 三类总体表现。
- `calibration`: 置信度与正确性的关系。
- `OOD robustness`: 在图结构、领域、话术和证据体制变化下的性能。

## 7. 数据集设计：synthetic core + real-grounded subset

### 7.1 为什么必须双轨

只做 synthetic 不够，因为审稿人会质疑现实相关性。只做真实数据也不够，因为真实数据难以严格控制 hidden structure、countermodel witness 和 coverage completeness。

因此必须采用双轨结构：

> synthetic core 负责严格性、标签可验证性和 coverage completeness；real-grounded subset 负责现实相关性、跨领域代表性和外部可信 grounding。

### 7.2 Synthetic Core

建议规模：600-1000 条。

每条 synthetic 样本必须包含：

- gold SCM；
- public/gold partition；
- observable variables；
- target causal query；
- label derivation rule；
- 若为 `unidentifiable`，提供 compatible countermodel witness；
- 若为 `valid`，提供识别公式或明确设计依据；
- 若为 `invalid`，提供反驳依据。

Synthetic core 覆盖矩阵：

```text
Pearl level
× label
× failure mechanism
× evidence regime
× persuasion style
× OOD axis
```

核心 failure mechanisms 至少包括：

- hidden confounding；
- selection bias / collider bias；
- invalid instrument；
- frontdoor partial measurement；
- mediation misinterpretation；
- heterogeneity overgeneralization；
- extrapolation outside support；
- post-treatment control；
- counterfactual ambiguity；
- transportability failure；
- temporal reversal；
- proxy variable misuse。

### 7.3 Real-grounded Subset

建议规模：180-300 条。最低不能少于 100 条。

它不是直接复制现有数据集标签，而是将公开可信来源转写为本 benchmark 的任务格式。

每条 real-grounded 样本必须包含：

```text
source_dataset
source_paper_or_url
domain
claim_text
public_evidence_summary
visible_information
hidden_or_evaluator_only_information
identifying_assumptions
gold_label
label_rationale
failure_mode
human_audit_record
optional_countermodel_witness
```

### 7.4 多领域真实数据源方案

推荐覆盖 8-10 个真实领域，而不是只覆盖医疗和经济。

| 领域 | 推荐来源 | 目标样本数 | 用途 |
| --- | --- | ---: | --- |
| 跨领域因果识别 | CausalReasoningBenchmark | 35-50 | 作为真实因果 query 与识别设计骨架 |
| 经济与公共政策 | Causal Claims in Economics, causaldata, AEA / NBER replication packages | 35-50 | 构造经济论文中的 causal claim auditing |
| 医疗与临床试验 | Evidence Inference 2.0, EBM-NLP, EvidenceOutcomes | 30-45 | 构造 treatment-comparator-outcome 声明 |
| 广告与营销平台 | Criteo Uplift, Hillstrom Email | 20-30 | 构造真实 A/B test 与 uplift 因果声明 |
| 推荐系统与平台治理 | KuaiRand, KuaiRec, Open Bandit Dataset | 20-35 | 覆盖曝光偏差、随机曝光、推荐策略效果 |
| 教育 | Project STAR, Learning Mindsets, World Bank education impact evaluations | 20-30 | 覆盖教育干预与学习结果 |
| 发展经济学与社会治理 | World Bank Impact Evaluation Catalog, AEA RCT Registry | 25-40 | 覆盖贫困、农业、卫生、水资源、就业等政策干预 |
| 自然科学与生物扰动 | CausalBench single-cell, LINCS L1000 | 20-30 | 覆盖真实实验扰动和生物响应 |
| 科学与健康声明语言 | CHECKWHY, SciFact, HealthVer, COVID-Fact | 20-40 | 提供自然语言 claim/evidence 多样性，需重标 causal identifiability |

### 7.5 数据源分级

**A 类：最接近直接可用**

- CausalReasoningBenchmark；
- Evidence Inference 2.0；
- Criteo Uplift；
- KuaiRand / Open Bandit；
- ACIC / RealCause。

这些数据源有较强的因果任务结构、干预或识别设计基础，适合优先转写。

**B 类：需要显著改造**

- CHECKWHY；
- SciFact / HealthVer / COVID-Fact；
- Causal Claims in Economics；
- causaldata；
- World Bank / AEA RCT Registry。

这些数据源真实且可信，但原始标签不等于我们的 `valid / invalid / unidentifiable`，必须重新做 public/gold 切分与因果识别标注。

**C 类：只做辅助覆盖**

- CauseNet；
- Causal News Corpus；
- Tübingen cause-effect pairs；
- bnlearn networks。

这些可辅助生成语言、图结构或机制变体，但不能作为 real-grounded subset 的主证据来源。

## 8. 标注与审核协议

### 8.1 标注者任务

每个样本至少由两名标注者独立完成：

1. 判断 claim 的 causal query。
2. 判断 public evidence 中可见的变量、设计、证据和假设。
3. 判断是否存在足以识别 causal query 的公开条件。
4. 给出 `valid / invalid / unidentifiable` 标签。
5. 写出 label rationale。
6. 标记 failure mechanism。
7. 标记 public/gold partition 是否合理。

冲突样本由第三人裁决。

### 8.2 标签判定标准

`valid`：

- public evidence 中包含足够的识别设计或假设；
- claim 与估计方向、效应范围或结论一致；
- 不依赖 evaluator-only hidden information。

`invalid`：

- public evidence 或 gold evidence 明确反驳 claim；
- claim 与实验方向、估计结果、设计约束或已知机制矛盾；
- 或 claim 把 association / subgroup / proxy / post-treatment pattern 错当因果结论。

`unidentifiable`：

- public evidence 不足以识别目标 causal query；
- 存在未公开的关键识别假设；
- 或可构造至少两个与 public evidence 相容但 causal effect 不同的 countermodels。

### 8.3 审核指标

必须报告：

- inter-annotator agreement；
- label-level disagreement；
- public/gold partition disagreement；
- failure-mode disagreement；
- adjudication rate；
- human audit pass rate；
- 被删除样本比例与原因。

若 `unidentifiable` 标签 agreement 低，必须优先修订判定标准，而不是继续堆数据。

## 9. Benchmark 有效性与完备性证明

### 9.1 理论有效性

核心命题：

> 对于 causal claim C 和 public evidence E，若存在两个 causal models M1 与 M2 均与 E 相容，但它们对 C 对应的 causal query 给出不同结论，则任何只基于 E 的 verifier 都不能 soundly accept C。

这个命题说明：

- `unidentifiable` 不是主观拒答；
- 它来自 causal identification 的不可唯一性；
- unsafe acceptance 是可被严格定义的错误。

### 9.2 构造有效性

Synthetic core 中每条样本都必须由 generator 记录：

- SCM 或设计规格；
- public view；
- gold view；
- label derivation；
- countermodel witness；
- OOD 轴；
- paired-flip 对应样本。

这样可以证明 benchmark 标签不是拍脑袋。

### 9.3 范围内完备性

不能声称“覆盖整个真实世界”。正式表述应为：

> 本 benchmark 在预定义任务空间内实现 scope-bounded coverage completeness。

任务空间为：

```text
causal level
× label
× failure mechanism
× evidence regime
× persuasion style
× domain
× OOD axis
```

完备性证明方式不是证明无限现实世界，而是证明：

- 每个核心 failure mechanism 至少有 k 个 synthetic 样本；
- 每个 label 在每个主要 evidence regime 下均被覆盖；
- 每个真实领域至少覆盖两类以上 failure mechanism；
- `unidentifiable` 类至少覆盖 hidden confounding、selection、invalid IV、support violation、counterfactual ambiguity 等核心不可识别机制；
- paired-flip 样本证明标签不是由表面文本决定。

### 9.4 现实相关性

Real-grounded subset 必须证明：

- 样本来自已发表论文、公开实验、业界认可数据或权威数据目录；
- 每条样本保留 citation；
- 不复用原始标签作为最终标签；
- 通过人工审核确认转写后的 public/gold 信息切分与标签合理；
- 覆盖医疗、经济、教育、平台、公共政策、发展、自然科学等多领域。

## 10. 实验设计

### Experiment 1：强 LLM baseline 失败诊断

目的：证明现有强 LLM verifier 会在该任务上 unsafe accept。

Baseline 设置：

- direct judge；
- CoT judge；
- self-consistency；
- refusal-aware prompting；
- tool-use prompting；
- debate-style prompting；
- retrieval-augmented setting；
- strongest available commercial and open models at submission time。

要求：

- 记录模型版本、调用日期、temperature、prompt、raw response；
- 报告每个模型的 unsafe acceptance、wise refusal、over-refusal；
- 主表使用最强 baseline，不使用弱 baseline 充当主要对照。

投稿门槛：

- 若最强 baseline 在 `unidentifiable` 上 unsafe acceptance 很低，说明 benchmark 太容易，需要提高攻击强度或任务复杂度。

### Experiment 2：Benchmark 可靠性实验

目的：证明 benchmark 测的是 causal oversight，而不是模板识别。

必须包含：

- leakage study：gold-only 信息加入后性能显著上升；
- public-only vs gold-view 对比；
- paired-flip：最小改动 public/gold 条件导致标签变化；
- OOD split：按 domain、graph family、failure mechanism、persuasion style 切分；
- contamination check：检查 claim、答案、模板是否可能泄漏；
- surface-bias probe：移除领域词、改写 claim、打乱话术后观察性能。

投稿门槛：

- 若模型主要依赖模板、关键词或领域先验解题，则 benchmark 不合格。

### Experiment 3：Real-grounded subset human audit

目的：证明真实子集的标签和信息切分可信。

必须报告：

- 数据源分布；
- 领域分布；
- label 分布；
- failure-mode 分布；
- 双人标注 agreement；
- 第三人裁决比例；
- 被删除样本比例；
- 标注指南和典型案例。

投稿门槛：

- 若 public/gold partition 经常被标注者认为不合理，则必须重写样本。

### Experiment 4：Reference verifier 诊断价值

目的：证明 countermodel-grounded reasoning 与任务失败机制对齐。

Reference verifier 包含：

- claim parser；
- causal query extractor；
- assumption ledger；
- identification checker；
- countermodel search；
- abstention decision；
- tool-backed evidence checking。

Ablation：

- no-countermodel；
- no-assumption-ledger；
- no-abstention；
- no-tools；
- no-public/gold separation；
- no-causal-identification module。

投稿门槛：

- 若 verifier 只提升 accuracy 但不降低 unsafe acceptance，则不能作为有效 reference method。

### Experiment 5：攻击与说服鲁棒性

目的：证明 unsafe acceptance 会被自然语言说服性策略放大。

攻击风格：

- authority framing；
- confidence inflation；
- selective evidence；
- statistical significance emphasis；
- mechanism storytelling；
- consensus framing；
- hidden-assumption concealment；
- causal overgeneralization。

报告：

- 每种攻击风格下的 unsafe acceptance；
- 哪些模型最容易被说服；
- 哪些 failure mechanism 最难拒绝。

## 11. 论文叙事重写

### 11.1 不再作为主线的内容

以下内容不应出现在标题、摘要、贡献列表和主实验叙事中：

- Causal Traitor 作为戏剧化主叙事；
- 多智能体博弈；
- jury；
- evolution；
- DSR；
- game balance；
- difficulty controller。

这些最多出现在 appendix 或 implementation details 中。

### 11.2 主叙事

论文应按以下逻辑展开：

1. 现实中高风险 causal claims 常处于信息不对称。
2. verifier 只能看到 public evidence。
3. public evidence 不足时，正确行为是 `unidentifiable`，不是强行判断。
4. 现有 benchmark 没有系统评估 causal unidentifiability refusal。
5. 我们提出 benchmark，使 unsafe acceptance 可测量、可复现、可比较。
6. 强 LLM 在该 benchmark 上仍有显著 unsafe acceptance。
7. countermodel-grounded reference verifier 能降低这类失败，但不被声称为最终方法。

### 11.3 推荐标题方向

可选标题：

- `When Causal Claims Are Not Identifiable: Benchmarking LLM Verifiers under Information Asymmetry`
- `Selective Causal Claim Auditing under Information Asymmetry`
- `Do LLM Verifiers Know When Causal Evidence Is Insufficient?`
- `Unsafe Acceptance of Unidentifiable Causal Claims by Large Language Models`

## 12. 对老师问题的显式回答

### 问题 1：我们到底解决什么现实问题？

我们解决的是：在医疗、政策、教育、平台治理、广告、经济和科学研究等高风险场景中，LLM verifier 面对只有公开证据的因果声明时，可能错误接受实际上不可识别或证据不足的 causal claim。

这不是普通 fact-checking，而是信息不对称下的 causal oversight。

### 问题 2：现有场景下存在什么不足？

现实审核者常看不到完整实验设计、隐藏混杂、选择机制、内部日志或完整 SCM。现有 LLM 又倾向于把相关性、显著性、权威表述和流畅因果叙事当成因果成立证据，导致 unsafe acceptance。

### 问题 3：现有工作为什么不能解决？

现有 causal benchmark 多测“会不会做因果题”，fact verification 多测“文本证据支不支持”，refusal benchmark 多测“一般不知道”。它们没有把 public/gold 信息不对称、因果不可识别、wise refusal 和 unsafe acceptance 统一成一个标准评测任务。

### 问题 4：为什么因果是必要的？

因为本任务的正确答案由 causal identification 决定。即使 public evidence 很多，只要存在与其相容但给出不同因果效应的 countermodels，claim 就不能被 soundly accepted。

### 问题 5：为什么选择 benchmark，而不是直接提出 verifier？

因为当前领域首先缺少标准化任务来暴露和量化该失败模式。没有 benchmark，任何 verifier 的改进都缺少稳定评估对象。我们的 verifier 当前更适合作为 reference method，而不是主贡献。

### 问题 6：如何证明 benchmark 有效？

通过四层证据：

1. 理论命题定义 `unidentifiable` 和 unsafe acceptance。
2. synthetic core 提供 SCM、label derivation 和 countermodel witness。
3. coverage matrix 证明预定义任务空间内的范围完备性。
4. real-grounded subset + human audit 证明现实相关性和标签可信度。

### 问题 7：如何证明 benchmark 完备？

我们不证明对整个真实世界完备，而证明对预定义任务空间的 scope-bounded completeness。也就是对主要 Pearl level、label、failure mechanism、evidence regime、persuasion style、domain 和 OOD axis 都有系统覆盖。

### 问题 8：我们的主要贡献到底是什么？

主要贡献是 benchmark 和任务定义，不是复杂系统本身。准确表述是：

> 我们提出第一个专门面向信息不对称因果声明审计、评估 LLM 对不可识别因果声明是否会 unsafe accept 的 public/gold-partitioned benchmark，并通过跨领域真实 grounding、严格标签协议和真实 LLM baseline 证明其诊断价值。

如果审稿阶段发现“第一个”表述风险较高，应改为：

> 我们提出一个系统化 benchmark，专门评估信息不对称下 causal unidentifiability refusal 这一此前未被充分覆盖的失败模式。

## 13. 投稿前硬性验收表

| 模块 | 最低门槛 | 理想门槛 |
| --- | --- | --- |
| Synthetic core | 600 条 | 1000+ 条 |
| Real-grounded subset | 100 条 | 180-300 条 |
| 真实领域 | 5 个 | 8-10 个 |
| 真实 LLM baseline | 5 个模型 | 商业闭源 + 开源强模型全覆盖 |
| Prompt settings | 4 类 | 6-8 类 |
| Human audit | 双人标注 + 裁决 | 双人标注 + 专家抽检 |
| Leakage study | public/gold 对比 | 加 contamination、surface-bias、paired-flip |
| OOD split | 至少 2 个轴 | domain / graph / mechanism / persuasion 全覆盖 |
| Reference verifier | 完整 ablation | 证明 unsafe acceptance 显著下降 |
| 论文叙事 | benchmark-first | 多智能体叙事完全降级 |

## 14. 如果执行后仍不达标，如何调整

1. **若强 LLM 表现太好**  
   说明 benchmark 太容易。应增加隐藏混杂、无效 IV、support violation、跨领域迁移、攻击性话术和 paired-flip 难度。

2. **若 human agreement 太低**  
   说明标签定义不清。应缩小任务范围，重写 annotation guideline，并删除争议样本。

3. **若 real-grounded subset 难以覆盖全部标签**  
   不要强行均衡。可以保持 real-grounded subset 偏真实分布，由 synthetic core 负责标签和机制均衡。

4. **若 reference verifier 不强**  
   不把 verifier 写成贡献，只保留 benchmark 与 failure analysis。

5. **若审稿人质疑新颖性**  
   强调不是 generic causal benchmark，也不是 generic refusal benchmark，而是 public/gold information asymmetry 下的 causal unidentifiability refusal。

6. **若审稿人质疑现实外推**  
   强调 scope-bounded claim，并用多领域 real-grounded subset、citation、human audit 支撑现实相关性。

## 15. 最终判断

V3 方案相较旧方案的关键进步是：

- 从“复杂系统”收敛到“明确问题”；
- 从“多智能体博弈”收敛到“信息不对称因果声明审计”；
- 从“做一个 verifier”收敛到“建立 benchmark 与证据闭环”；
- 从“举例说明有效”升级为“理论命题 + synthetic control + real grounding + human audit + leakage/OOD 检查”；
- 从“能跑实验”升级为“能让审稿人判断 benchmark 是否可信”。

因此，最终路线应确定为：

> 以 benchmark 为主贡献，以 causal identification failure 为理论基础，以 unsafe acceptance / wise refusal 为核心指标，以 synthetic core 证明严格性，以 multi-domain real-grounded subset 证明现实相关性，以真实强 LLM baseline 证明诊断价值，以 countermodel-grounded verifier 作为 reference method。

只要这些证据完整完成，并且实验结果确实显示强 LLM 在该任务上存在系统性 unsafe acceptance，本项目就具备 AAAI 2027 / ICLR 2027 级别投稿的真实可能性。若这些证据不能完成，则不应贸然以顶会为目标投稿。

## 参考数据源与相关工作

- CLadder: <https://arxiv.org/abs/2312.04350>
- CHECKWHY: <https://aclanthology.org/2024.acl-long.835/>
- CausalReasoningBenchmark: <https://arxiv.org/abs/2602.20571>
- Causal Claims in Economics: <https://www.causal.claims/>
- Evidence Inference 2.0: <https://aclanthology.org/2020.bionlp-1.13/>
- EBM-NLP: <https://aclanthology.org/P18-1019/>
- EvidenceOutcomes: <https://arxiv.org/abs/2506.05380>
- Criteo Uplift Prediction Dataset: <https://ailab.criteo.com/criteo-uplift-prediction-dataset/>
- Hillstrom MineThatData Email Challenge: <https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html>
- ACIC 2019 Data Challenge: <https://sites.google.com/view/ACIC2019DataChallenge/data-challenge>
- RealCause: <https://arxiv.org/abs/2011.15007>
- causaldata R package: <https://search.r-project.org/CRAN/refmans/causaldata/html/00Index.html>
- DoWhy examples: <https://petergtz.github.io/dowhy/main/example_notebooks/nb_index.html>
- KuaiRand: <https://kuairand.com/>
- KuaiRec: <https://arxiv.org/abs/2202.10842>
- Open Bandit Dataset: <https://arxiv.org/abs/2008.07146>
- World Bank Impact Evaluation Microdata Catalog: <https://microdata.worldbank.org/catalog/impact_evaluation>
- AEA RCT Registry: <https://www.aeaweb.org/journals/policies/rct-registry>
- CausalBench single-cell perturbation benchmark: <https://github.com/causalbench/causalbench>
- SciFact: <https://arxiv.org/abs/2004.14974>
- HealthVer: <https://aclanthology.org/2021.findings-emnlp.297/>
- COVID-Fact: <https://aclanthology.org/2021.acl-long.165/>
