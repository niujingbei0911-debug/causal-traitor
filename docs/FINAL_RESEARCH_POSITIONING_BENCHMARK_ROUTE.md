# 项目论文方案最终思考：从“多智能体因果博弈”收敛到“信息不对称下的因果声明审计 Benchmark”

## 一、最终定位

当前项目不应再被定义为“多智能体因果辩论系统”，也不应主要定义为“提出一个新的 verifier 方法”。更合理、更稳健的定位是：

> 我们提出一个面向信息不对称因果声明审计的新评测任务与 benchmark，用于评估 LLM verifier 在只能看到公开证据时，是否能够正确区分 `valid / invalid / unidentifiable` 的因果声明，尤其避免在公开证据不足时做出 unsafe acceptance。

因此，论文主贡献应为：

1. **新任务**：Selective Adversarial Causal Oversight under Information Asymmetry。
2. **新 benchmark**：public/gold partitioned, leakage-free causal claim verification benchmark。
3. **参考方法**：Countermodel-Grounded Selective Verification，作为 reference verifier，而不是主贡献本身。

## 二、现实问题是什么

现实中大量高风险判断都以自然语言因果声明形式出现，例如：

- 医疗机构声称某治疗方案导致康复率提升；
- 政策部门声称某培训项目提高就业率；
- 平台公司声称某推荐机制降低有害内容暴露；
- 企业声称某广告策略带来销售增长。

这些场景的共同结构是：**claim 提出方掌握更多内部信息，而外部审核者只能看到公开证据**。公开证据可能包括观测数据、报告摘要、部分统计结果和一段很有说服力的因果解释，但审核者看不到隐藏混杂变量、选择机制、完整 SCM、原始实验设计或 gold label。

因此，核心风险不是普通事实错误，而是：

> LLM verifier 把“公开证据不足以识别因果效应”的情况误判为“因果声明成立”。

这就是 unsafe acceptance。

## 三、现有工作的不足

已有工作已经覆盖了很多相关方向，因此我们不能声称“第一个 causal LLM benchmark”或“第一个 LLM refusal benchmark”。

现有方向包括：

- CLadder、CausalBench 等评估 LLM 的因果推理能力；
- CHECKWHY 做 causal fact verification；
- DoVerifier 使用 do-calculus 验证给定 causal expression；
- AbstentionBench、UA-Bench 等评估 LLM 拒答和不确定性；
- LLM-as-a-judge、AI debate、persuasion attack 等研究 judge/debate 的脆弱性。

但这些工作没有系统回答一个特定问题：

> 在信息不对称下，当 verifier 只能看到 public evidence，而 causal claim 的成立依赖隐藏结构或不可检验识别假设时，LLM 能否意识到该 claim 当前不可识别，并做出 wise refusal？

这就是我们的空白。

## 四、为什么必须引入因果，而不是普通拒答

普通拒答研究通常关注“模型知不知道答案”或“证据是否足够回答”。但因果审计的问题更特殊：

> 即使有大量观测数据，因果效应也可能不可识别。

例如，公开数据显示“参加培训的人就业率更高”，这只能支持 association。若存在隐藏混杂，例如个人能力、求职动机、家庭背景，那么不能直接推出“培训导致就业率提升”。

因此，这里的“不知道”不是模型知识不足，而是 **因果识别意义上的不可识别**。正确行为必须依赖 causal identification，而不是普通 uncertainty calibration、RAG、CoT 或 debate。

这说明因果不是装饰，而是定义正确答案的理论基础。

## 五、为什么 benchmark 路线最合理

如果走 verifier-first，需要证明我们的方法显著优于已有 verifier，并且算法本身有足够理论创新。当前项目的 verifier 仍偏规则化和工程化，单独作为顶会方法论文风险较高。

相反，benchmark-first 更符合项目已有基础，也更能回答老师的问题：

- 现实问题明确：高风险因果声明审计；
- 现有不足明确：已有 benchmark 没有系统评估 information asymmetry 下的 causal unidentifiability refusal；
- 贡献边界明确：我们不声称覆盖所有因果推理，只覆盖一个定义清楚的审计任务；
- verifier 作用明确：reference method，用于证明 benchmark 有诊断价值。

所以最终路线应为：

> Benchmark-first, reference-verifier-assisted.

## 六、Benchmark 到底评测什么

输入：

- `public_evidence`
- `claim_text`
- `attacker_rationale`
- observable variables
- optional proxy / selection / instrument hints

输出：

- `valid`
- `invalid`
- `unidentifiable`

核心指标：

- `unsafe_acceptance_rate`
- `wise_refusal_recall`
- `wise_refusal_precision`
- `over_refusal_rate`
- calibration / macro-F1 / OOD robustness

其中最核心的不是 accuracy，而是 unsafe acceptance 和 wise refusal。

## 七、如何证明 benchmark 有效

我们不能，也不应该声称 benchmark 对整个真实世界无限完备。合理主张是：

> 在预定义任务空间内，本 benchmark 对核心失败模式具有 scope-bounded completeness，并且通过 synthetic core 与 real-grounded subset 同时证明形式严谨性和现实相关性。

证明分四层：

1. **理论有效性**

   若存在两个与 public evidence 相容、但对目标 causal query 给出不同答案的 SCM，则 verifier 不能 soundly accept 该 claim。

   这证明 `unidentifiable` 是因果识别理论要求，不是随意添加的第三类标签。

2. **构造有效性**

   synthetic core 中每个样本都有 gold SCM、public/gold partition、gold label、countermodel witness 或 identifying assumption basis。

   这证明标签不是人工拍脑袋。

3. **覆盖有效性**

   预先定义 failure-mode taxonomy：hidden confounding、selection bias、invalid IV、frontdoor partial measurement、heterogeneity overgeneralization、counterfactual ambiguity 等。

   用 coverage matrix 证明每个核心 cell 都被覆盖。

4. **现实相关性**

   建立 real-grounded subset，来自医疗、经济、教育、公共政策、平台治理等领域。每条样本包含 citation、public evidence、claim、识别假设、隐藏信息说明、标签依据，并进行 human audit。

## 八、完备性如何表述

不能说“覆盖整个真实世界”。应说：

> 我们在一个明确限定的任务空间内证明 coverage completeness。

任务空间可以定义为：

```text
Pearl causal level
× identification status
× failure mechanism
× evidence regime
× persuasion style
× OOD axis
```

即：

- L1 / L2 / L3；
- valid / invalid / unidentifiable；
- hidden confounding / selection bias / invalid IV / counterfactual ambiguity 等；
- sufficient / contradictory / underdetermined public evidence；
- authority / confidence / consensus / concealment pressure；
- graph-family / mechanism / attack-family / context / paired-flip OOD。

只要 benchmark 对这个定义空间中的核心机制有系统覆盖，就可以声称 scope-bounded completeness。

## 九、reference verifier 的作用

Countermodel-Grounded Selective Verification 不是主贡献，但非常重要。它证明这个 benchmark 不只是“出题”，还能推动新的 verifier 设计。

核心思想：

> 如果存在一个与 public evidence 兼容、但对 causal query 给出不同答案的 countermodel，则 claim 不能被当前公开证据唯一支持。

因此 verifier 的流程是：

1. 解析 claim；
2. 建立 assumption ledger；
3. 搜索 compatible countermodel；
4. 若 countermodel 存在，拒绝 unsafe acceptance；
5. 若识别假设 unresolved，输出 unidentifiable；
6. 只有识别条件和工具证据共同支持时，才输出 valid。

这直接对应因果识别问题的内在结构。

## 十、需要补强的工作

当前项目要真正说服老师和审稿人，还必须补：

1. 接入真实 LLM baselines，而不是只用规则模拟 baseline；
2. 建立 100-300 条 real-grounded subset；
3. 做 human audit，验证标签、public/gold 分区、witness 是否合理；
4. 做 leakage study，证明 gold-only 信息会显著虚高性能；
5. 做 ablation，证明 no-countermodel / no-abstention / no-ledger 会退化；
6. 做 persuasion robustness，证明现有 LLM judge 会被说服性话术影响。

## 十一、对老师问题的显式回答

**老师问：我们到底解决什么问题？**

答：解决 LLM 在信息不对称因果声明审计中，错误接受公开证据不足以识别的 causal claim 的问题。

**老师问：现实场景是什么？**

答：医疗、政策、教育、平台治理、企业报告等高风险因果声明审核场景，claimant 掌握更多隐藏信息，verifier 只能看公开证据。

**老师问：现有工作不足在哪里？**

答：现有 causal benchmark 多测因果题目回答能力，现有 refusal benchmark 多测一般未知问题，现有 fact verification 多测文本证据支持关系，缺少对 causal unidentifiability under information asymmetry 的系统评测。

**老师问：为什么必须用因果？**

答：因为问题成因是 causal identification failure。普通 RAG、CoT、debate、calibration 不能判断 public evidence 是否足以识别 intervention/counterfactual claim。

**老师问：为什么选择 benchmark？**

答：当前领域缺少标准化评测来暴露这一失败模式。先定义任务和 benchmark，比直接提出 verifier 更基础、更必要，也更符合当前项目已有基础。

**老师问：如何证明 benchmark 有效和完备？**

答：通过理论命题证明 unidentifiable 的必要性，通过 synthetic core 保证标签可验证，通过 taxonomy coverage matrix 证明范围内完备，通过 real-grounded subset 和 human audit 证明现实相关性。

**老师问：verifier 是不是主贡献？**

答：不是。verifier 是 reference method，用于证明 countermodel reasoning 对降低 unsafe acceptance 有帮助，并验证 benchmark 的诊断价值。

## 十二、最终结论

项目最终应从“复杂多智能体因果博弈系统”收敛为：

> 一个面向信息不对称高风险因果声明审计的、范围明确、无泄漏、可验证、带真实 grounding 的 benchmark。

它的核心价值不是“我们做了一个复杂系统”，而是：

> 我们抽象出真实世界中反复出现但现有评测没有系统覆盖的失败机制：LLM 在公开证据不足以识别因果效应时，仍被自然语言因果声明诱导而 unsafe accept。我们用因果识别理论定义正确行为，用 benchmark 让这种失败可测量、可复现、可比较。

## 参考工作

- CLadder: <https://arxiv.org/abs/2312.04350>
- CausalBench: <https://aclanthology.org/2024.sighan-1.17/>
- CHECKWHY: <https://aclanthology.org/2024.acl-long.835/>
- DoVerifier: <https://arxiv.org/abs/2601.21210>
- AbstentionBench: <https://arxiv.org/abs/2506.09038>
- AI debate: <https://arxiv.org/abs/2402.06782>
