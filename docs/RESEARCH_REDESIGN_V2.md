# Causal Traitor 方案深度优化 v2

## 1. 最核心的结论

### 1.1 方向有价值，但“原始版本”不够安全

截至 **2026 年 4 月**，你们关心的方向并不是空白地带，而是一个正在快速升温、但仍然存在关键缺口的前沿交叉区：

- LLM 因果推理 benchmark
- LLM + causal discovery / latent confounder
- multi-agent debate / scalable oversight
- fresh benchmark / anti-contamination evaluation

这意味着两件事同时成立：

1. **这个方向是有价值的，绝对不是过时题。**
2. **如果你们不进一步收缩问题定义，论文会非常容易撞进已有工作。**

所以，真正的问题不是“做不做这个方向”，而是：

> 你们要把项目收缩到哪个**足够尖锐且仍然空缺**的问题版本上。

### 1.2 我的最终判断

我认为你们**可以继续做下去**，但前提是立刻把论文主线从：

> 多智能体因果欺骗系统

收缩为：

> **信息不对称下的对抗性因果监督（Adversarial Causal Oversight）**

再更具体一点：

> **当攻击者掌握隐藏因子或结构先验，并用自然语言包装错误或不可识别的因果 claim 时，LLM verifier 能否仅凭可观测证据正确输出 `valid / invalid / unidentifiable`，并给出可验证的 countermodel witness？**

这是我目前看下来，**最有新颖性、最有理论抓手、最能复用你们现有资产、也最像顶会论文**的版本。

## 2. 最新文献核查后的新颖性判断

### 2.1 现在已经有人做过什么

#### A. 因果 benchmark 已经不是空白

- **CLadder (NeurIPS 2023)** 已经把 Pearl 三层自然语言化 benchmark 做出来了。
- **Reality or Mirage? (NeurIPS 2024)** 进一步指出很多 LLM 因果能力可能只是浅层模式匹配，并且强调了 fresh benchmark 的必要性。
- **CausalPitfalls (arXiv:2505.13770, v2 on 2026-03-04)** 明确开始做“统计陷阱”层面的因果 benchmark，比如 Simpson's paradox、selection bias。
- **CausalFlip (arXiv:2602.20094, 2026-02-23)** 已经开始针对“semantic matching”做反制，构造语义相近但因果答案相反的问题对。

结论：

> 单纯做“LLM 因果 benchmark”已经不够新。

#### B. 因果 + debate 也已经不是空白

- **MAD (ICML 2024)** 证明了 multi-agent debate 对推理和 factuality 有帮助。
- **Scalable Oversight (NeurIPS 2024)** 明确研究了 weak judge / strong agents / information asymmetry。
- **CRAwDAD (arXiv:2511.22854, v2 on 2026-03-09)** 已经把 dual-agent debate 用到了 CLadder 上，并报告了准确率提升。

结论：

> 单纯做“因果 + 辩论”也已经不够新。

#### C. 隐变量 / latent confounder + LLM 也已经有人在做

- **Causal Discovery via Conditional Independence Testing with Proxy Variables (ICML 2024)** 说明代理变量与 latent confounder 识别已经是正统研究前沿。
- **Efficient and Trustworthy Causal Discovery with Latent Variables and Complex Relations (ICLR 2025)** 强调 latent-variable causal discovery 不仅要有效，还要 trustworthy，甚至在假设失效时发 error signal。
- **Latent Variable Causal Discovery under Selection Bias (ICML 2025)** 把 latent variable + selection bias 这个更难的设置往前推进了一步。
- **VIGOR+ (arXiv:2512.19349, 2025-12-22)** 已经开始做 LLM 生成 hidden confounder，再用统计模型迭代验证。
- **TLVD (arXiv:2602.14456, 2026-02-16)** 用 multi-LLM collaboration 做 latent variable inference，并强调 traceability。

结论：

> 单纯做“LLM 发现隐变量”或“LLM 参与 latent variable 推理”也不够新。

#### D. 社区已经开始质疑 LLM 在因果问题中的真实作用

- **LLM Cannot Discover Causality... (arXiv:2506.00844, 2025-06-01)** 直接指出，prompt engineering 甚至 ground-truth knowledge injection 会夸大 LLM 在 causal discovery 中的表现。
- **Unbiased Evaluation of LLMs from a Causal Perspective (ICML 2025)** 强调 evaluation bias 与 benchmark contamination。

结论：

> 任何看起来像 oracle leakage、benchmark contamination、prompt 泄题的设计，都会非常危险。

### 2.2 你们仍然可能占据的“真空白”

我目前没有检索到和你们未来最优版本**高度同构**的工作。这里我必须明确说明：

> 这是一种**基于现有检索结果的推断**，不是数学意义上的“绝对无人做过”。

我认为仍然存在的空白是：

#### 空白 1：`valid / invalid / unidentifiable` 三分类的对抗性因果 claim verification

现有很多工作在做：
- causal QA
- causal discovery
- debate 提升 accuracy
- latent confounder generation

但我没有看到一个成熟工作把重点放在：

- **自然语言因果 claim verification**
- **攻击者拥有额外隐藏信息**
- **judge / verifier 只能看可观测证据**
- **输出空间显式包含 `unidentifiable`**

这点非常重要，因为这比普通因果 QA 更接近真实科研和真实审稿。

#### 空白 2：countermodel-grounded verification

我没有检索到已经成熟成型的工作，把 verifier 的核心放在：

- 构造与观测分布一致
- 但对目标 query 给出相反答案
- 从而拒绝或降级 claim

也就是：

> **用“可观测一致但因果答案相反”的 countermodel 作为 verifier 的主要证据对象。**

这会比普通 chain-of-thought、tool-augmented judging、甚至普通 debate 更有理论抓手。

#### 空白 3：leakage-free adversarial causal oversight benchmark

CLadder、CausalProbe、CausalPitfalls、CausalFlip 都各自很强，但我还没看到一个 benchmark 同时强调：

- 信息不对称
- hidden confounding / selection bias / proxy availability
- adversarial persuasive claim
- abstention / unidentifiability
- anti-contamination freshness
- explicit witness / countermodel annotation

这可以成为你们非常强的 benchmark 贡献。

### 2.3 新颖性最终判词

如果你们坚持原版“大而全的多智能体因果叛徒系统”，**新颖性不够安全**。

如果你们改成：

> `Adversarial Causal Oversight + Countermodel-Grounded Verification + Leakage-Free Benchmark`

那么我认为这条线在 **2026 年 4 月** 仍然是：

- **足够新**
- **足够前沿**
- **有明显研究价值**
- **也比原方案更容易形成顶会论文叙事**

## 3. 最严苛视角下的全项目问题清单

下面这一节按“论文致命性”排序，而不是按代码层轻重排序。

### 3.1 一级问题：会直接破坏论文可信度

#### 问题 A：信息不对称设定被实现破坏

当前 `CausalScenario` 同时暴露：
- `true_dag`
- `hidden_variables`
- `full_data`
- `true_scm`

见 [game/types.py](C:/Users/njb18/Desktop/causal-traitor/game/types.py)。

而 `Agent B / ToolExecutor / Agent C` 会直接消费这些字段：
- [agents/agent_b.py#L369](C:/Users/njb18/Desktop/causal-traitor/agents/agent_b.py#L369)
- [agents/agent_b.py#L416](C:/Users/njb18/Desktop/causal-traitor/agents/agent_b.py#L416)
- [agents/tool_executor.py#L223](C:/Users/njb18/Desktop/causal-traitor/agents/tool_executor.py#L223)
- [agents/tool_executor.py#L581](C:/Users/njb18/Desktop/causal-traitor/agents/tool_executor.py#L581)

这意味着：

- 论文里说“只有 A 知道隐藏信息”
- 但实现里 B/C 其实也知道

这是顶会视角下最危险的问题。

#### 问题 B：Agent B 的初始假设用了 oracle hints

`Agent B.propose_hypothesis()` 会从 `ground_truth` 里直接拿：
- `instrument`
- `mediator`
- `observational_difference`
- `observational_slope`

见 [agents/agent_b.py#L274](C:/Users/njb18/Desktop/causal-traitor/agents/agent_b.py#L274)。

这相当于在真实评估中提前给科学家透题。

#### 问题 C：工具链用真图 / 真SCM 验证，无法支撑“现实可用”主张

例如：
- `causal_graph_validator` 在真图上跑
- `backdoor_adjustment_check` 直接拿真 graph 验证
- `counterfactual_inference` 直接拿 `true_scm`

相关位置：
- [causal_tools/meta_tools.py#L145](C:/Users/njb18/Desktop/causal-traitor/causal_tools/meta_tools.py#L145)
- [causal_tools/l2_intervention.py#L112](C:/Users/njb18/Desktop/causal-traitor/causal_tools/l2_intervention.py#L112)
- [agents/tool_executor.py#L604](C:/Users/njb18/Desktop/causal-traitor/agents/tool_executor.py#L604)

这会导致你们的 judge 看起来很强，但强在“知道答案结构”，不是强在“真的能监督”。

### 3.2 二级问题：会让论文 claim 被审稿人拆穿

#### 问题 D：实验标签空间不对

现在多数实验还是：
- `agent_a` 赢
- `agent_b` 赢

而不是：
- `valid`
- `invalid`
- `unidentifiable`

这会把一个因果识别问题错误地包装成博弈胜负问题。

#### 问题 E：`unidentifiable` 缺失

当前系统默认任何一轮都要分胜负。这会和 Pearl-style identification 的核心精神冲突。

很多情况下正确答案应该是：

> 仅凭当前观测信息，无法唯一支持该 claim。

如果没有这个输出空间，论文会天然偏向过度裁决。

#### 问题 F：评价指标过度系统化、欠科学化

DSR、Flow、Arms Race、Game Balance 等作为 demo 指标没问题，但如果它们占据 main table，就会显得偏工程秀。

### 3.3 三级问题：会削弱结果的统计可信度

#### 问题 G：实验没有多 seed / 置信区间 / 显著性

当前实验脚本本质上都是单次运行汇总：
- [experiments/exp1_causal_levels/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp1_causal_levels/run.py)
- [experiments/exp2_jury_ablation/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp2_jury_ablation/run.py)
- [experiments/exp3_difficulty/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp3_difficulty/run.py)
- [experiments/exp4_evolution/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp4_evolution/run.py)

没有：
- seed sweep
- bootstrap CI
- significance test
- family-level OOD split

#### 问题 H：benchmark 太小、模板太固定

当前核心故事只有 3 类：
- smoking_cancer
- education_income
- drug_recovery

这更像 demo scenario，不像 benchmark。

#### 问题 I：freshness / contamination 机制不足

近两年文献已经在反复强调 contamination 风险。如果你们后面还是用非常固定的模板、固定变量名、固定叙事，很容易让结果失真。

### 3.4 四级问题：实现与文档之间存在错位

#### 问题 J：进化机制实际没有按设计工作

`DebateEngine` 传给 `AgentA.adapt_strategy()` 的字段是：
- `detected`
- `strategy_used`

见 [game/debate_engine.py#L449](C:/Users/njb18/Desktop/causal-traitor/game/debate_engine.py#L449)

但 `AgentA.adapt_strategy()` 读取的是：
- `caught`
- `strategy`

见 [agents/agent_a.py#L213](C:/Users/njb18/Desktop/causal-traitor/agents/agent_a.py#L213)

这意味着 avoid-set 很可能根本没按预期更新。

#### 问题 K：配置和实际默认值不一致

`configs/default.yaml` 里 `adjustment_rate: 0.1`

见 [configs/default.yaml#L42](C:/Users/njb18/Desktop/causal-traitor/configs/default.yaml#L42)

但 `DifficultyController` 默认值是 `0.18`

见 [game/difficulty.py#L19](C:/Users/njb18/Desktop/causal-traitor/game/difficulty.py#L19)

这会让“文档里的实验设定”和“实际运行机制”不完全一致。

#### 问题 L：部分评估代码本身就不稳

`MetricResult` 需要 `name/value/category`

见 [evaluation/metrics.py#L20](C:/Users/njb18/Desktop/causal-traitor/evaluation/metrics.py#L20)

但 `Scorer.score_round()` 里构造 `MetricResult` 时没有给 `category`

见 [evaluation/scorer.py#L141](C:/Users/njb18/Desktop/causal-traitor/evaluation/scorer.py#L141)

如果这段真正跑起来，会直接是实现问题。

#### 问题 M：某些“因果量”名字比方法更强

例如 `probability_of_necessity` / `probability_of_sufficiency`

见 [causal_tools/l3_counterfactual.py#L148](C:/Users/njb18/Desktop/causal-traitor/causal_tools/l3_counterfactual.py#L148)

当前实现更像观测分布下的启发式近似，而不是严格识别的 PN / PS。这个命名如果直接写进论文，会被因果方向审稿人抓得很紧。

## 4. 哪些东西应该保留，哪些必须砍

### 4.1 高价值保留资产

这些我建议尽量保留，因为它们可直接转化为论文资产：

1. **Pearl 三层任务骨架**
2. **SCM 场景生成器**
3. **工具执行与工具编排框架**
4. **对抗交互协议**
5. **实验脚本外壳**
6. **可视化系统**

### 4.2 可以保留但必须降级的资产

这些不是没有价值，而是不要再当 main contribution：

1. `jury`
2. `difficulty`
3. `evolution`
4. `game balance` 类指标

它们可以作为：
- appendix
- ablation
- demo

### 4.3 必须推翻或重写的资产

1. **oracle-style information access**
2. **winner-only evaluation**
3. **把 true graph 当 verifier 可见输入**
4. **把启发式值写成严格因果量**

## 5. 最优论文路线

### 5.1 我建议的最终 paper package

#### Contribution 1: 新任务

`Adversarial Causal Oversight`

输入：
- observed data
- optional proxies
- adversarial natural-language causal claim
- optional debate transcript

输出：
- `valid / invalid / unidentifiable`
- confidence
- assumption ledger
- witness / countermodel

#### Contribution 2: 新 benchmark

要求同时具备：
- Pearl L1/L2/L3
- hidden confounder
- selection bias
- proxy variable
- adversarial claim generation
- fresh split
- OOD split
- witness annotation

#### Contribution 3: 新 verifier

核心步骤：
- claim parsing
- assumption ledger
- countermodel search
- tool-backed adjudication

### 5.2 为什么这条路线比原版强

因为它同时满足四件事：

1. **和最新文献有明确区别**
2. **有理论命题可写**
3. **能最大化复用你们现有代码**
4. **更像一篇论文，而不是一个系统作品集**

## 6. 最大化复用现有成果的改造策略

### 6.1 不推倒重来，而是“分层复用”

#### 第一层：直接复用

- `game/data_generator.py`
- `causal_tools/`
- `agents/tool_executor.py`
- `experiments/` 的脚本骨架
- `visualization/`

#### 第二层：轻改后复用

- `Agent A`
  从“叛徒”改成 attacker / claim generator
- `Agent C`
  从“裁判”改成 verifier-first
- `Agent B`
  从“科学家”改成 baseline debater / proposer

#### 第三层：仅保留为 appendix

- jury
- difficulty
- evolution

### 6.2 最省开发量的落地顺序

1. 先不动 UI
2. 先不动 live game
3. 先不做更多角色
4. 先做 clean benchmark schema
5. 先做 information partition
6. 先把 verifier 改出来
7. 最后再把 jury / evolution 重新接回 appendix

## 7. 新方案的具体研究问题

我建议最终把论文研究问题写成下面这个版本：

> Can LLM-based verifiers reliably audit adversarial causal claims under information asymmetry, especially when hidden confounding, selection bias, and non-identifiability make persuasive explanations easy but correct adjudication hard?

这个版本有几个优点：

- 不和 CRAwDAD 直接撞 “debate improves CLadder accuracy”
- 不和 TLVD 直接撞 “latent variable semantic inference”
- 不和 CausalPitfalls / CausalFlip 直接撞 “benchmark only”
- 不和传统 causal discovery 直接撞 “recover graph from data”

## 8. 论文题目建议

### 最稳妥版本

- **Adversarial Causal Oversight under Information Asymmetry**

### 更方法导向版本

- **Countermodel-Grounded Verification for Adversarial Causal Claims**

### 更 benchmark 导向版本

- **A Leakage-Free Benchmark for Adversarial Causal Claim Verification**

`Causal Traitor` 可以保留为项目名或系统名，不建议直接做论文主标题。

## 9. 接下来最值得做的事

### P0

冻结论文主线，停止继续增加系统模块。

### P1

重构 benchmark schema：
- hidden confounder
- proxy
- selection bias
- valid / invalid / unidentifiable

### P2

彻底切断 oracle leakage。

### P3

把 verifier 改成 countermodel-first。

### P4

重做实验矩阵：
- main benchmark
- robustness
- leakage study
- identifiability ablation
- transfer
- human audit

## 10. 最终判词

在你提出的四点约束下，我的最终判断是：

1. **最新 arXiv 必须看，而且看完后更说明你们不能再做泛泛的大系统叙事。**
2. **这个方向仍然值得继续，但只有在“对抗性因果监督”这个更窄的版本上才足够新。**
3. **项目里确实有不少细微但危险的问题，尤其是 oracle leakage、标签空间错误、理论定位不稳、实验统计性不足。**
4. **现有成果可以保留相当大一部分，但必须按“主论文资产 / appendix 资产 / demo 资产”重新分层。**

一句话总结：

> 你们现在不是没有希望，而是已经站在一个还不错的起点上；但如果想冲顶会，就必须从“复杂系统堆叠”切换到“新任务 + 新 verifier + 新 benchmark”的论文思维。
