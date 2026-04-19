# 课程展示仓库导览

这份文档的目标不是定义项目，而是帮助你在课程展示、组会汇报、老师问答时，
用一条不会跑偏的路径来介绍这个仓库。

## 一句话介绍

现在这个仓库的正确介绍方式是：

> 我们把项目从“多智能体因果欺骗系统”收缩成了一个论文主线明确的研究仓库：
> 一个 selective causal oversight 新任务、一个 leakage-free benchmark、
> 一个 countermodel-grounded selective verifier。

不要把仓库开场介绍成：

> 一个很复杂的多智能体博弈平台

因为那是旧叙事，会把重点带偏。

## 展示时推荐的仓库讲解顺序

### 第一步：先讲主线文档

按这个顺序：

1. [FINAL_CONSTRUCTION_BLUEPRINT_V2.md](C:/Users/njb18/Desktop/causal-traitor/docs/FINAL_CONSTRUCTION_BLUEPRINT_V2.md)
2. [ENGINEERING_EXECUTION_PLAN_V2.md](C:/Users/njb18/Desktop/causal-traitor/docs/ENGINEERING_EXECUTION_PLAN_V2.md)
3. [PHASE_TASK_CARDS_V2.md](C:/Users/njb18/Desktop/causal-traitor/docs/PHASE_TASK_CARDS_V2.md)
4. [PROJECT_MAP.md](C:/Users/njb18/Desktop/causal-traitor/docs/PROJECT_MAP.md)

这一段要讲清楚：

- 研究问题是什么
- 最终交付物是什么
- 哪些东西是主线，哪些只是 appendix/demo

### 第二步：讲 benchmark

进入：

- [schema.py](C:/Users/njb18/Desktop/causal-traitor/benchmark/schema.py)
- [graph_families.py](C:/Users/njb18/Desktop/causal-traitor/benchmark/graph_families.py)
- [generator.py](C:/Users/njb18/Desktop/causal-traitor/benchmark/generator.py)
- [attacks.py](C:/Users/njb18/Desktop/causal-traitor/benchmark/attacks.py)
- [witnesses.py](C:/Users/njb18/Desktop/causal-traitor/benchmark/witnesses.py)

这里重点讲：

- public/gold 信息分区
- selective outputs：`identified / contradicted / underdetermined` 与 `valid / invalid / unidentifiable`
- graph family + attack template + witness 的数据生成逻辑

### 第三步：讲 verifier

进入：

- [claim_parser.py](C:/Users/njb18/Desktop/causal-traitor/verifier/claim_parser.py)
- [assumption_ledger.py](C:/Users/njb18/Desktop/causal-traitor/verifier/assumption_ledger.py)
- [countermodel_search.py](C:/Users/njb18/Desktop/causal-traitor/verifier/countermodel_search.py)
- [decision.py](C:/Users/njb18/Desktop/causal-traitor/verifier/decision.py)
- [pipeline.py](C:/Users/njb18/Desktop/causal-traitor/verifier/pipeline.py)

这里重点讲：

1. claim parsing
2. assumption ledger
3. countermodel search
4. tool-backed adjudication
5. final decision rule

### 第四步：讲评估和实验

进入：

- [metrics.py](C:/Users/njb18/Desktop/causal-traitor/evaluation/metrics.py)
- [scorer.py](C:/Users/njb18/Desktop/causal-traitor/evaluation/scorer.py)
- [benchmark_harness.py](C:/Users/njb18/Desktop/causal-traitor/experiments/benchmark_harness.py)
- [experiments/README.md](C:/Users/njb18/Desktop/causal-traitor/experiments/README.md)

然后再依次点主实验：

- [exp_main_benchmark/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp_main_benchmark/run.py)
- [exp_leakage_study/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp_leakage_study/run.py)
- [exp_identifiability_ablation/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp_identifiability_ablation/run.py)
- [exp_adversarial_robustness/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp_adversarial_robustness/run.py)
- [exp_ood_generalization/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp_ood_generalization/run.py)
- [exp_human_audit/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp_human_audit/run.py)

这一段的讲法应该是：

- main benchmark 证明整体有效
- leakage study 证明评估必须 leakage-free
- ablation 证明关键组件必要
- robustness/OOD 证明不是只会做熟题
- human audit 证明输出解释可接受

## 不建议在前半段就展开讲的内容

这些内容不是不能讲，而是不应该太早讲：

- `jury`
- `difficulty`
- `evolution`
- `visualization`
- `main.py`
- `run_live_game.py`
- `docs/legacy/`

它们更适合放在：

- demo 展示
- appendix 说明
- 老师追问“你们原来系统还有什么”时再补充

## 如果老师直接打开仓库，你最希望他看什么

最理想的阅读路径是：

1. `README.md`
2. `docs/FINAL_CONSTRUCTION_BLUEPRINT_V2.md`
3. `docs/PROJECT_MAP.md`
4. `benchmark/`
5. `verifier/`
6. `evaluation/`
7. `experiments/README.md`

## 如果老师问“为什么仓库里还有旧系统痕迹”

推荐回答：

> 我们已经完全转向新方案。
> 旧系统代码和文档没有删，是因为它们还承担 supplemental demo、appendix 和历史演化证据的作用。
> 但项目主线、论文主线和实验主线已经全部切换到 v2 文档定义的
> selective task + benchmark + verifier 结构。
