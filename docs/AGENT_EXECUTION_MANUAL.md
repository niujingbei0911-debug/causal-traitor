# The Causal Traitor v3
## AI Agent 执行手册

> 目的：为 AI agent 系统提供可直接执行的工程实施说明  
> 依赖文档：[FINAL_CONSTRUCTION_BLUEPRINT.md](C:/Users/njb18/Desktop/causal-traitor/docs/FINAL_CONSTRUCTION_BLUEPRINT.md)  
> 使用方式：先读本手册，再按阶段派发 agent 任务  
> 适用策略：顺序执行主线，局部并行，严格控制写集冲突

---

## 1. 执行原则

### 1.1 总原则

所有 agent 的执行都必须遵守：

1. **先主线，后 demo**
2. **先 benchmark，后 verifier**
3. **先切断 oracle leakage，后做实验**
4. **先形成 clean task definition，后接回 debate / jury / evolution**

### 1.2 严格禁止

在主线完成前，禁止优先投入大量工作到：

- UI 美化
- live demo 交互增强
- jury 机制扩展
- difficulty 新策略
- evolution 新策略
- 新增更多“故事场景”但不重构 benchmark schema

### 1.3 推荐执行模式

推荐把整个施工过程拆成：

- `Sequential backbone`
  需要强依赖顺序的主线任务
- `Parallel sidecars`
  可并行但不能阻塞主线的任务

---

## 2. 最终目标目录结构

以下是建议的最终目录形态。

```text
causal-traitor/
├── benchmark/
│   ├── __init__.py
│   ├── schema.py
│   ├── generator.py
│   ├── graph_families.py
│   ├── attacks.py
│   ├── witnesses.py
│   ├── split_builder.py
│   └── loaders.py
├── verifier/
│   ├── __init__.py
│   ├── claim_parser.py
│   ├── assumption_ledger.py
│   ├── countermodel_search.py
│   ├── decision.py
│   ├── outputs.py
│   └── pipeline.py
├── agents/
│   ├── agent_a.py
│   ├── agent_b.py
│   ├── agent_c.py
│   ├── jury.py
│   ├── tool_executor.py
│   └── prompts/
├── causal_tools/
├── game/
│   ├── data_generator.py
│   ├── debate_engine.py
│   ├── types.py
│   ├── difficulty.py
│   ├── evolution.py
│   ├── llm_service.py
│   └── config.py
├── evaluation/
│   ├── metrics.py
│   ├── scorer.py
│   ├── tracker.py
│   ├── reporting.py
│   └── significance.py
├── experiments/
│   ├── exp_main_benchmark/
│   ├── exp_adversarial_robustness/
│   ├── exp_identifiability_ablation/
│   ├── exp_leakage_study/
│   ├── exp_ood_generalization/
│   ├── exp_cross_model_transfer/
│   ├── exp_human_audit/
│   ├── exp2_jury_ablation/
│   ├── exp3_difficulty/
│   └── exp4_evolution/
├── visualization/
├── tests/
│   ├── test_benchmark.py
│   ├── test_verifier.py
│   ├── test_information_partition.py
│   ├── test_evaluation.py
│   ├── test_tools.py
│   ├── test_agents.py
│   └── test_integration.py
├── docs/
│   ├── DESIGN.md
│   ├── FINAL_CONSTRUCTION_BLUEPRINT.md
│   └── AGENT_EXECUTION_MANUAL.md
├── main.py
└── run_live_game.py
```

### 2.1 目录含义

- `benchmark/`
  主论文资产，负责样本 schema、程序化生成、切分与 witness。
- `verifier/`
  主论文资产，负责 parser、ledger、countermodel、decision。
- `agents/`
  保留角色外壳，但主线会以 verifier 为中心重构。
- `game/`
  继续保留调度器和 demo 逻辑，但必须和主论文任务定义对齐。
- `experiments/`
  主论文实验与 appendix 实验分层保留。

---

## 3. 阶段总览

### 3.1 主施工阶段

必须按以下顺序执行：

1. `Phase 0`
   冻结任务定义与信息分区
2. `Phase 1`
   benchmark schema 与 public/gold view 重构
3. `Phase 2`
   verifier 核心实现
4. `Phase 3`
   实验指标与评估协议重构
5. `Phase 4`
   主实验跑通
6. `Phase 5`
   appendix 与 demo 资产回接

### 3.2 阶段依赖图

```text
Phase 0
  ↓
Phase 1 (benchmark)
  ↓
Phase 2 (verifier)
  ↓
Phase 3 (evaluation)
  ↓
Phase 4 (main experiments)
  ↓
Phase 5 (jury / difficulty / evolution / visualization reconnect)
```

---

## 4. Phase 0：任务冻结与结构防错

### 4.1 目标

在任何实现工作开始前，把“什么是任务、谁能看什么、输出是什么”固定下来。

### 4.2 需要修改或新增的文件

#### 必改

- [game/types.py](C:/Users/njb18/Desktop/causal-traitor/game/types.py)
- [docs/FINAL_CONSTRUCTION_BLUEPRINT.md](C:/Users/njb18/Desktop/causal-traitor/docs/FINAL_CONSTRUCTION_BLUEPRINT.md)

#### 新增

- `benchmark/schema.py`

### 4.3 执行动作

1. 在 `game/types.py` 中拆分现有 `CausalScenario`
   
   建议改为三类：

   - `GoldCausalInstance`
   - `PublicCausalInstance`
   - `OversightExample`

2. 显式定义：
   
   - attacker 可见字段
   - verifier 可见字段
   - evaluator 专用字段

3. 从 schema 层面禁止 verifier 读取：
   
   - `true_dag`
   - `hidden_variables`
   - `true_scm`
   - `full_data`

4. 统一输出标签空间：
   
   - `valid`
   - `invalid`
   - `unidentifiable`

### 4.4 验收标准

- 任何 verifier 入口函数都只能接收 `PublicCausalInstance`
- `GoldCausalInstance` 不会直接流入 `Agent B / Agent C / ToolExecutor`
- 代码中不存在再用 `winner == agent_b` 替代核心标签的主线逻辑

### 4.5 推荐测试

新增：

- `tests/test_information_partition.py`

至少检查：

- verifier 无法访问 gold fields
- gold/public 转换函数正确
- 非法访问会报错或在类型层被阻断

---

## 5. Phase 1：Benchmark 重构

### 5.1 目标

把“3 个固定故事”升级为“程序化 causal oversight benchmark”。

### 5.2 文件写集

#### 新增文件

- `benchmark/schema.py`
- `benchmark/generator.py`
- `benchmark/graph_families.py`
- `benchmark/attacks.py`
- `benchmark/witnesses.py`
- `benchmark/split_builder.py`
- `benchmark/loaders.py`
- `tests/test_benchmark.py`

#### 需要复用/修改

- [game/data_generator.py](C:/Users/njb18/Desktop/causal-traitor/game/data_generator.py)
- [game/types.py](C:/Users/njb18/Desktop/causal-traitor/game/types.py)

### 5.3 推荐实施顺序

#### Step 1：抽取 schema

先写 `benchmark/schema.py`，定义：

- `GoldCausalInstance`
- `PublicCausalInstance`
- `ClaimInstance`
- `Witness`
- `BenchmarkSplitManifest`

#### Step 2：抽取 graph family

在 `benchmark/graph_families.py` 中定义 family 级模板：

- `l1_latent_confounding_family`
- `l1_selection_bias_family`
- `l2_valid_backdoor_family`
- `l2_invalid_iv_family`
- `l3_counterfactual_ambiguity_family`

#### Step 3：迁移现有场景生成逻辑

把 [game/data_generator.py](C:/Users/njb18/Desktop/causal-traitor/game/data_generator.py) 中的：

- smoking
- education
- drug

从“固定场景”重构为“showcase family”。

要求：

- 这些场景保留
- 但只作为 benchmark family 中的一个可解释子族
- 不再作为唯一测试集

#### Step 4：攻击样本生成

在 `benchmark/attacks.py` 中实现：

- `association_overclaim`
- `hidden_confounder_denial`
- `invalid_adjustment_claim`
- `counterfactual_overclaim`
- `unidentifiable_disguised_as_valid`

#### Step 5：witness 生成

在 `benchmark/witnesses.py` 中生成：

- support witness
- countermodel witness
- assumption witness

#### Step 6：split builder

在 `benchmark/split_builder.py` 中实现：

- `train/dev/test_iid/test_ood` 切分
- family holdout
- lexical holdout
- variable renaming holdout

### 5.4 验收标准

- benchmark 可以在程序化 family 上批量生成样本
- 每个样本都包含 `gold_label`
- 每个样本都可导出 `public view`
- 存在 `test_ood`
- showcase 场景仍可导出用于 demo

### 5.5 并行建议

可并行：

- graph family 实现
- attacks 实现
- witness schema 实现

不可并行：

- public/gold access contract
- split manifest 最终格式

---

## 6. Phase 2：Verifier 核心实现

### 6.1 目标

实现主论文方法：

`Claim Parsing -> Assumption Ledger -> Countermodel Search -> Decision`

### 6.2 文件写集

#### 新增

- `verifier/claim_parser.py`
- `verifier/assumption_ledger.py`
- `verifier/countermodel_search.py`
- `verifier/decision.py`
- `verifier/outputs.py`
- `verifier/pipeline.py`
- `tests/test_verifier.py`

#### 需要修改

- [agents/agent_c.py](C:/Users/njb18/Desktop/causal-traitor/agents/agent_c.py)
- [agents/tool_executor.py](C:/Users/njb18/Desktop/causal-traitor/agents/tool_executor.py)
- [causal_tools/meta_tools.py](C:/Users/njb18/Desktop/causal-traitor/causal_tools/meta_tools.py)

### 6.3 推荐实施顺序

#### Step 1：Claim parser

在 `verifier/claim_parser.py` 中实现：

- query type parsing
- treatment/outcome extraction
- claim strength extraction
- assumption cue extraction
- abstention-risk flagging

输入：
- claim text
- optional transcript

输出：
- structured claim object

#### Step 2：Assumption ledger

在 `verifier/assumption_ledger.py` 中实现：

- 从 structured claim 推出识别假设
- 对每个假设给出状态：
  - supported
  - contradicted
  - unresolved

#### Step 3：Countermodel search

在 `verifier/countermodel_search.py` 中分层实现：

##### L1

- hidden confounder candidate
- reverse direction candidate
- selection candidate

##### L2

- alternative adjustment-compatible model
- weak IV / invalid IV candidate
- proxy-based alternative explanation

##### L3

- same observational fit, different counterfactual answer
- alternative SCM family

#### Step 4：Decision rule

在 `verifier/decision.py` 中固定决策顺序：

1. 强 countermodel -> `invalid`
2. 多解且 query 不一致 -> `unidentifiable`
3. 核心识别假设 unresolved -> `unidentifiable`
4. 工具和 ledger 共同支持 -> `valid`

#### Step 5：Pipeline

在 `verifier/pipeline.py` 中形成统一入口：

- parse
- ledger
- countermodel
- tools
- decide

### 6.4 Agent C 改造要求

对 [agents/agent_c.py](C:/Users/njb18/Desktop/causal-traitor/agents/agent_c.py) 的要求：

1. 降低“综合打分器”权重
2. 变成 verifier wrapper
3. 最终输出必须以 `verifier.pipeline` 为主
4. 保留现有 debate/jury 接口，以便 appendix 和 demo 重用

### 6.5 ToolExecutor 改造要求

对 [agents/tool_executor.py](C:/Users/njb18/Desktop/causal-traitor/agents/tool_executor.py) 的要求：

1. 禁止读取 gold-only graph/scm
2. 如果 public side 没有 graph，就不能偷偷回退到真图
3. 输出工具 trace，供 verifier 使用
4. 将“support score”从 heuristic 主裁判降级为 evidence component

### 6.6 验收标准

- verifier 能输出三分类
- verifier 能输出 ledger
- verifier 能输出 witness
- verifier 在没有 countermodel 时才允许进入支持性判定

---

## 7. Phase 3：评估体系重构

### 7.1 目标

让评估真正服务论文，而不是继续服务旧的博弈型系统指标。

### 7.2 文件写集

#### 必改

- [evaluation/metrics.py](C:/Users/njb18/Desktop/causal-traitor/evaluation/metrics.py)
- [evaluation/scorer.py](C:/Users/njb18/Desktop/causal-traitor/evaluation/scorer.py)

#### 建议新增

- `evaluation/reporting.py`
- `evaluation/significance.py`
- `tests/test_evaluation.py`

### 7.3 执行动作

#### Step 1：主指标重构

保留主指标：

- verdict accuracy
- macro F1
- invalid claim acceptance rate
- unidentifiable awareness
- ECE
- Brier
- countermodel coverage

#### Step 2：次级指标降级

以下指标迁入 appendix：

- DSR
- difficulty-related metrics
- evolution-related metrics
- jury metrics

#### Step 3：修复 scorer

对 [evaluation/scorer.py](C:/Users/njb18/Desktop/causal-traitor/evaluation/scorer.py)：

- 不再以 game winner 为主对象
- 改为以 verdict correctness 为主对象
- 修掉 `MetricResult` 构造不完整问题

#### Step 4：加入统计显著性

在 `evaluation/significance.py` 中实现：

- bootstrap CI
- paired bootstrap test
- McNemar
- Holm-Bonferroni

### 7.4 验收标准

- 主实验报告默认输出 mean ± std
- 能生成带 CI 的表格
- 能执行 paired significance

---

## 8. Phase 4：主实验实施

### 8.1 目标

完成论文主实验，而不是继续跑旧系统 sanity runs。

### 8.2 文件写集

#### 新增实验目录

- `experiments/exp_main_benchmark/`
- `experiments/exp_adversarial_robustness/`
- `experiments/exp_identifiability_ablation/`
- `experiments/exp_leakage_study/`
- `experiments/exp_ood_generalization/`
- `experiments/exp_cross_model_transfer/`
- `experiments/exp_human_audit/`

#### 旧实验处理

- [experiments/exp2_jury_ablation/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp2_jury_ablation/run.py)
- [experiments/exp3_difficulty/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp3_difficulty/run.py)
- [experiments/exp4_evolution/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp4_evolution/run.py)

这些不删，但标为 appendix。

### 8.3 实施顺序

#### 先做

1. `exp_main_benchmark`
2. `exp_leakage_study`
3. `exp_identifiability_ablation`

#### 再做

4. `exp_adversarial_robustness`
5. `exp_ood_generalization`
6. `exp_cross_model_transfer`

#### 最后做

7. `exp_human_audit`

### 8.4 每个实验必须输出

- config
- seed list
- raw predictions
- aggregated metrics
- CI
- markdown summary
- artifact json

### 8.5 验收标准

- 至少 3 seeds
- 至少一个 OOD split
- 至少一个 leakage 对照
- 至少一个 human audit 子集

---

## 9. Phase 5：Appendix 与 Demo 回接

### 9.1 目标

在主论文系统稳定后，把旧系统资产重新接回 supplemental 层。

### 9.2 文件写集

#### 可回接

- [agents/jury.py](C:/Users/njb18/Desktop/causal-traitor/agents/jury.py)
- [game/difficulty.py](C:/Users/njb18/Desktop/causal-traitor/game/difficulty.py)
- [game/evolution.py](C:/Users/njb18/Desktop/causal-traitor/game/evolution.py)
- [visualization/api.py](C:/Users/njb18/Desktop/causal-traitor/visualization/api.py)
- [visualization/frontend/src/App.tsx](C:/Users/njb18/Desktop/causal-traitor/visualization/frontend/src/App.tsx)
- [run_live_game.py](C:/Users/njb18/Desktop/causal-traitor/run_live_game.py)

### 9.3 回接原则

1. 不允许反向污染主论文 benchmark
2. 不允许重新依赖 gold-only fields
3. 所有 demo 逻辑都必须基于新的 public instance schema

---

## 10. Agent 并行拆分建议

### 10.1 推荐 4 类 agent

#### Agent A：Schema / Benchmark Agent

负责：

- `benchmark/`
- `game/types.py`
- public/gold partition

写集：

- `benchmark/*`
- `game/types.py`
- `tests/test_benchmark.py`
- `tests/test_information_partition.py`

#### Agent B：Verifier Agent

负责：

- `verifier/`
- `agents/agent_c.py`
- `agents/tool_executor.py`

写集：

- `verifier/*`
- `agents/agent_c.py`
- `agents/tool_executor.py`
- `tests/test_verifier.py`

#### Agent C：Evaluation / Experiments Agent

负责：

- `evaluation/*`
- `experiments/exp_main_benchmark/*`
- `experiments/exp_leakage_study/*`
- `experiments/exp_identifiability_ablation/*`

写集：

- `evaluation/*`
- `experiments/exp_main_benchmark/*`
- `experiments/exp_leakage_study/*`
- `experiments/exp_identifiability_ablation/*`

#### Agent D：Appendix / Demo Agent

负责：

- `jury`
- `difficulty`
- `evolution`
- `visualization`

注意：

必须等主线稳定后再进入。

### 10.2 不可并行写集

以下文件尽量不要多人同时改：

- [game/types.py](C:/Users/njb18/Desktop/causal-traitor/game/types.py)
- [agents/agent_c.py](C:/Users/njb18/Desktop/causal-traitor/agents/agent_c.py)
- [agents/tool_executor.py](C:/Users/njb18/Desktop/causal-traitor/agents/tool_executor.py)
- [evaluation/scorer.py](C:/Users/njb18/Desktop/causal-traitor/evaluation/scorer.py)
- [main.py](C:/Users/njb18/Desktop/causal-traitor/main.py)

### 10.3 最佳并行时机

#### 可并行

- benchmark families
- claim parser
- evaluation metrics
- experiment shell scripts

#### 必须串行

- information partition contract
- final verifier decision protocol
- evaluation label contract

---

## 11. 每阶段交付物

### Phase 0 交付物

- schema contract
- information partition rules
- label definitions

### Phase 1 交付物

- benchmark generator v1
- split manifest v1
- witness schema v1

### Phase 2 交付物

- verifier pipeline v1
- countermodel search v1
- agent_c wrapper v1

### Phase 3 交付物

- evaluation pipeline v1
- statistical reporting v1

### Phase 4 交付物

- main tables v1
- ablation tables v1
- leakage study v1

### Phase 5 交付物

- appendix experiments
- visualization reconnect
- live demo update

---

## 12. 交付验收清单

在宣布“主线完成”前，必须全部满足：

- [ ] verifier 不访问 gold-only fields
- [ ] benchmark 支持 `valid / invalid / unidentifiable`
- [ ] 至少一个 OOD split
- [ ] 至少一个 leakage study
- [ ] 至少一个 countermodel witness
- [ ] 至少 3 seeds
- [ ] 有 CI 和显著性
- [ ] appendix 模块不反向污染主线

---

## 13. 对 AI Agent 的投喂建议

### 13.1 每次只给 agent 一个清晰子任务

推荐格式：

1. 背景
2. 目标
3. 前置依赖
4. 只允许修改
5. 禁止修改
6. 必须完成
7. 验收标准
8. 建议测试命令

### 13.2 通用任务卡模板

下面这份模板可以直接复制给 agent：

```text
任务名称：

背景：

前置依赖：

目标：

只允许修改：
- ...

可只读参考：
- ...

禁止修改：
- ...

必须完成：
1. ...
2. ...
3. ...

验收标准：
- ...
- ...

建议运行：
- ...
- ...

完成后汇报：
- 改动文件列表
- 关键设计选择
- 未完成项或风险
```

### 13.3 阶段 gate 规则

所有阶段必须遵守以下 gate：

- `Phase 0` 未合并前，禁止开始 `Phase 1+`
- `Phase 1` 未形成稳定 public/gold schema 前，禁止开始 `Phase 2`
- `Phase 2` 未完成 verifier pipeline 前，禁止开始正式主实验
- `Phase 3` 未完成统计报告层前，禁止输出主结论表
- `Phase 5` 必须最后做，不能抢跑

### 13.4 Phase 0 任务卡

#### 任务卡 P0-S1：信息分区与 schema 契约

- 背景：
  当前项目最大的结构性风险是 oracle leakage。必须先把 gold/public access contract 固定下来。
- 前置依赖：
  无。
- 只允许修改：
  - [game/types.py](C:/Users/njb18/Desktop/causal-traitor/game/types.py)
  - `benchmark/schema.py`
  - `tests/test_information_partition.py`
- 可只读参考：
  - [docs/FINAL_CONSTRUCTION_BLUEPRINT.md](C:/Users/njb18/Desktop/causal-traitor/docs/FINAL_CONSTRUCTION_BLUEPRINT.md)
  - [docs/AGENT_EXECUTION_MANUAL.md](C:/Users/njb18/Desktop/causal-traitor/docs/AGENT_EXECUTION_MANUAL.md)
- 禁止修改：
  - [agents/agent_c.py](C:/Users/njb18/Desktop/causal-traitor/agents/agent_c.py)
  - [agents/tool_executor.py](C:/Users/njb18/Desktop/causal-traitor/agents/tool_executor.py)
  - `benchmark/generator.py`
- 必须完成：
  1. 拆分 `CausalScenario` 为 gold/public 语义明确的结构。
  2. 定义 verifier 可见字段与不可见字段。
  3. 为 public/gold 转换写测试。
- 验收标准：
  - verifier 输入类型不再包含 `true_dag / hidden_variables / true_scm / full_data`
  - 测试能验证 gold-only 字段不会流入 verifier
- 建议运行：
  - `pytest tests/test_information_partition.py -q`
  - `pytest tests/test_integration.py -q`

#### 任务卡 P0-S2：标签空间冻结

- 背景：
  主论文任务必须统一到 `valid / invalid / unidentifiable`。
- 前置依赖：
  P0-S1 完成或至少 schema 已稳定。
- 只允许修改：
  - `benchmark/schema.py`
  - [game/types.py](C:/Users/njb18/Desktop/causal-traitor/game/types.py)
  - `tests/test_information_partition.py`
- 可只读参考：
  - [evaluation/metrics.py](C:/Users/njb18/Desktop/causal-traitor/evaluation/metrics.py)
  - [experiments/exp1_causal_levels/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp1_causal_levels/run.py)
- 禁止修改：
  - [evaluation/scorer.py](C:/Users/njb18/Desktop/causal-traitor/evaluation/scorer.py)
  - [agents/agent_c.py](C:/Users/njb18/Desktop/causal-traitor/agents/agent_c.py)
- 必须完成：
  1. 在 schema 中固化 label enum。
  2. 标注文档或代码注释说明 `winner != label`。
  3. 为后续 evaluator 留出 verdict 字段。
- 验收标准：
  - schema 层已经没有把 game winner 当主标签的歧义

### 13.5 Phase 1 任务卡

#### 任务卡 P1-S1：benchmark schema 落地

- 前置依赖：
  Phase 0 完成。
- 只允许修改：
  - `benchmark/schema.py`
  - `tests/test_benchmark.py`
- 禁止修改：
  - [game/data_generator.py](C:/Users/njb18/Desktop/causal-traitor/game/data_generator.py)
- 必须完成：
  1. 定义 benchmark 样本主 schema。
  2. 定义 witness schema。
  3. 定义 split manifest schema。
- 验收标准：
  - schema 可序列化
  - schema 支持 `valid / invalid / unidentifiable`
  - 测试覆盖 schema 基本构造

#### 任务卡 P1-S2：graph family 模板

- 前置依赖：
  P1-S1 完成。
- 只允许修改：
  - `benchmark/graph_families.py`
  - `tests/test_benchmark.py`
- 禁止修改：
  - `benchmark/schema.py`
  - [game/types.py](C:/Users/njb18/Desktop/causal-traitor/game/types.py)
- 必须完成：
  1. 实现 L1/L2/L3 至少各 2 个 graph family。
  2. family 必须区分 identifiable 与 potentially unidentifiable 情形。
  3. 为 family 提供 deterministic seed 接口。
- 验收标准：
  - 可按 family name 生成结构
  - family 输出可供 generator 复用

#### 任务卡 P1-S3：showcase 场景迁移

- 前置依赖：
  P1-S2 完成。
- 只允许修改：
  - [game/data_generator.py](C:/Users/njb18/Desktop/causal-traitor/game/data_generator.py)
  - `benchmark/generator.py`
  - `tests/test_benchmark.py`
- 禁止修改：
  - [agents/agent_a.py](C:/Users/njb18/Desktop/causal-traitor/agents/agent_a.py)
  - [agents/agent_c.py](C:/Users/njb18/Desktop/causal-traitor/agents/agent_c.py)
- 必须完成：
  1. 保留 smoking / education / drug 三类 showcase 逻辑。
  2. 把它们转成 benchmark family 的子族，而不是唯一测试集。
  3. 提供 public instance 导出。
- 验收标准：
  - showcase 场景还能跑
  - 但 generator 已支持非 showcase 的程序化样本

#### 任务卡 P1-S4：攻击样本生成

- 前置依赖：
  P1-S1 完成。
- 只允许修改：
  - `benchmark/attacks.py`
  - `tests/test_benchmark.py`
- 禁止修改：
  - `benchmark/schema.py`
  - [agents/agent_a.py](C:/Users/njb18/Desktop/causal-traitor/agents/agent_a.py)
- 必须完成：
  1. 实现至少 5 类攻击模板。
  2. 每类攻击都能生成 claim text 与 attacker rationale。
  3. 攻击文本风格可随机化。
- 验收标准：
  - 攻击生成器可复现
  - 能生成与 label 兼容的 claim

#### 任务卡 P1-S5：witness 生成

- 前置依赖：
  P1-S1 完成。
- 只允许修改：
  - `benchmark/witnesses.py`
  - `tests/test_benchmark.py`
- 禁止修改：
  - `benchmark/attacks.py`
  - [agents/tool_executor.py](C:/Users/njb18/Desktop/causal-traitor/agents/tool_executor.py)
- 必须完成：
  1. 生成 support witness。
  2. 生成 countermodel witness。
  3. 生成 assumption witness。
- 验收标准：
  - 每类 witness 都能和 schema 对齐
  - unidentifiable 样本可给出 countermodel 级 witness

#### 任务卡 P1-S6：split builder

- 前置依赖：
  P1-S2、P1-S4、P1-S5 完成。
- 只允许修改：
  - `benchmark/split_builder.py`
  - `benchmark/loaders.py`
  - `tests/test_benchmark.py`
- 禁止修改：
  - [game/data_generator.py](C:/Users/njb18/Desktop/causal-traitor/game/data_generator.py)
- 必须完成：
  1. 生成 `train/dev/test_iid/test_ood`。
  2. 支持 family holdout、lexical holdout、variable renaming holdout。
  3. 输出 split manifest。
- 验收标准：
  - 至少一个 OOD split
  - manifest 可被 loader 正确读取

### 13.6 Phase 2 任务卡

#### 任务卡 P2-S1：claim parser

- 前置依赖：
  Phase 1 完成。
- 只允许修改：
  - `verifier/claim_parser.py`
  - `verifier/outputs.py`
  - `tests/test_verifier.py`
- 禁止修改：
  - [agents/agent_c.py](C:/Users/njb18/Desktop/causal-traitor/agents/agent_c.py)
  - [agents/tool_executor.py](C:/Users/njb18/Desktop/causal-traitor/agents/tool_executor.py)
- 必须完成：
  1. 提取 query type。
  2. 提取 treatment/outcome。
  3. 提取 claim strength 与 assumption cues。
- 验收标准：
  - parser 输出结构化对象
  - 测试覆盖 L1/L2/L3 示例

#### 任务卡 P2-S2：assumption ledger

- 前置依赖：
  P2-S1 完成。
- 只允许修改：
  - `verifier/assumption_ledger.py`
  - `tests/test_verifier.py`
- 禁止修改：
  - `verifier/countermodel_search.py`
- 必须完成：
  1. 将 claim 转为显式 assumption ledger。
  2. 标出 supported / contradicted / unresolved。
  3. 输出 machine-readable ledger。
- 验收标准：
  - ledger 至少覆盖 confounding / IV / selection / counterfactual 假设

#### 任务卡 P2-S3：countermodel search

- 前置依赖：
  P2-S2 完成。
- 只允许修改：
  - `verifier/countermodel_search.py`
  - `tests/test_verifier.py`
- 可只读参考：
  - [causal_tools/l2_intervention.py](C:/Users/njb18/Desktop/causal-traitor/causal_tools/l2_intervention.py)
  - [causal_tools/l3_counterfactual.py](C:/Users/njb18/Desktop/causal-traitor/causal_tools/l3_counterfactual.py)
- 禁止修改：
  - [agents/tool_executor.py](C:/Users/njb18/Desktop/causal-traitor/agents/tool_executor.py)
- 必须完成：
  1. 实现 L1 countermodel 候选。
  2. 实现 L2 countermodel 候选。
  3. 实现 L3 至少一个 observationally equivalent but counterfactually different 候选。
- 验收标准：
  - 能输出 `found_countermodel`
  - 能区分 invalid 与 unidentifiable 的建议

#### 任务卡 P2-S4：decision rule

- 前置依赖：
  P2-S3 完成。
- 只允许修改：
  - `verifier/decision.py`
  - `verifier/pipeline.py`
  - `tests/test_verifier.py`
- 禁止修改：
  - [evaluation/scorer.py](C:/Users/njb18/Desktop/causal-traitor/evaluation/scorer.py)
- 必须完成：
  1. 固化四段式决策顺序。
  2. 支持 `valid / invalid / unidentifiable`。
  3. 输出 confidence、ledger、witness、tool trace。
- 验收标准：
  - pipeline 可一键跑通
  - 无 countermodel 时才进入支持性判定

#### 任务卡 P2-S5：Agent C verifier 包装

- 前置依赖：
  P2-S4 完成。
- 只允许修改：
  - [agents/agent_c.py](C:/Users/njb18/Desktop/causal-traitor/agents/agent_c.py)
  - `verifier/pipeline.py`
  - `tests/test_agents.py`
- 禁止修改：
  - [agents/jury.py](C:/Users/njb18/Desktop/causal-traitor/agents/jury.py)
  - [game/debate_engine.py](C:/Users/njb18/Desktop/causal-traitor/game/debate_engine.py)
- 必须完成：
  1. 把 `Agent C` 改成 verifier-first wrapper。
  2. 保留旧接口兼容性。
  3. 输出新 verdict 结构。
- 验收标准：
  - Agent C 默认主路径调用 verifier pipeline
  - 旧 debate 路径不直接破坏现有 demo

#### 任务卡 P2-S6：ToolExecutor 去 oracle 化

- 前置依赖：
  P0-S1 已完成。
- 只允许修改：
  - [agents/tool_executor.py](C:/Users/njb18/Desktop/causal-traitor/agents/tool_executor.py)
  - [causal_tools/meta_tools.py](C:/Users/njb18/Desktop/causal-traitor/causal_tools/meta_tools.py)
  - `tests/test_tool_executor.py`
- 禁止修改：
  - `benchmark/schema.py`
- 必须完成：
  1. 禁止 public side 回退到 true graph / true SCM。
  2. 输出标准化 tool trace。
  3. 将 support score 降级为 evidence component。
- 验收标准：
  - ToolExecutor 在 public instance 上可运行
  - 无法偷偷读取 gold-only 信息

### 13.7 Phase 3 任务卡

#### 任务卡 P3-S1：主指标重构与 scorer 修复

- 前置依赖：
  Phase 2 完成。
- 只允许修改：
  - [evaluation/metrics.py](C:/Users/njb18/Desktop/causal-traitor/evaluation/metrics.py)
  - [evaluation/scorer.py](C:/Users/njb18/Desktop/causal-traitor/evaluation/scorer.py)
  - `tests/test_evaluation.py`
- 禁止修改：
  - [agents/agent_c.py](C:/Users/njb18/Desktop/causal-traitor/agents/agent_c.py)
- 必须完成：
  1. 主指标转为 verdict-centric。
  2. 修复 `MetricResult` 构造问题。
  3. 将 DSR / jury / evolution 指标降级为 appendix 指标。
- 验收标准：
  - scorer 不再依赖 game winner
  - evaluation 测试通过

#### 任务卡 P3-S2：统计报告层

- 前置依赖：
  P3-S1 完成。
- 只允许修改：
  - `evaluation/reporting.py`
  - `evaluation/significance.py`
  - `tests/test_evaluation.py`
- 禁止修改：
  - [experiments/exp_main_benchmark/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp_main_benchmark/run.py)
- 必须完成：
  1. bootstrap CI
  2. paired bootstrap or McNemar
  3. Holm-Bonferroni
- 验收标准：
  - 能输出 mean ± std 与 CI
  - 能对两组预测做显著性检验

### 13.8 Phase 4 任务卡

#### 任务卡 P4-S1：Main benchmark 实验框架

- 前置依赖：
  Phase 3 完成。
- 只允许修改：
  - `experiments/exp_main_benchmark/run.py`
  - `tests/test_integration.py`
- 禁止修改：
  - [agents/jury.py](C:/Users/njb18/Desktop/causal-traitor/agents/jury.py)
  - [game/evolution.py](C:/Users/njb18/Desktop/causal-traitor/game/evolution.py)
- 必须完成：
  1. 多 seed 运行。
  2. 输出 raw predictions、aggregated metrics、CI。
  3. 支持 IID/OOD split。
- 验收标准：
  - 至少 3 seeds
  - 至少一个 OOD split

#### 任务卡 P4-S2：Leakage study

- 前置依赖：
  P4-S1 完成。
- 只允许修改：
  - `experiments/exp_leakage_study/run.py`
  - `tests/test_integration.py`
- 禁止修改：
  - [agents/tool_executor.py](C:/Users/njb18/Desktop/causal-traitor/agents/tool_executor.py)
- 必须完成：
  1. 比较 clean partition 与 oracle-leaking partition。
  2. 输出性能虚高对照。
  3. 形成 markdown summary。
- 验收标准：
  - 至少一组 clean vs leaking 结果
  - 可支撑论文中的 leakage 警示结论

#### 任务卡 P4-S3：Identifiability ablation

- 前置依赖：
  P4-S1 完成。
- 只允许修改：
  - `experiments/exp_identifiability_ablation/run.py`
  - `tests/test_integration.py`
- 禁止修改：
  - `benchmark/schema.py`
- 必须完成：
  1. no ledger
  2. no countermodel
  3. no abstention
  4. no tools
- 验收标准：
  - 每个 ablation 都有独立结果
  - 能比较 invalid acceptance 与 unidentifiable awareness

#### 任务卡 P4-S4：Robustness / OOD / Transfer

- 前置依赖：
  P4-S1 完成。
- 只允许修改：
  - `experiments/exp_adversarial_robustness/run.py`
  - `experiments/exp_ood_generalization/run.py`
  - `experiments/exp_cross_model_transfer/run.py`
- 禁止修改：
  - [game/difficulty.py](C:/Users/njb18/Desktop/causal-traitor/game/difficulty.py)
  - [game/evolution.py](C:/Users/njb18/Desktop/causal-traitor/game/evolution.py)
- 必须完成：
  1. 按攻击强度分层。
  2. 跑 OOD split。
  3. 跑跨模型家族组合。
- 验收标准：
  - robustness / OOD / transfer 各自至少一张主表

#### 任务卡 P4-S5：Human audit

- 前置依赖：
  P4-S1 完成。
- 只允许修改：
  - `experiments/exp_human_audit/run.py`
  - `evaluation/reporting.py`
- 禁止修改：
  - `verifier/*`
- 必须完成：
  1. 抽样生成标注包。
  2. 设计标注字段。
  3. 输出双人标注一致性统计接口。
- 验收标准：
  - human audit 数据格式明确
  - 可输出基本一致性统计

### 13.9 Phase 5 任务卡

#### 任务卡 P5-S1：Appendix 模块回接

- 前置依赖：
  Phase 4 主实验完成。
- 只允许修改：
  - [agents/jury.py](C:/Users/njb18/Desktop/causal-traitor/agents/jury.py)
  - [game/difficulty.py](C:/Users/njb18/Desktop/causal-traitor/game/difficulty.py)
  - [game/evolution.py](C:/Users/njb18/Desktop/causal-traitor/game/evolution.py)
  - `experiments/exp2_jury_ablation/run.py`
  - `experiments/exp3_difficulty/run.py`
  - `experiments/exp4_evolution/run.py`
- 禁止修改：
  - `benchmark/*`
  - `verifier/*`
- 必须完成：
  1. 让 appendix 模块适配新 public instance schema。
  2. 确保不再依赖 gold-only fields。
  3. 跑通 appendix 实验。
- 验收标准：
  - appendix 模块不污染主线

#### 任务卡 P5-S2：Visualization / Demo 回接

- 前置依赖：
  P5-S1 完成或至少主 schema 已稳定。
- 只允许修改：
  - [visualization/api.py](C:/Users/njb18/Desktop/causal-traitor/visualization/api.py)
  - [visualization/frontend/src/App.tsx](C:/Users/njb18/Desktop/causal-traitor/visualization/frontend/src/App.tsx)
  - [run_live_game.py](C:/Users/njb18/Desktop/causal-traitor/run_live_game.py)
- 禁止修改：
  - `benchmark/*`
  - `verifier/*`
- 必须完成：
  1. 让 live demo 基于新 schema。
  2. 保留现有展示价值。
  3. 不引入主线额外依赖。
- 验收标准：
  - demo 能跑
  - 但不会反向修改主论文数据流

### 13.10 推荐先后派发顺序

最推荐的投喂顺序是：

1. `P0-S1`
2. `P0-S2`
3. `P1-S1`
4. `P1-S2`
5. `P1-S4`
6. `P1-S5`
7. `P1-S3`
8. `P1-S6`
9. `P2-S1`
10. `P2-S2`
11. `P2-S3`
12. `P2-S4`
13. `P2-S6`
14. `P2-S5`
15. `P3-S1`
16. `P3-S2`
17. `P4-S1`
18. `P4-S2`
19. `P4-S3`
20. `P4-S4`
21. `P4-S5`
22. `P5-S1`
23. `P5-S2`

### 13.11 可并行组合

在不冲突的情况下，推荐这些组合并行：

- `P1-S2` + `P1-S4` + `P1-S5`
- `P2-S1` 完成后，可并行启动 `P2-S2` 的部分规则草稿与 `P3-S1` 的评估设计草稿，但正式提交仍需等 `Phase 2` 稳定
- `P4-S2` + `P4-S3` 可并行
- `P4-S4` 中 robustness / OOD / transfer 可并行拆成子 agent

### 13.12 不要让 agent 自由发挥的地方

以下部分必须严格按手册，不要让 agent 随意改动方向：

- 标签空间定义
- information partition contract
- verifier 决策顺序
- 主实验指标集合
- 主论文 vs appendix 的边界

---

## 14. 最终说明

本手册的作用不是代替蓝图，而是把蓝图转成：

- 可派发
- 可并行
- 可验收
- 可避免冲突

的工程执行说明。

如果 AI agent 系统严格按本手册施工，最重要的收益是：

1. 不会被 UI / jury / evolution 分散主线
2. 不会在 oracle leakage 未修复前提前跑实验
3. 不会多人同时改最危险的核心文件
4. 能把当前仓库最大化复用到新的论文主线中

这份文档就是你后续用于驱动 agent 的执行说明书。
