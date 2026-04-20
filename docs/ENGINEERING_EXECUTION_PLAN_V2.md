# The Causal Traitor v4
## 工程施工与执行方案 v2

> 版本：v2 Draft
> 日期：2026-04-20
> 目标：将 `FINAL_CONSTRUCTION_BLUEPRINT_V2.md` 落地为可执行、低冲突、论文优先的工程施工路线
> 原则：论文主线优先、contract 先于实现、benchmark credibility 先于系统复杂度

---

## 0. 总原则

所有工程施工必须遵守：

1. 先论文主线，后 demo
2. 先 contract，后实现
3. 先切断 leakage，后跑实验
4. 先 benchmark credibility，后系统扩展
5. 所有新增模块都必须回答“它服务哪条论文主论点”

成功标准不是“系统更复杂”，而是：

- 主问题是否更锋利
- benchmark 是否更可信
- 方法是否更必要
- 结果是否更难被反驳
- 文档与实验是否可复现

---

## 1. 仓库中的主线与非主线划分

### 主线目录

- `benchmark/`
- `verifier/`
- `evaluation/`
- `experiments/`
- `tests/`
- `docs/`

### 非主线目录

- `agents/` 中旧 debate/jury 外壳
- `game/` 中 difficulty/evolution 模块
- `visualization/`
- `main.py`
- `run_live_game.py`

### 执行规则

主线稳定前，非主线只允许：

- 兼容性修复
- schema 对齐
- appendix 标记

不允许：

- 扩功能
- 美化 UI
- 增加博弈复杂度

---

## 2. 统一工作流分解

建议拆成 6 条工作流。

### Workstream A：Research Framing

负责：

- 蓝图与执行文档
- canonical source-of-truth 切换
- superseded 文档降级声明
- 任务定义与指标定义
- annotation protocol
- 论文叙事统一

主要文件：

- `docs/FINAL_CONSTRUCTION_BLUEPRINT_V2.md`
- `docs/ENGINEERING_EXECUTION_PLAN_V2.md`
- `README.md`
- `docs/PROJECT_MAP.md`
- `docs/COURSE_REPO_WALKTHROUGH.md`
- `docs/ARTIFACT_LAYOUT.md`
- `docs/FINAL_CONSTRUCTION_BLUEPRINT.md`
- `docs/AGENT_EXECUTION_MANUAL.md`
- 新增 `docs/ANNOTATION_PROTOCOL.md`
- 新增 `docs/PAPER_WRITING_OUTLINE_V2.md`

### Workstream B：Benchmark Core v2

负责：

- benchmark schema
- graph families
- persuasion overlays
- split builder
- real-grounded subset schema / loader

主要文件：

- `benchmark/schema.py`
- `benchmark/graph_families.py`
- `benchmark/attacks.py`
- `benchmark/generator.py`
- `benchmark/witnesses.py`
- `benchmark/split_builder.py`
- `benchmark/loaders.py`
- 新增 `benchmark/persuasion_overlays.py`
- 新增 `benchmark/real_grounded.py`

### Workstream C：Verifier Core v2

负责：

- parser
- assumption ledger
- countermodel search
- selective decision
- refusal reason / missing information outputs

主要文件：

- `verifier/outputs.py`
- `verifier/claim_parser.py`
- `verifier/assumption_ledger.py`
- `verifier/countermodel_search.py`
- `verifier/decision.py`
- `verifier/pipeline.py`
- `agents/tool_executor.py`

### Workstream D：Evaluation & Scoring v2

负责：

- wise refusal metrics
- unsafe acceptance metrics
- identification-stage metrics
- human audit aggregation
- reporting / significance

主要文件：

- `evaluation/metrics.py`
- `evaluation/scorer.py`
- `evaluation/reporting.py`
- `evaluation/significance.py`

### Workstream E：Experiments v2

负责：

- benchmark harness
- stronger baseline matrix
- main benchmark
- leakage study
- ablation
- OOD suite
- persuasion robustness
- real-grounded subset evaluation
- witness faithfulness

主要文件：

- `experiments/benchmark_harness.py`
- `experiments/exp_main_benchmark/run.py`
- `experiments/exp_leakage_study/run.py`
- `experiments/exp_identifiability_ablation/run.py`
- `experiments/exp_ood_generalization/run.py`
- `experiments/exp_human_audit/run.py`
- 新增 `experiments/exp_persuasion_robustness/run.py`
- 新增 `experiments/exp_real_grounded_subset/run.py`
- 新增 `experiments/exp_witness_faithfulness/run.py`

### Workstream F：Human Audit & Paper Assets

负责：

- real-grounded case collection
- dual annotation
- conflict arbitration
- final tables / figures / case studies

主要文件：

- `docs/ANNOTATION_PROTOCOL.md`
- `outputs/review/`
- `outputs/mainline/`
- 图表导出与 paper-facing markdown summaries

---

## 3. 不可并行写集

以下文件尽量只允许一个负责人同时修改：

- `benchmark/schema.py`
- `benchmark/generator.py`
- `verifier/decision.py`
- `verifier/pipeline.py`
- `evaluation/scorer.py`
- `experiments/benchmark_harness.py`
- `docs/FINAL_CONSTRUCTION_BLUEPRINT_V2.md`

原因：

- 它们定义了全局 contract
- 冲突会污染所有实验
- 回滚成本高

---

## 4. 推荐阶段划分

### Phase 0：冻结主线与 contract

目标：

- 冻结论文身份
- 冻结任务定义
- 冻结指标 contract
- 冻结非目标列表
- 完成仓库级 canonical source-of-truth 切换
- 让 v3 文档进入 superseded 状态，而不是继续与 v2 并行生效

必须完成：

- `docs/FINAL_CONSTRUCTION_BLUEPRINT_V2.md`
- `docs/ENGINEERING_EXECUTION_PLAN_V2.md`
- `docs/PHASE_TASK_CARDS_V2.md`
- `README.md` 已切到 v2 canonical docs
- `docs/PROJECT_MAP.md` 已切到 v2 active project definition
- `docs/COURSE_REPO_WALKTHROUGH.md` 已切到 v2 主线阅读顺序
- `docs/FINAL_CONSTRUCTION_BLUEPRINT.md` 与 `docs/AGENT_EXECUTION_MANUAL.md` 已标记 superseded

验收标准：

- 团队对主问题与贡献写法一致
- 仓库入口处不再把 v3 文档声明为 canonical baseline
- v3 文档只作为历史参考存在，不再作为 active execution baseline
- 不再扩展旧系统主线
- 所有后续实现以 v2 文档为准

### Phase 1：Schema 与评估 contract 升级

目标：

- 升级 benchmark schema
- 升级 verifier 输出 schema
- 升级 evaluation contract

主要修改：

- `benchmark/schema.py`
- `verifier/outputs.py`
- `evaluation/metrics.py`
- `evaluation/scorer.py`

新增关键字段：

- `identification_status`
- `missing_information_spec`
- `refusal_reason`
- wise refusal 系列 metrics

测试：

- `tests/test_information_partition.py`
- `tests/test_verifier.py`
- `tests/test_evaluation.py`

验收标准：

- verifier 输出结构固定
- evaluation 能正确读取新输出
- 不破坏 public/gold partition

### Phase 2：Benchmark v2 升级

目标：

- 新增 persuasion overlay
- 新增更难的 OOD
- 新增 real-grounded subset 数据通路

主要修改：

- `benchmark/graph_families.py`
- `benchmark/attacks.py`
- `benchmark/generator.py`
- `benchmark/split_builder.py`
- `benchmark/witnesses.py`

新增文件：

- `benchmark/persuasion_overlays.py`
- `benchmark/real_grounded.py`

测试：

- `tests/test_benchmark.py`
- 新增 `tests/test_real_grounded_subset.py`
- 新增 `tests/test_persuasion_overlays.py`

验收标准：

- OOD 不再只依赖 lexical/rename
- 至少有一种 mechanism OOD
- synthetic 与 real-grounded contract 一致

### Phase 3：Verifier v2 升级

目标：

- refusal-aware selective decision
- richer countermodel output
- missing-information explanation

主要修改：

- `verifier/claim_parser.py`
- `verifier/assumption_ledger.py`
- `verifier/countermodel_search.py`
- `verifier/decision.py`
- `verifier/pipeline.py`
- `agents/tool_executor.py`

测试：

- `tests/test_verifier.py`

验收标准：

- verifier 能输出双层状态
- verifier 能输出 refusal reason
- countermodel witness 成为正式结构化证据

### Phase 4：Evaluation 与实验 harness 升级

目标：

- 接入新的 selective metrics
- 完成 stronger baseline matrix
- 完成新实验 runner

主要修改：

- `evaluation/*`
- `experiments/benchmark_harness.py`

新增实验：

- `experiments/exp_persuasion_robustness/run.py`
- `experiments/exp_real_grounded_subset/run.py`
- `experiments/exp_witness_faithfulness/run.py`

验收标准：

- 所有 runner 统一输出：
  - config
  - seed list
  - raw predictions
  - aggregated metrics
  - CI
  - significance
  - markdown summary
- persuasion robustness 需同时区分：
  - taxonomy-complete pressure axis
  - paper-facing primary report axis

### Phase 5：主实验跑通与 benchmark 反向硬化

目标：

- 跑完主实验
- 根据结果反向提高 benchmark 难度
- 消除饱和问题

必须跑：

- main benchmark
- leakage study
- ablation
- persuasion robustness
- OOD suite

关键 gate：

- 若主系统仍接近完美表现，则返回 Phase 2 提高 benchmark 难度
- 若 OOD gap 过低，则返回 Phase 2 或 4 增强 OOD
- 若 baseline 太弱，则返回 Phase 4 补 baseline

验收标准：

- 结果不再过饱和
- OOD 与 persuasion 实验有解释力
- leakage study 显著成立

### Phase 6：Real-grounded subset 与 human audit

目标：

- 完成 real-grounded 数据子集
- 完成 dual annotation
- 完成 agreement 和 conflict analysis

新增产出：

- annotation package
- adjudication report
- gold label quality report
- witness faithfulness report

验收标准：

- 有真实双人标注
- 有 agreement
- 有 conflict arbitration
- 有正文可写的案例分析

### Phase 7：论文写作与 artifact 冻结

目标：

- 生成主表、附表、图
- 固定最终结论边界
- 完成 reproducibility package

验收标准：

- 每项主论点都有对应实验支撑
- 没有关键结果只放 appendix
- 所有 claim 有对应图表或 proposition 支撑

### Phase 8：Supplemental / Demo

目标：

- 只处理 appendix/demo
- 不再反向影响主线

允许处理：

- jury appendix
- evolution appendix
- visualization reconnect

---

## 5. 推荐团队分布方案

### 四人团队最优配置

**负责人 A：Benchmark / Data Lead**

- 负责 Workstream B
- 负责 synthetic core 与 real-grounded subset 数据 contract

**负责人 B：Verifier / Method Lead**

- 负责 Workstream C
- 负责 parser、ledger、countermodel、decision、tool integration

**负责人 C：Evaluation / Experiment Lead**

- 负责 Workstream D 和 E
- 负责 metrics、scorer、benchmark harness、baseline 接入与主实验脚本

**负责人 D：Audit / Writing Lead**

- 负责 Workstream A 和 F
- 负责蓝图、annotation protocol、human audit、paper tables、case studies 与术语统一

### 三人团队压缩配置

**负责人 A：Benchmark + Real-Grounded**

- benchmark core
- persuasion overlays
- real-grounded subset

**负责人 B：Verifier + Tools**

- parser
- ledger
- countermodel
- decision
- tool executor

**负责人 C：Evaluation + Experiments + Writing**

- metrics
- harness
- experiment runners
- human audit aggregation
- docs / paper

### 两人团队最低可行配置

如果只有两人，不建议追求完整 v2。只保：

- main benchmark
- leakage study
- one strong ablation
- one strong OOD suite
- small real-grounded subset
- minimal human audit

必须砍掉：

- cross-model transfer
- supplemental demo
- jury/evolution appendix
- 过多 baseline 变体

---

## 6. 不同类型负责人的协作规则

### Benchmark Lead 不得自行决定

- 新增总体标签空间
- 修改 verifier 输出 contract
- 修改评估主指标定义

### Verifier Lead 不得自行决定

- 修改 benchmark gold label 生成逻辑
- 修改 public/gold 信息分区 contract
- 修改 main metric 定义

### Evaluation Lead 不得自行决定

- 修改 task definition
- 修改 verdict label semantics
- 把 appendix 指标提升为主表指标

### Writing Lead 不得自行决定

- 在没有实验支撑时修改核心结论
- 夸大理论或新颖性 claim

---

## 7. 阶段交付物清单

### Phase 0

- `FINAL_CONSTRUCTION_BLUEPRINT_V2`
- `ENGINEERING_EXECUTION_PLAN_V2`
- `PHASE_TASK_CARDS_V2`
- `README / PROJECT_MAP / COURSE_REPO_WALKTHROUGH / ARTIFACT_LAYOUT` 已切换到 v2
- `FINAL_CONSTRUCTION_BLUEPRINT.md / AGENT_EXECUTION_MANUAL.md` 已标记 superseded

### Phase 1

- updated schema contract
- updated metrics contract
- updated tests

### Phase 2

- benchmark generator v2
- persuasion overlay
- harder OOD split builder
- real-grounded subset schema/loader

### Phase 3

- verifier pipeline v2
- selective decision output
- richer countermodel witness

### Phase 4

- upgraded evaluation stack
- new experiment runners
- stronger baseline matrix

### Phase 5

- main tables v1
- leakage study v1
- robustness/OOD tables v1

### Phase 6

- real-grounded evaluation package
- human audit results
- agreement report

### Phase 7

- paper-ready artifact set
- reproducibility package
- frozen figures/tables

---

## 8. 统一测试与验证要求

### 主线测试

- `tests/test_information_partition.py`
- `tests/test_benchmark.py`
- `tests/test_verifier.py`
- `tests/test_evaluation.py`

### 新增测试建议

- `tests/test_real_grounded_subset.py`
- `tests/test_persuasion_overlays.py`
- `tests/test_wise_refusal_metrics.py`
- `tests/test_witness_faithfulness.py`

### 每个 phase 结束必须满足

- 对应测试通过
- 没有 gold leakage
- schema 没有 break
- artifacts 可复现
- 结果可写入 markdown summary

---

## 9. 内部资源分层

### 必须做

- wise refusal 正式化
- harder OOD
- persuasion overlays
- real-grounded subset
- stronger baselines
- leakage study
- component ablation
- human audit

### 强烈建议做

- propositions
- witness faithfulness experiment
- identification vs verdict disentanglement

### 可选做

- cross-model transfer
- demo / visualization
- jury appendix
- evolution appendix

---

## 10. 最终施工判词

今后的工程施工不再以“系统更复杂”为成功标准，而以以下问题为成功标准：

- 主问题是否更锋利
- benchmark 是否更可信
- 方法是否更必要
- 结果是否更难被反驳
- 文档与实验是否更可复现

如果一个模块不能直接提高这五项中的至少一项，它就不应优先施工。
