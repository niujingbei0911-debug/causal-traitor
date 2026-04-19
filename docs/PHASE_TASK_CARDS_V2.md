# The Causal Traitor v4
## 阶段任务卡与执行拆分 v2

> 版本：v2 Draft
> 日期：2026-04-20
> 用途：把 `FINAL_CONSTRUCTION_BLUEPRINT_V2.md` 与 `ENGINEERING_EXECUTION_PLAN_V2.md` 落成可直接派发的阶段任务卡
> 原则：每张任务卡只服务一个清晰子目标，避免写集冲突，避免 agent 自由发挥

---

## 使用说明

每张任务卡必须包含：

- 背景
- 目标
- 前置依赖
- 允许修改
- 禁止修改
- 必须交付
- 验收标准
- 推荐测试

所有任务卡默认遵守：

- 不允许绕过 public/gold partition
- 不允许随意扩大 scope
- 不允许把 appendix/demo 模块重新拉回主线

---

## Phase 0：主线冻结

### 任务卡 P0-S1：v2 蓝图冻结

**背景**

当前仓库已有主线文档，但需要统一切换到 v2 叙事：selective causal oversight、wise refusal、real-grounded subset。

**目标**

冻结新的论文主线、任务定义、指标 contract 与非目标列表。

**前置依赖**

- 无

**允许修改**

- `docs/FINAL_CONSTRUCTION_BLUEPRINT_V2.md`
- `docs/ENGINEERING_EXECUTION_PLAN_V2.md`
- `docs/PHASE_TASK_CARDS_V2.md`

**禁止修改**

- `benchmark/*`
- `verifier/*`
- `evaluation/*`
- `experiments/*`

**必须交付**

- v2 蓝图
- v2 工程计划
- v2 阶段任务卡

**验收标准**

- 任务名称、核心问题、贡献写法一致
- `wise refusal` 成为正式主轴
- `jury/difficulty/evolution/visualization` 被明确降级

**推荐测试**

- 团队内部 walkthrough
- 核对文档间是否有互相矛盾表述

### 任务卡 P0-S2：仓库 canonical source-of-truth 切换

**背景**

仅新增 v2 文档还不足以完成主线冻结。如果仓库入口、项目地图、课程导览仍把 v3 文档当成 canonical baseline，后续执行会继续被带回旧主线。

**目标**

把仓库入口层全部切换到 v2 source-of-truth。

**前置依赖**

- P0-S1

**允许修改**

- `README.md`
- `docs/PROJECT_MAP.md`
- `docs/COURSE_REPO_WALKTHROUGH.md`
- `docs/ARTIFACT_LAYOUT.md`

**禁止修改**

- `benchmark/*`
- `verifier/*`
- `evaluation/*`
- `experiments/*`

**必须交付**

- 仓库根入口切换到 v2
- 项目地图切换到 v2
- 课程导览切换到 v2
- artifact 布局文档的主线引用切换到 v2

**验收标准**

- `README.md` 的 canonical docs 仅指向 v2 文档
- `docs/PROJECT_MAP.md` 的 active project definition 仅指向 v2 文档
- `docs/COURSE_REPO_WALKTHROUGH.md` 的推荐阅读顺序已切到 v2
- `docs/ARTIFACT_LAYOUT.md` 不再把 v3 蓝图作为当前主线依据

**推荐测试**

- `rg "Current canonical|Canonical Baseline|Read these first|FINAL_CONSTRUCTION_BLUEPRINT.md|AGENT_EXECUTION_MANUAL.md" README.md docs/PROJECT_MAP.md docs/COURSE_REPO_WALKTHROUGH.md docs/ARTIFACT_LAYOUT.md`

### 任务卡 P0-S3：旧文档 superseded 标记与执行禁用声明

**背景**

如果 v3 文档不被显式标记为 superseded，它们就会继续与 v2 并行生效，造成双重执行协议。

**目标**

把旧蓝图与旧执行手册安全降级为历史参考文档，并明确禁止它们继续作为新任务的执行依据。

**前置依赖**

- P0-S2

**允许修改**

- `docs/FINAL_CONSTRUCTION_BLUEPRINT.md`
- `docs/AGENT_EXECUTION_MANUAL.md`
- `docs/legacy/README.md`

**禁止修改**

- `benchmark/*`
- `verifier/*`
- `evaluation/*`
- `experiments/*`

**必须交付**

- v3 蓝图 superseded banner
- v3 执行手册 superseded banner
- legacy README 中的新旧关系说明

**验收标准**

- v3 文档文件头部有明确 superseded 声明
- superseded 声明中指向 v2 文档集合
- 对所有新工作，v3 文档不再被当作 active execution baseline

**推荐测试**

- `rg "Superseded for new work|active source of truth|active execution baseline" docs/FINAL_CONSTRUCTION_BLUEPRINT.md docs/AGENT_EXECUTION_MANUAL.md docs/legacy/README.md`

---

## Phase 1：Schema 与评估 contract

### 任务卡 P1-S1：Schema 升级为 selective outputs

**背景**

当前 schema 已支持 `valid/invalid/unidentifiable`，但缺少 `identification_status`、`refusal_reason` 等字段。

**目标**

升级 benchmark/verifier 输出 contract，使 selective decision 正式进入主线。

**前置依赖**

- P0-S3

**允许修改**

- `benchmark/schema.py`
- `verifier/outputs.py`
- `tests/test_information_partition.py`
- `tests/test_verifier.py`

**禁止修改**

- `experiments/*`
- `visualization/*`

**必须交付**

- 新的结构化输出字段
- 对应测试

**验收标准**

- verifier 输出包含 `identification_status`
- verifier 输出包含 `missing_information_spec` 或等价字段
- public/gold partition 未被破坏

**推荐测试**

- `pytest tests/test_information_partition.py tests/test_verifier.py -q`

### 任务卡 P1-S2：主指标 contract 升级

**背景**

当前评估以 accuracy/F1 为主，尚未把 wise refusal 形式化为主指标。

**目标**

新增：

- `wise_refusal_recall`
- `wise_refusal_precision`
- `over_commitment_rate`
- `over_refusal_rate`

**前置依赖**

- P1-S1
- P0-S3

**允许修改**

- `evaluation/metrics.py`
- `evaluation/scorer.py`
- `tests/test_evaluation.py`

**禁止修改**

- `benchmark/*`
- `experiments/*`

**必须交付**

- 新指标定义
- scorer 接入
- 测试

**验收标准**

- scorer 能输出新指标
- 老指标不被破坏

**推荐测试**

- `pytest tests/test_evaluation.py -q`

---

## Phase 2：Benchmark v2

### 任务卡 P2-S1：Persuasion overlay

**背景**

当前攻击主要是结构性 causal attack，缺少 persuasion layer。

**目标**

在现有 attack taxonomy 上叠加：

- authority pressure
- expert tone pressure
- confidence pressure
- consensus pressure
- concealment pressure

**前置依赖**

- P1-S1
- P0-S3

**允许修改**

- `benchmark/attacks.py`
- 新增 `benchmark/persuasion_overlays.py`
- `benchmark/generator.py`
- `tests/test_benchmark.py`
- 新增 `tests/test_persuasion_overlays.py`

**禁止修改**

- `evaluation/*`
- `visualization/*`

**必须交付**

- persuasion style 生成逻辑
- 样本元信息记录

**验收标准**

- 同一结构性 attack 可生成不同 persuasion 风格
- 风格信息可被实验脚本读取

**推荐测试**

- `pytest tests/test_benchmark.py tests/test_persuasion_overlays.py -q`

### 任务卡 P2-S2：Harder OOD 与 paired-flip

**背景**

当前 OOD 仍偏表面化，实验区分度不够。

**目标**

新增：

- mechanism OOD
- attack-family OOD
- paired-flip OOD
- context-shift OOD

**前置依赖**

- P2-S1

**允许修改**

- `benchmark/graph_families.py`
- `benchmark/generator.py`
- `benchmark/split_builder.py`
- `tests/test_benchmark.py`

**禁止修改**

- `verifier/*`
- `evaluation/*`

**必须交付**

- 新 OOD 标注逻辑
- split manifest 扩展

**验收标准**

- OOD bucket 不再只依赖 lexical/rename
- paired-flip 样本可生成

**推荐测试**

- `pytest tests/test_benchmark.py -q`

### 任务卡 P2-S3：Real-grounded subset contract

**背景**

当前 benchmark 以 synthetic 为主，缺少外部 grounding。

**目标**

建立 literature-grounded / semi-real 子集的数据 contract、加载逻辑和导出格式。

**前置依赖**

- P1-S1

**允许修改**

- 新增 `benchmark/real_grounded.py`
- `benchmark/loaders.py`
- `benchmark/schema.py`
- 新增 `tests/test_real_grounded_subset.py`

**禁止修改**

- `verifier/*`
- `evaluation/*`

**必须交付**

- real-grounded case schema
- loader / serializer

**验收标准**

- synthetic 与 real-grounded 可共用核心 claim schema
- source citation 字段存在

**推荐测试**

- `pytest tests/test_real_grounded_subset.py -q`

---

## Phase 3：Verifier v2

### 任务卡 P3-S1：Refusal-aware decision rule

**背景**

当前 verifier 有 `unidentifiable`，但尚未显式输出 refusal reason / missing info。

**目标**

把最终 decision 升级成 refusal-aware selective decision。

**前置依赖**

- P1-S1
- P1-S2

**允许修改**

- `verifier/decision.py`
- `verifier/pipeline.py`
- `tests/test_verifier.py`

**禁止修改**

- `benchmark/graph_families.py`

**必须交付**

- refusal reason
- missing information spec
- identification status 接入最终 decision

**验收标准**

- `unidentifiable` 不再只是标签
- 输出中明确说明为何拒绝承诺

**推荐测试**

- `pytest tests/test_verifier.py -q`

### 任务卡 P3-S2：Countermodel witness 丰富化

**背景**

countermodel 目前已结构化，但还需要更像“正式证据对象”。

**目标**

增强 countermodel 输出字段：

- type
- match score
- query disagreement
- triggered assumptions
- explanation

**前置依赖**

- P3-S1

**允许修改**

- `verifier/countermodel_search.py`
- `verifier/decision.py`
- `tests/test_verifier.py`

**禁止修改**

- `evaluation/*`

**必须交付**

- richer countermodel witness

**验收标准**

- countermodel witness 可直接进入评估和 human audit

**推荐测试**

- `pytest tests/test_verifier.py -q`

---

## Phase 4：Evaluation 与实验框架

### 任务卡 P4-S1：Main benchmark baseline matrix 升级

**背景**

当前 baseline 不足以支撑主会级比较。

**目标**

补齐至少以下 baseline：

- direct judge
- CoT judge
- self-consistency judge
- tool baseline
- debate baseline
- refusal-aware baseline

**前置依赖**

- P1-S2
- P3-S1

**允许修改**

- `experiments/benchmark_harness.py`
- `experiments/exp_main_benchmark/run.py`
- `tests/test_benchmark_harness.py`

**禁止修改**

- `benchmark/schema.py`

**必须交付**

- baseline registry
- 主实验可跑矩阵

**验收标准**

- main benchmark 可输出完整 baseline 对比

**推荐测试**

- `pytest tests/test_benchmark_harness.py -q`

### 任务卡 P4-S2：Persuasion robustness 实验

**背景**

v2 benchmark 新增 persuasion layer，需要独立实验承接。

**目标**

新增 persuasion robustness runner。

**前置依赖**

- P2-S1
- P4-S1

**允许修改**

- 新增 `experiments/exp_persuasion_robustness/run.py`
- `experiments/benchmark_harness.py`

**禁止修改**

- `verifier/*`

**必须交付**

- persuasion robustness artifact
- 按 pressure 类型切分的聚合报表
- 对应 markdown summary

**验收标准**

- 能按 pressure 类型切分报表

**推荐测试**

- `python -m experiments.exp_persuasion_robustness.run --help`

### 任务卡 P4-S3：Real-grounded subset 实验

**背景**

real-grounded subset 需要独立评估入口。

**目标**

新增 real-grounded subset runner。

**前置依赖**

- P2-S3
- P4-S1

**允许修改**

- 新增 `experiments/exp_real_grounded_subset/run.py`
- `benchmark/loaders.py`

**禁止修改**

- `verifier/*`

**必须交付**

- real-grounded evaluation artifact
- synthetic / real-grounded 分层报表
- 对应 markdown summary

**验收标准**

- synthetic 与 real-grounded 结果可分开汇报

**推荐测试**

- `python -m experiments.exp_real_grounded_subset.run --help`

### 任务卡 P4-S4：Witness faithfulness 实验

**背景**

需要证明 witness 不是 decoration。

**目标**

新增 witness ablation / corruption / shuffle 实验。

**前置依赖**

- P3-S2

**允许修改**

- 新增 `experiments/exp_witness_faithfulness/run.py`
- `tests/test_verifier.py`

**禁止修改**

- `benchmark/graph_families.py`

**必须交付**

- witness faithfulness artifact

**验收标准**

- 删除或破坏 witness 后，系统结果或解释质量应可观测退化

**推荐测试**

- `python -m experiments.exp_witness_faithfulness.run --help`

---

## Phase 5：主实验与反向硬化

### 任务卡 P5-S1：Main benchmark 正式跑通

**背景**

当 baseline matrix、selective metrics 与新 OOD 已接入后，必须先检查主实验是否仍然饱和，否则后续所有结论都会失真。

**目标**

跑通主表与统计显著性，并检查是否出现饱和。

**前置依赖**

- Phase 4 完成

**允许修改**

- `experiments/*`
- `outputs/mainline/*`

**禁止修改**

- `visualization/*`

**必须交付**

- main benchmark 表
- significance 报告

**验收标准**

- 若结果过于完美，必须返回 Phase 2 调 hard

**推荐测试**

- `python -m experiments.exp_main_benchmark.run`

### 任务卡 P5-S2：Leakage / Ablation / OOD / Persuasion 全套跑通

**背景**

Main benchmark 只证明整体有效性，不足以形成完整论文证据链。必须把 leakage、ablation、OOD 和 persuasion 结果全部拉齐。

**目标**

形成完整证据链。

**前置依赖**

- P5-S1

**允许修改**

- `experiments/*`
- `outputs/mainline/*`

**禁止修改**

- `benchmark/schema.py`
- `verifier/*`
- `evaluation/*`

**必须交付**

- leakage study
- ablation tables
- OOD tables
- persuasion robustness tables

**验收标准**

- 所有结果都有 markdown summary
- 所有结果都有 raw predictions

**推荐测试**

- `python -m experiments.exp_leakage_study.run`
- `python -m experiments.exp_identifiability_ablation.run`
- `python -m experiments.exp_ood_generalization.run`
- `python -m experiments.exp_persuasion_robustness.run`

---

## Phase 6：Human audit 与 paper assets

### 任务卡 P6-S1：Annotation protocol 固化

**背景**

human audit 流程已经存在，但没有正式 protocol 会导致标注口径不一致，后续 agreement 与仲裁结果不可信。

**目标**

写出正式 annotation protocol。

**前置依赖**

- P4-S3

**允许修改**

- 新增 `docs/ANNOTATION_PROTOCOL.md`
- `experiments/exp_human_audit/run.py`

**禁止修改**

- `benchmark/schema.py`
- `verifier/*`

**必须交付**

- annotation instructions
- dual-annotation format
- arbitration format

**验收标准**

- protocol 明确区分 annotator A、annotator B、arbiter 的职责
- protocol 明确四项主审字段
- human audit runner 输出格式与 protocol 一致

**推荐测试**

- `python -m experiments.exp_human_audit.run --help`

### 任务卡 P6-S2：Dual annotation 执行与汇总

**背景**

只有 annotation package 还不够，必须形成真实的双人标注、agreement 和冲突仲裁证据。

**目标**

完成 150-300 条样本的双人标注与一致性统计。

**前置依赖**

- P6-S1

**允许修改**

- `outputs/review/*`
- `experiments/exp_human_audit/run.py`

**禁止修改**

- `benchmark/*`
- `verifier/*`

**必须交付**

- agreement report
- conflict summary
- representative cases

**验收标准**

- 有 Cohen’s kappa 或等价 agreement
- 有冲突仲裁记录

**推荐测试**

- `python -m experiments.exp_human_audit.run --annotations <filled_annotations_path>`

---

## Phase 7：论文写作冻结

### 任务卡 P7-S1：主表、附表、案例图冻结

**背景**

在实验结果成熟后，必须把 paper-facing artifacts 固化，否则写作阶段会不断受未冻结结果影响。

**目标**

把所有主线结果冻结为 paper-facing artifacts。

**前置依赖**

- P5-S2
- P6-S2

**允许修改**

- `outputs/mainline/*`
- 新增 `docs/PAPER_WRITING_OUTLINE_V2.md`

**禁止修改**

- `benchmark/*`
- `verifier/*`

**必须交付**

- final tables
- figure captions
- case study notes

**验收标准**

- 主表、附表、案例图的文件路径固定
- paper outline 中的结果引用指向具体 artifact
- 不再依赖临时 review 文件生成正文图表

**推荐测试**

- 团队内部检查：每项核心 claim 是否都能对应到固定 artifact

### 任务卡 P7-S2：结论边界校正

**背景**

如果最终 claim 边界不与实验和理论完全一致，论文最容易在审稿中被抓住“过度 claim”问题。

**目标**

确保论文中所有 claim 都与实验和理论一致。

**前置依赖**

- P7-S1

**允许修改**

- 所有 docs

**禁止修改**

- `benchmark/*`
- `verifier/*`
- `evaluation/*`
- `experiments/*`

**必须交付**

- final claim checklist

**验收标准**

- 无过度 claim
- 无关键结论仅依赖 appendix

**推荐测试**

- 逐条核对摘要、贡献、结论与主表/理论 proposition 的对应关系

---

## 最终说明

这套任务卡不是为了让团队“多做事情”，而是为了保证：

- 每个阶段都直接提升论文质量
- 每个模块都服务主论点
- 每次执行都有明确的写集边界、交付物和验收标准

如果某项任务无法明确提升：

- 主问题锋利度
- benchmark 可信度
- 方法必要性
- 结果可辩护性
- artifact 可复现性

则它不应优先执行。
