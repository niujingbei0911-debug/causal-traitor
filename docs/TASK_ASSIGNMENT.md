# The Causal Traitor — 三人协作开发任务分工

> 基于 `causal_traitor_v2.md` 设计方案
> 3人团队并行开发计划

---

## 一、模块依赖关系总览

```
                    ┌─────────────┐
                    │  configs/   │  ← 所有模块依赖
                    │ default.yaml│
                    └──────┬──────┘
                           │
              ┌────────────┼────────────────┐
              ▼            ▼                ▼
     ┌────────────┐ ┌───────────┐  ┌──────────────┐
     │ game/      │ │ causal_   │  │ agents/      │
     │ data_      │ │ tools/    │  │ prompts/     │
     │ generator  │ │ L1/L2/L3  │  │ agent_a/b/c  │
     └─────┬──────┘ └─────┬─────┘  │ jury         │
           │              │        └──────┬───────┘
           │              └───────┐       │
           ▼                      ▼       ▼
     ┌───────────┐         ┌────────────────┐
     │ game/     │         │ game/          │
     │ difficulty│         │ debate_engine  │ ← 核心枢纽
     └─────┬─────┘         └───────┬────────┘
           │                       │
           └───────────┬───────────┘
                       ▼
              ┌────────────────┐
              │ game/          │
              │ evolution      │
              └───────┬────────┘
                      │
              ┌───────┼────────┐
              ▼       ▼        ▼
     ┌──────────┐ ┌────────┐ ┌──────────────┐
     │evaluation│ │ viz/   │ │ experiments/ │
     │ metrics  │ │ api +  │ │ exp1-4       │
     │ scorer   │ │frontend│ │              │
     │ tracker  │ └────────┘ └──────────────┘
     └──────────┘
```

**关键路径**：`configs` → `data_generator` + `causal_tools` + `agents/prompts` → `debate_engine` → `evolution` → `evaluation` + `visualization` + `experiments`

---

## 二、人员角色定义

| 角色 | 代号 | 核心职责 | 技术侧重 |
|------|------|---------|----------|
| 成员 A | **博弈架构师** | 数据生成、辩论引擎、难度控制、进化机制 | Python后端、SCM建模、博弈逻辑 |
| 成员 B | **因果工具师** | 因果工具链、Agent Prompt、陪审团机制 | 因果推断库(DoWhy/CausalML)、LLM Prompt工程 |
| 成员 C | **评估可视化师** | 评估体系、可视化前后端、实验脚本 | React/D3.js前端、FastAPI、实验设计 |

---

## 三、开发阶段与任务分配

### Phase 0：项目初始化（第1天，3人协作）

所有人共同完成，确保开发环境一致：

| 任务 | 负责人 | 产出 |
|------|--------|------|
| 创建Git仓库 + 分支策略 | 成员 A | `main`, `dev`, 个人分支 |
| `requirements.txt` + 虚拟环境 | 成员 A | 依赖锁定 |
| `configs/default.yaml` 配置结构 | 成员 B | 全局配置模板 |
| 项目目录骨架 + `README.md` | 成员 C | 目录结构 |
| 统一代码规范（black/ruff/mypy） | 全员 | `.pre-commit-config.yaml` |

---

### Phase 1：基础层并行开发（第1-2天）

三人完全并行，无交叉依赖：

#### 成员 A — 博弈架构师

| 任务 | 文件 | 工作量 | 说明 |
|------|------|--------|------|
| SCM数据生成器 | `game/data_generator.py` | ★★★ | 3个场景(吸烟/教育/药物)的SCM定义、观测数据采样、隐变量注入 |
| 难度控制器 | `game/difficulty.py` | ★★☆ | `DifficultyController`类，Flow理论自适应调节，6参数体系 |
| 配置加载器 | `game/config.py` | ★☆☆ | YAML配置解析，运行时参数管理 |

**Phase 1 产出**：能独立生成带隐变量的因果数据集，难度可调

#### 成员 B — 因果工具师

| 任务 | 文件 | 工作量 | 说明 |
|------|------|--------|------|
| L1关联层工具 | `causal_tools/l1_association.py` | ★★☆ | `correlation_analysis`, `conditional_independence_test` |
| L2干预层工具 | `causal_tools/l2_intervention.py` | ★★★ | `backdoor_adjustment_check`, `iv_estimation`, `sensitivity_analysis`, `frontdoor_estimation` |
| L3反事实层工具 | `causal_tools/l3_counterfactual.py` | ★★★ | `counterfactual_inference`, `scm_identification_test`, `ett_computation`, `abduction_action_prediction` |
| 元工具 | `causal_tools/meta_tools.py` | ★★☆ | `argument_logic_check`, `causal_graph_validator`, `select_tools`决策树 |

**Phase 1 产出**：完整的12个因果分析工具函数，可独立调用测试

#### 成员 C — 评估可视化师

| 任务 | 文件 | 工作量 | 说明 |
|------|------|--------|------|
| 评估指标计算 | `evaluation/metrics.py` | ★★★ | 14个指标的计算函数（DSR/DCS/DCX/DAcc/DPre/DRec/CLS/TUR/FDR/ARI/NC/SD/JAcc/JCon） |
| 综合评分器 | `evaluation/scorer.py` | ★★☆ | `compute_overall_score`，加权评分公式 |
| 进化追踪器 | `evaluation/tracker.py` | ★★☆ | `EvolutionTracker`类，策略多样性/军备竞赛指数/Nash收敛度 |
| 数据模型定义 | `evaluation/models.py` | ★☆☆ | Pydantic数据模型，统一各模块间的数据结构 |

**Phase 1 产出**：完整的评估计算引擎，可接收模拟数据进行评分

---

### Phase 2：智能体层并行开发（第3-5天）

依赖Phase 1的基础模块，三人继续并行：

#### 成员 A — 博弈架构师

| 任务 | 文件 | 工作量 | 说明 |
|------|------|--------|------|
| 辩论引擎核心 | `game/debate_engine.py` | ★★★★ | 三阶辩论协议调度、轮次管理、消息路由、Agent调用编排 |
| 进化引擎 | `game/evolution.py` | ★★★ | 策略总结生成、进化Prompt注入、多轮反馈循环 |

**依赖**：`game/data_generator.py`(自己Phase1产出) + `agents/`(成员B产出，可先用Mock)

#### 成员 B — 因果工具师

| 任务 | 文件 | 工作量 | 说明 |
|------|------|--------|------|
| Agent A Prompt + 逻辑 | `agents/agent_a.py` + `agents/prompts/agent_a_*.py` | ★★★ | 7B叛徒角色，L1/L2/L3欺骗策略库(12种策略)，进化Prompt |
| Agent B Prompt + 逻辑 | `agents/agent_b.py` + `agents/prompts/agent_b_*.py` | ★★☆ | 14B科学家角色，因果假设提出，防御策略 |
| Agent C Prompt + 逻辑 | `agents/agent_c.py` + `agents/prompts/agent_c_*.py` | ★★★ | 72B审计官角色，工具调用编排，代码执行沙箱集成，判决逻辑 |
| 陪审团机制 | `agents/jury.py` + `agents/prompts/jury_*.py` | ★★★ | 陪审员Prompt、`JuryAggregator`加权投票、共识度计算、Agent C整合逻辑 |

**依赖**：`causal_tools/`(自己Phase1产出) + `configs/`

#### 成员 C — 评估可视化师

| 任务 | 文件 | 工作量 | 说明 |
|------|------|--------|------|
| FastAPI后端 | `visualization/api.py` | ★★★ | REST API + WebSocket实时推送，辩论状态流 |
| React前端框架 | `visualization/frontend/` | ★★★★ | 项目脚手架、路由、状态管理、Tailwind样式 |
| 因果图可视化组件 | `visualization/frontend/src/components/CausalGraph.tsx` | ★★★ | D3.js因果DAG渲染，隐变量高亮，动态更新 |

**依赖**：`evaluation/`(自己Phase1产出) + API接口约定(与成员A协商)

---

### Phase 3：集成与联调（第6-7天）

模块间开始对接，需要密切协作：

#### 成员 A — 博弈架构师

| 任务 | 文件 | 工作量 | 说明 |
|------|------|--------|------|
| 全流程集成 | `main.py` | ★★★ | 串联 data_generator → agents → debate_engine → jury → evaluation → evolution |
| LLM服务对接 | `game/llm_service.py` | ★★☆ | vLLM/Ollama统一接口，Qwen2.5-7B/14B/72B模型加载 |
| 端到端冒烟测试 | `tests/test_integration.py` | ★★☆ | 单轮完整辩论流程验证 |

#### 成员 B — 因果工具师

| 任务 | 文件 | 工作量 | 说明 |
|------|------|--------|------|
| Agent ↔ 工具链集成 | `agents/tool_executor.py` | ★★★ | Agent C代码执行沙箱(RestrictedPython/Docker)，工具调用安全封装 |
| Prompt调优 | `agents/prompts/*.py` | ★★★ | 基于实际LLM输出调整所有Prompt，确保JSON格式输出稳定 |
| 单元测试 | `tests/test_agents.py`, `tests/test_tools.py` | ★★☆ | 各Agent和工具的独立测试 |

#### 成员 C — 评估可视化师

| 任务 | 文件 | 工作量 | 说明 |
|------|------|--------|------|
| 辩论实时面板 | `visualization/frontend/src/components/DebatePanel.tsx` | ★★★ | WebSocket接收辩论流，实时渲染对话 |
| 陪审团面板 | `visualization/frontend/src/components/JuryPanel.tsx` | ★★☆ | 投票动画、共识度仪表盘 |
| 难度/进化面板 | `visualization/frontend/src/components/DifficultyPanel.tsx` | ★★☆ | 难度曲线图、进化轨迹图、DSR趋势线 |
| 前后端联调 | — | ★★☆ | API对接、WebSocket调试 |

---

### Phase 4：实验与收尾（第8-11天）

#### 成员 A — 博弈架构师

| 任务 | 文件 | 工作量 | 说明 |
|------|------|--------|------|
| 实验一：因果层级基准 | `experiments/exp1_causal_levels/` | ★★★ | L1/L2/L3各20轮，收集DSR/DAcc/CLS |
| 实验四：进化博弈 | `experiments/exp4_evolution/` | ★★☆ | 有/无进化对比，10轮 |

#### 成员 B — 因果工具师

| 任务 | 文件 | 工作量 | 说明 |
|------|------|--------|------|
| 实验二：陪审团消融 | `experiments/exp2_jury_ablation/` | ★★★ | 无/3人/5人陪审团对比，各30轮 |
| 实验三：动态难度 | `experiments/exp3_difficulty/` | ★★☆ | 固定简单/固定困难/动态调节对比 |

#### 成员 C — 评估可视化师

| 任务 | 文件 | 工作量 | 说明 |
|------|------|--------|------|
| 实验结果可视化 | `visualization/frontend/src/pages/Results.tsx` | ★★★ | 4个实验的图表展示 |
| 课程展示Demo脚本 | `demo/demo_script.py` | ★★☆ | 15分钟演示流程自动化 |
| 课程PPT协助 | — | ★★☆ | 截图、数据整理 |

---

## 四、工作量平衡分析

### 按阶段统计（★ = 0.5人天）

| 阶段 | 成员A(博弈架构师) | 成员B(因果工具师) | 成员C(评估可视化师) |
|------|:-:|:-:|:-:|
| Phase 0 | 0.5天 | 0.5天 | 0.5天 |
| Phase 1 | 3天 (★×6) | 4天 (★×10) | 3.5天 (★×8) |
| Phase 2 | 3.5天 (★×7) | 4天 (★×11) | 4天 (★×10) |
| Phase 3 | 3天 (★×7) | 3.5天 (★×8) | 3.5天 (★×9) |
| Phase 4 | 2.5天 (★×5) | 2.5天 (★×5) | 3天 (★×7) |
| **合计** | **12.5天** | **14.5天** | **14.5天** |

> 成员A工作量略少，因此承担Phase 0的仓库搭建和Phase 3的全流程集成主导角色作为补偿。

### 技能要求

| 成员 | 必备技能 | 加分技能 |
|------|---------|---------|
| 成员A | Python、概率图模型/SCM基础、异步编程 | LangChain/LlamaIndex |
| 成员B | Python、因果推断理论(Pearl)、DoWhy/CausalML | Prompt工程、Docker |
| 成员C | React+TypeScript、D3.js、FastAPI | WebSocket、数据可视化 |

---

## 五、接口约定（并行开发关键）

为确保三人并行开发时模块可独立编译测试，需在Phase 0约定以下接口：

### 5.1 数据生成器输出格式

```python
# 成员A定义，成员B/C遵循
@dataclass
class CausalScenario:
    scenario_id: str                    # "smoking_cancer" | "education_income" | "drug_recovery"
    causal_level: str                   # "L1" | "L2" | "L3"
    observed_data: pd.DataFrame         # 观测数据 X
    full_data: pd.DataFrame             # 完整数据 X + U（仅Agent A可见）
    true_scm: dict                      # 真实SCM结构
    hidden_variables: list[str]         # 隐变量名称列表
    ground_truth: dict                  # 真实因果效应
    difficulty_config: dict             # 难度参数
```

### 5.2 Agent统一接口

```python
# 成员B定义，成员A的debate_engine调用
class BaseAgent(ABC):
    @abstractmethod
    async def respond(self, context: DebateContext) -> AgentResponse:
        """生成一轮辩论回复"""
        pass

@dataclass
class DebateContext:
    scenario: CausalScenario
    history: list[dict]                 # 历史对话记录
    current_round: int
    causal_level: str
    evolution_context: dict | None      # 进化策略（可选）

@dataclass
class AgentResponse:
    content: str                        # 回复文本
    strategy_used: str                  # 使用的策略编号
    confidence: float                   # 置信度
    metadata: dict                      # 额外元数据
```

### 5.3 评估接口

```python
# 成员C定义，成员A的evolution模块调用
class GameEvaluator:
    def evaluate_round(self, round_data: RoundData) -> RoundMetrics:
        """评估单轮结果"""
        pass

    def evaluate_session(self, session_data: list[RoundData]) -> SessionMetrics:
        """评估整个会话"""
        pass

@dataclass
class RoundData:
    scenario: CausalScenario
    debate_transcript: list[dict]
    jury_votes: list[dict]
    agent_c_verdict: dict
    ground_truth: dict
    deception_success: bool
```

### 5.4 可视化API

```python
# 成员C定义，成员A的main.py调用
# WebSocket事件格式
{
    "event": "debate_turn" | "jury_vote" | "verdict" | "difficulty_update" | "evolution_step",
    "data": { ... },
    "timestamp": "ISO8601"
}
```

---

## 六、Git分支策略

```
main ─────────────────────────────────────────── 稳定版本
  │
  └── dev ────────────────────────────────────── 集成分支
       ├── feat/game-engine      (成员A)
       ├── feat/causal-tools     (成员B)
       ├── feat/eval-viz         (成员C)
       ├── feat/agents           (成员B)
       ├── feat/debate-engine    (成员A)
       └── feat/frontend         (成员C)
```

**合并规则**：
- 个人分支 → `dev`：需至少1人Code Review
- `dev` → `main`：需全员确认 + 通过CI测试

---

## 七、里程碑检查点

| 时间 | 里程碑 | 验收标准 |
|------|--------|---------|
| Day 1 | M0: 项目初始化 | 仓库创建、环境统一、接口约定文档完成 |
| Day 2 | M1: 基础层完成 | 数据生成器可运行、12个因果工具可调用、评估引擎可计算 |
| Day 5 | M2: 智能体层完成 | 4个Agent可独立对话、陪审团可投票、前端框架可访问 |
| Day 7 | M3: 集成联调完成 | 端到端单轮辩论可运行、前端可实时展示 |
| Day 11 | M4: 实验与展示就绪 | 4个实验完成、Demo脚本可运行、PPT就绪 |

---

## 八、风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| 72B模型本地跑不动 | 高 | 高 | 备选方案：用API调用(如通义千问API)或降级为32B |
| Prompt输出格式不稳定 | 中 | 中 | 成员B在Phase 2预留1天做格式鲁棒性处理(retry+正则提取) |
| 前后端联调延迟 | 中 | 低 | 成员C在Phase 2先用Mock数据开发前端，Phase 3再对接真实API |
| 因果工具计算耗时 | 低 | 中 | 对耗时工具加缓存，实验时用小数据集先验证 |

---

> **文档版本**: v1.0
> **创建时间**: 2025年4月14日
> **基于**: `causal_traitor_v2.md` 设计方案
