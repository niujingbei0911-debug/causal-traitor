# 🎭 The Causal Traitor（因果叛徒）

> **多智能体信息不对称下的因果欺骗与隐变量反侦察**
> Multi-Agent Causal Deception and Hidden Variable Detection under Information Asymmetry

## 📋 项目概述

本项目构建一个多智能体对抗博弈平台，探索因果推理在信息不对称场景下的攻防对抗。系统包含三个核心角色：

| 角色 | 模型 | 职责 |
|------|------|------|
| **Agent A（叛徒）** | Qwen2.5-7B | 构造因果欺骗，隐藏混杂变量（11种欺骗策略） |
| **Agent B（科学家）** | Qwen2.5-14B | 识别因果谬误，发现隐变量 |
| **Agent C（审计员）** | Qwen2.5-72B | 评估因果论证质量，裁决胜负 |
| **Jury（陪审团）** | 3-5个中等模型 | 多视角加权投票，增强鲁棒性 |

当前实现不是简单的“规则脚本演示”。A/B/C 现在都走 **LLM-first 的结构化决策链**：
- Agent A 先让 LLM 产出误导性 `causal_claim/content/evidence/strategy`
- Agent B 先让 LLM 产出结构化检测结论，再和工具结果合并
- Agent C 先让 LLM 产出结构化裁决，再与 jury / tool executor 结果对齐

工具链仍然非常重要，但主要作用已经从“直接决定一切”转成了“证据、约束与 fallback”。

### 核心创新点

1. **Pearl三层因果阶梯对抗**：覆盖关联 P(Y|X)、干预 P(Y|do(X))、反事实 P(Y_x|X=x',Y=y') 三个层级
2. **动态难度控制器**：基于Flow理论，目标DSR维持在30%-50%
3. **多轮进化博弈**：Agent A 策略回避 + Agent C 防御升级 + 差异化进化上下文，军备竞赛检测驱动对抗升级
4. **12种因果工具集成**：DoWhy、CausalML、causallearn等工具的实际调用
5. **Mock Fallback机制**：无需LLM API即可完整运行，内置统计校准的Mock Agent

### 当前运行状态

- `main.py`、`exp1`、`exp2`、`exp3`、`exp4` 都可以实际运行并落盘 `JSON/CSV/MD` 结果。
- `evaluation/tracker.py` 会把每次运行写到 `logs/<run_id>/`。
- 前端可通过 `run_live_game.py` 收到实时 WebSocket 事件。
- DashScope API 可作为默认后端；断网或调用失败时会退回 mock fallback。
- 当前博弈仍偏向 `B + C + jury` 一侧，真实实验里 `Agent A` 还需要继续增强。

## 🏗️ 项目结构

```
causal-traitor/
├── main.py                      # CLI入口，端到端运行辩论博弈
├── run_live_game.py             # 实时博弈入口（WebSocket事件推送）
├── CLAUDE.md                    # AI辅助开发指引
├── requirements.txt             # Python依赖
├── .pre-commit-config.yaml      # 代码规范（black/ruff/mypy）
│
├── docs/                        # 项目文档
│   ├── DESIGN.md               # 详细设计方案（v2.1）
│   ├── TASK_ASSIGNMENT.md      # 三人协作任务分工
│   ├── CODE_REVIEW_REPORT.md   # 代码审查报告
│   └── PROJECT_PROGRESS_REPORT.md # 项目进度报告
│
├── agents/                      # 智能体模块
│   ├── agent_a.py              # 叛徒Agent（因果欺骗，11种策略）
│   ├── agent_b.py              # 科学家Agent（隐变量检测）
│   ├── agent_c.py              # 审计员Agent（裁决评估）
│   ├── jury.py                 # 陪审团机制（加权投票）
│   ├── tool_executor.py        # Agent C 工具调用编排 + 沙箱执行
│   └── prompts/                # Agent提示词模板
│       ├── agent_a_prompts.py
│       ├── agent_b_prompts.py
│       ├── agent_c_prompts.py
│       └── jury_prompts.py
│
├── causal_tools/                # 因果工具集（12个工具函数）
│   ├── l1_association.py       # L1关联层：correlation_analysis, conditional_independence_test
│   ├── l2_intervention.py      # L2干预层：backdoor_adjustment, iv_estimation, sensitivity_analysis, frontdoor_estimation
│   ├── l3_counterfactual.py    # L3反事实层：counterfactual_inference, scm_identification, ett_computation, abduction_action_prediction
│   └── meta_tools.py           # 元工具：argument_logic_check, causal_graph_validator, select_tools
│
├── game/                        # 博弈引擎
│   ├── config.py               # YAML配置加载器
│   ├── types.py                # 共享数据结构（GamePhase, CausalScenario等）
│   ├── llm_service.py          # LLM后端适配（mock/dashscope/vllm/ollama）
│   ├── debate_engine.py        # 辩论引擎（真实Agent优先，失败时回退Mock）
│   ├── difficulty.py           # 动态难度控制器（Flow理论，DSR目标30%-50%）
│   ├── evolution.py            # 策略进化追踪（11种策略 + 军备竞赛 + 复杂度/灵敏度趋势）
│   └── data_generator.py       # 因果场景数据生成（3个SCM场景）
│
├── evaluation/                  # 评估模块
│   ├── metrics.py              # 14项评估指标（CausalMetrics）
│   ├── scorer.py               # 评分器（因果论证质量评估）
│   └── tracker.py              # 实验追踪器（ExperimentTracker）
│
├── visualization/               # 可视化模块
│   ├── api.py                  # FastAPI + WebSocket后端（实时事件推送）
│   └── frontend/               # React + D3.js + Tailwind CSS 前端
│       ├── index.html          # 入口HTML
│       ├── package.json        # Node依赖（react, d3, tailwindcss, vite）
│       ├── vite.config.ts      # Vite构建配置
│       ├── tailwind.config.js  # Tailwind配置
│       ├── postcss.config.js   # PostCSS配置
│       ├── tsconfig.json       # TypeScript配置
│       └── src/
│           ├── App.tsx         # 主应用组件（四面板布局）
│           ├── main.tsx        # React入口
│           ├── types.ts        # TypeScript类型定义
│           ├── useGameSocket.ts # WebSocket Hook（实时数据流）
│           ├── index.css       # 全局样式
│           └── components/
│               ├── CausalGraph.tsx    # D3.js因果图可视化
│               ├── DebatePanel.tsx    # 辩论过程面板
│               ├── JuryPanel.tsx      # 陪审团投票面板
│               └── DifficultyPanel.tsx # 难度曲线面板
│
├── experiments/                 # 实验脚本
│   ├── exp1_causal_levels/     # 实验1：因果层级基准（L1/L2/L3对比）
│   ├── exp2_jury_ablation/     # 实验2：陪审团消融实验
│   ├── exp3_difficulty/        # 实验3：动态难度 vs 固定难度对比
│   └── exp4_evolution/         # 实验4：进化博弈分析
│
├── configs/
│   └── default.yaml            # 全局配置（模型、难度、进化参数）
│
└── tests/                       # 测试
    ├── test_agents.py          # Agent单元测试
    ├── test_tools.py           # 因果工具测试
    ├── test_tool_executor.py   # 工具执行器测试
    └── test_integration.py     # 端到端集成测试
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+（仅前端可视化需要）
- DashScope / OpenAI-compatible API key（若要跑真实 LLM 决策）
- CUDA 12.0+（仅本地LLM推理需要，当前 `vllm` / `ollama` 仍是占位后端）

### 安装

```bash
# 克隆仓库
git clone https://github.com/niujingbei0911-debug/causal-traitor.git
cd causal-traitor

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装Python依赖
pip install -r requirements.txt

# （可选）安装前端依赖
cd visualization/frontend
npm install
cd ../..
```

### 运行

```bash
# 1. 运行完整辩论博弈
python main.py --rounds 3 --output outputs/run.json

# 2. 启动可视化后端（FastAPI + WebSocket）
python -c "import uvicorn; from visualization.api import VisualizationAPI; uvicorn.run(VisualizationAPI({'api_host':'127.0.0.1','api_port':8001,'websocket_path':'/ws/game'}).create_app(), host='127.0.0.1', port=8001)"

# 3. 启动前端开发服务器
cd visualization/frontend && npm install && npm run dev -- --host 127.0.0.1 --port 5173

# 4. 推送实时博弈到前端
python run_live_game.py --rounds 6 --delay 1.0 --ws ws://127.0.0.1:8001/ws/game

# 5. 运行实验
python -m experiments.exp1_causal_levels.run --rounds-per-level 20
python -m experiments.exp2_jury_ablation.run --rounds 30 --level 2
python -m experiments.exp3_difficulty.run --rounds 30
python -m experiments.exp4_evolution.run --rounds 10 --level 2

# 6. 运行测试
pytest tests/ -v
```

### LLM后端配置

在 `configs/default.yaml` 中配置模型后端：

```yaml
models:
  agent_a:
    name: qwen2.5-7b-instruct
    backend: dashscope
  agent_b:
    name: qwen2.5-14b-instruct
    backend: dashscope
  agent_c:
    name: qwen2.5-72b-instruct
    backend: dashscope
```

- `dashscope` / `api`：真实 Qwen API 路径
- `mock`：离线 fallback，用于测试、断网和本地联调
- `vllm` / `ollama`：已保留配置入口，但当前仍会退回 mock
- API key 通过 `DASHSCOPE_API_KEY` 或 `OPENAI_API_KEY` 读取，也支持在模型配置里显式传 `api_key`

## 📊 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React + D3.js)              │
│  CausalGraph │ DebatePanel │ JuryPanel │ DifficultyPanel │
└──────────────────────┬──────────────────────────────────┘
                       │ WebSocket
┌──────────────────────┴──────────────────────────────────┐
│              FastAPI + WebSocket Backend                  │
│                  visualization/api.py                     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│                   DebateEngine                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────────┐  │
│  │ Agent A │  │ Agent B │  │ Agent C │  │   Jury    │  │
│  │ (叛徒)  │  │ (科学家)│  │ (审计员)│  │ (陪审团)  │  │
│  │  7B     │  │  14B    │  │  72B    │  │ 3-5×中等  │  │
│  └────┬────┘  └────┬────┘  └────┬────┘  └─────┬─────┘  │
│       │            │            │              │         │
│       │      LLM-first structured decisions + tools      │
│  ┌────┴────────────┴────────────┴──────────────┘        │
│  │         causal_tools (12个因果分析工具)                │
│  │    L1: correlation, CI_test                           │
│  │    L2: backdoor, IV, sensitivity, frontdoor           │
│  │    L3: counterfactual, SCM_id, ETT, abduction        │
│  │    Meta: logic_check, graph_validator, select_tools   │
│  └──────────────────────────────────────────────────────│
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Difficulty   │  │  Evolution   │  │  Evaluation   │  │
│  │ Controller   │  │  Tracker     │  │  Metrics(14)  │  │
│  │ (Flow理论)   │  │ (军备竞赛)   │  │  Scorer       │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 📅 当前状态

| 模块 | 状态 | 说明 |
|------|------|------|
| `game/` | ✅ 可运行 | 数据生成、难度控制、进化追踪、统一入口已接通 |
| `causal_tools/` | ✅ 可运行 | L1/L2/L3 与 meta tools 均可执行 |
| `agents/` | ✅ 可运行 | A/B/C 为 LLM-first + tool-backed，进化对抗已实现 |
| `experiments/` | ✅ 可运行 | exp1/2/3/4 均可落盘结果 |
| `visualization/` | ✅ 可运行 | 前后端和 live game 数据流可启动 |
| DashScope LLM | ✅ 已集成 | 真实 LLM 调用 + 超时容错 + Mock 回退 |
| 进化对抗 | ✅ 已实现 | 策略回避 + 防御升级 + 差异化上下文 |
| 游戏平衡 | 🚧 待调参 | 真实实验中 A 仍偏弱，需进一步调参 |

## 👥 团队分工

| 角色 | 负责模块 | 工作量 |
|------|----------|--------|
| A - 博弈架构师 | game/, main.py, tests/test_integration.py | 12.5人天 |
| B - 因果工具师 | causal_tools/, agents/（全部Agent + prompts + tool_executor） | 14.5人天 |
| C - 评估可视化师 | evaluation/, visualization/, experiments/ | 14.5人天 |

详细分工见 [docs/TASK_ASSIGNMENT.md](docs/TASK_ASSIGNMENT.md)

## 📚 文档索引

| 文档 | 说明 |
|------|------|
| [docs/DESIGN.md](docs/DESIGN.md) | 系统设计方案（v2），含Pearl因果阶梯协议、博弈机制、评估体系 |
| [docs/TASK_ASSIGNMENT.md](docs/TASK_ASSIGNMENT.md) | 三人协作开发任务分工与里程碑 |
| [docs/CODE_REVIEW_REPORT.md](docs/CODE_REVIEW_REPORT.md) | 代码审查报告，含架构评估与优化建议 |
| [docs/PROJECT_PROGRESS_REPORT.md](docs/PROJECT_PROGRESS_REPORT.md) | 项目进度报告，含DSR优化过程与实验结果 |

## 🔀 Git 协作规范

- `main` — 稳定发布分支
- `dev` — 开发集成分支
- `feat/*` — 功能开发分支

```bash
# 开发新功能
git checkout dev
git checkout -b feat/your-feature
# ... 开发 ...
git push origin feat/your-feature
# 创建 PR → dev
```

## 📄 License

MIT License
