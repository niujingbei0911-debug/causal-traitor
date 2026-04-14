# 🎭 The Causal Traitor（因果叛徒）

> **多智能体信息不对称下的因果欺骗与隐变量反侦察**
> Multi-Agent Causal Deception and Hidden Variable Detection under Information Asymmetry

## 📋 项目概述

本项目构建一个多智能体对抗博弈平台，探索因果推理在信息不对称场景下的攻防对抗。系统包含三个核心角色：

| 角色 | 模型 | 职责 |
|------|------|------|
| **Agent A（叛徒）** | Qwen2.5-7B | 构造因果欺骗，隐藏混杂变量 |
| **Agent B（科学家）** | Qwen2.5-14B | 识别因果谬误，发现隐变量 |
| **Agent C（审计员）** | Qwen2.5-72B | 评估因果论证质量，裁决胜负 |
| **Jury（陪审团）** | 3-5个中等模型 | 多视角投票，增强鲁棒性 |

### 核心创新点

1. **Pearl三层因果阶梯对抗**：覆盖关联 P(Y|X)、干预 P(Y|do(X))、反事实 P(Y_x|X=x',Y=y') 三个层级
2. **动态难度控制器**：基于Flow理论，目标欺骗成功率维持在0.4
3. **多轮进化博弈**：Agent策略随对局进化，追踪策略演化轨迹
4. **12种因果工具集成**：DoWhy、CausalML、causallearn等工具的实际调用

## 🏗️ 项目结构

```
causal-traitor/
├── docs/                    # 项目文档
│   ├── DESIGN.md           # 详细设计方案（v2）
│   └── TASK_ASSIGNMENT.md  # 任务分工文档
├── agents/                  # 智能体模块
│   ├── __init__.py
│   ├── agent_a.py          # 叛徒Agent（因果欺骗）
│   ├── agent_b.py          # 科学家Agent（隐变量检测）
│   ├── agent_c.py          # 审计员Agent（裁决评估）
│   ├── jury.py             # 陪审团机制
│   └── prompts/            # Agent提示词模板
│       └── __init__.py
├── causal_tools/            # 因果工具集
│   ├── __init__.py
│   ├── l1_association.py   # L1关联层工具
│   ├── l2_intervention.py  # L2干预层工具
│   ├── l3_counterfactual.py # L3反事实层工具
│   └── meta_tools.py       # 元工具（工具选择器等）
├── game/                    # 博弈引擎
│   ├── __init__.py
│   ├── debate_engine.py    # 辩论引擎
│   ├── difficulty.py       # 动态难度控制器
│   ├── evolution.py        # 策略进化追踪
│   └── data_generator.py   # 因果场景数据生成
├── evaluation/              # 评估模块
│   ├── __init__.py
│   ├── metrics.py          # 14项评估指标
│   ├── scorer.py           # 评分器
│   └── tracker.py          # 实验追踪器
├── visualization/           # 可视化模块
│   ├── __init__.py
│   ├── api.py              # FastAPI + WebSocket后端
│   └── frontend/           # React + D3.js前端
│       └── .gitkeep
├── experiments/             # 实验脚本
│   ├── exp1_basic_game/    # 实验1：基础博弈
│   ├── exp2_difficulty_adaptation/ # 实验2：难度自适应
│   ├── exp3_evolution_analysis/    # 实验3：进化分析
│   └── exp4_ablation_study/        # 实验4：消融实验
├── configs/                 # 配置文件
│   └── default.yaml
├── tests/                   # 测试
│   └── __init__.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- CUDA 12.0+（用于本地模型推理）
- Node.js 18+（用于前端可视化）

### 安装

```bash
# 克隆仓库
git clone https://github.com/niujingbei0911-debug/causal-traitor.git
cd causal-traitor

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 运行

```bash
# 启动基础博弈实验
python -m experiments.exp1_basic_game.run

# 启动Web可视化
python -m visualization.api
```

## 📅 开发计划

| 阶段 | 时间 | 内容 |
|------|------|------|
| Phase 0 | Day 1 | 环境搭建、接口定义 |
| Phase 1 | Day 1-2 | 核心引擎开发 |
| Phase 2 | Day 3-5 | Agent策略 + 因果工具 |
| Phase 3 | Day 6-7 | 评估系统 + 可视化 |
| Phase 4 | Day 8-11 | 实验运行 + 论文撰写 |

## 👥 团队分工

| 角色 | 负责模块 | 工作量 |
|------|----------|--------|
| A - 博弈架构师 | game/, agents/agent_c.py, agents/jury.py | 12.5人天 |
| B - 因果工具师 | causal_tools/, agents/agent_a.py, agents/agent_b.py | 14.5人天 |
| C - 评估可视化师 | evaluation/, visualization/, experiments/ | 14.5人天 |

详细分工见 [docs/TASK_ASSIGNMENT.md](docs/TASK_ASSIGNMENT.md)

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

## 📚 参考文献

详见 [docs/DESIGN.md](docs/DESIGN.md) 第10节

## 📄 License

MIT License
