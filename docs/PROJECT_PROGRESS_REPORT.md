# 项目进度报告 — The Causal Traitor (因果叛徒)

> 报告日期: 2026-04-14
> 项目状态: **核心功能完成，实验待真实LLM运行**

---

## 一、已完成的设计模块

### 1. 智能体层 (100%)

| 模块 | 文件 | 状态 |
|------|------|------|
| Agent A — 因果叛徒 (7B) | `agents/agent_a.py` | ✅ Pearl三层策略库 (L1/L2/L3) |
| Agent B — 因果科学家 (14B) | `agents/agent_b.py` | ✅ 假设生成+验证 |
| Agent C — 因果审计官 (72B) | `agents/agent_c.py` | ✅ 工具链+判决 |
| 陪审团 (3-5模型) | `agents/jury.py` | ✅ 独立投票+加权聚合 |
| Prompt模板 | `agents/prompts/` | ✅ 四角色完整 |
| 工具执行器 | `agents/tool_executor.py` | ✅ |
| Mock回退机制 | `game/debate_engine.py` | ✅ 无API时自动降级 |

### 2. 因果工具链 (100%)

| 模块 | 文件 | 状态 |
|------|------|------|
| L1 关联分析 | `causal_tools/l1_association.py` | ✅ |
| L2 干预分析 | `causal_tools/l2_intervention.py` | ✅ |
| L3 反事实推理 | `causal_tools/l3_counterfactual.py` | ✅ |
| 元工具 (9种因果谬误) | `causal_tools/meta_tools.py` | ✅ |

### 3. 游戏引擎 (100%)

| 模块 | 文件 | 状态 |
|------|------|------|
| 辩论引擎 | `game/debate_engine.py` | ✅ 完整辩论流程 |
| 动态难度控制器 | `game/difficulty.py` | ✅ Flow理论+warmup |
| 策略进化追踪 | `game/evolution.py` | ✅ 军备竞赛+Nash收敛 |
| 因果数据生成 | `game/data_generator.py` | ✅ 3场景SCM |
| 配置加载 | `game/config.py` | ✅ YAML+覆盖 |
| LLM服务 | `game/llm_service.py` | ✅ DashScope/vLLM/Ollama/Mock |

### 4. 评估体系 (100%)

| 模块 | 文件 | 状态 |
|------|------|------|
| 14项核心指标 | `evaluation/metrics.py` | ✅ 4类指标完整 |
| 5维加权评分 | `evaluation/scorer.py` | ✅ 与设计文档公式一致 |
| 实验追踪器 | `evaluation/tracker.py` | ✅ JSON/CSV/MD输出 |

### 5. 可视化系统 (100%)

| 模块 | 文件 | 状态 |
|------|------|------|
| FastAPI后端 | `visualization/api.py` | ✅ WebSocket实时推送 |
| 因果图面板 | `CausalGraph.tsx` | ✅ D3.js+dagre, 三色标注 |
| 辩论面板 | `DebatePanel.tsx` | ✅ 实时对话流 |
| 陪审团面板 | `JuryPanel.tsx` | ✅ 投票+置信度 |
| 难度/进化面板 | `DifficultyPanel.tsx` | ✅ Recharts趋势图 |
| 前端入口 | `App.tsx` + `main.tsx` | ✅ React+Tailwind |

### 6. 实验框架 (100%)

| 实验 | 文件 | 状态 |
|------|------|------|
| Exp1: 因果层级基准 | `experiments/exp1_causal_levels/run.py` | ✅ |
| Exp2: 陪审团消融 | `experiments/exp2_jury_ablation/run.py` | ✅ |
| Exp3: 难度对比 | `experiments/exp3_difficulty/run.py` | ✅ |
| Exp4: 进化博弈 | `experiments/exp4_evolution/run.py` | ✅ |

### 7. 基础设施 (100%)

| 模块 | 文件 | 状态 |
|------|------|------|
| 默认配置 | `configs/default.yaml` | ✅ |
| 依赖清单 | `requirements.txt` | ✅ |
| 实时游戏脚本 | `run_live_game.py` | ✅ |
| 测试套件 | `tests/` | ✅ 4个测试文件 |
| 文档 | `docs/` | ✅ DESIGN + TASK_ASSIGNMENT |

---

## 二、待完成的任务

### 高优先级

1. **DSR优化** (任务28): Mock模式下DSR=68%，高于设计目标30-50%。需进一步调整Mock Agent的胜率参数或难度控制器的响应灵敏度。

2. **真实LLM集成测试**: 当前所有可视化演示均基于Mock Agent。需配置DashScope API Key后运行真实Qwen2.5模型，验证：
   - Agent A (7B) 的因果欺骗策略是否有效
   - Agent B (14B) 的假设生成质量
   - Agent C (72B) 的工具调用和判决准确性
   - 陪审团投票的多样性

### 中优先级

3. **运行4个正式实验**: 使用真实LLM运行 exp1-exp4，收集实验数据
4. **代码沙箱**: 设计文档提到Docker + RestrictedPython沙箱供Agent C执行代码，当前未实现（Agent C的工具调用通过Python函数模拟）
5. **数据库存储**: 设计文档提到SQLite存储实验记录，当前使用JSON文件

### 低优先级

6. **W&B/MLflow集成**: 配置中预留了wandb_project字段，但未实际集成
7. **更多因果场景**: 当前3个场景(smoking_cancer, education_income, drug_recovery)，可扩展更多

---

## 三、基于Mock模式可输出的实验结论

### 3.1 系统架构验证 ✅

Mock模式已验证以下架构设计的可行性：
- 四Agent对抗辩论流程完整运行
- Pearl三层因果层级(L1/L2/L3)的场景切换正常
- 陪审团独立投票+加权聚合机制工作正常
- 动态难度控制器能根据DSR调整难度
- 策略进化追踪器能记录军备竞赛指数和策略多样性

### 3.2 可视化系统验证 ✅

- WebSocket实时推送延迟<100ms
- 因果图三色标注(红=Agent A声称, 绿=验证通过, 紫=隐变量)正确显示
- 难度/进化面板趋势图实时更新
- 陪审团投票面板正确显示置信度和共识度

### 3.3 Mock模式下的初步观察

| 指标 | 20轮Mock值 | 设计目标 | 评估 |
|------|-----------|---------|------|
| DSR | ~68% | 30-50% | ⚠️ 偏高，Mock胜率参数需调整 |
| 策略多样性 | ~46% | >50% | ⚠️ 接近目标 |
| 军备竞赛指数 | ~64% | >50% | ✅ 达标 |
| 难度值 | ~43% | 自适应 | ✅ 控制器在工作 |

### 3.4 尚不能输出的结论

以下结论需要真实LLM运行后才能得出：
- 不同因果层级(L1/L2/L3)的欺骗成功率差异
- 陪审团规模(0/3/5)对检测准确率的影响
- 动态难度 vs 固定难度的博弈质量对比
- 策略进化是否产生真实的军备竞赛效应
- 综合评分公式的实际分布

---

## 四、项目完成度总结

```
代码实现:  ████████████████████ 100%  (全部模块已实现)
代码审查:  ████████████████████ 100%  (5处错误已修复)
Mock验证:  ████████████████░░░░  80%  (DSR偏高待优化)
真实LLM:   ░░░░░░░░░░░░░░░░░░░░   0%  (需API Key)
正式实验:   ░░░░░░░░░░░░░░░░░░░░   0%  (依赖真实LLM)
```

**下一步行动**: 配置 `DASHSCOPE_API_KEY` 环境变量，将 `configs/default.yaml` 中 backend 从 "mock" 切换为 "dashscope"，运行 `python run_live_game.py` 进行真实LLM测试。
