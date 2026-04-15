# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**The Causal Traitor (因果叛徒)** — A multi-agent adversarial debate system for causal reasoning research. Three LLM agents (Traitor, Scientist, Auditor) + a Jury debate causal claims across Pearl's 3-layer causal hierarchy (Association → Intervention → Counterfactual). The Traitor constructs plausible but flawed causal arguments; the Scientist detects fallacies; the Auditor renders verdicts.

## Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Recommended local runtime in this repo (if present)
./miniconda-env/bin/python main.py --rounds 1 --output outputs/run.json

# Run a single game via the unified entry point
python main.py --rounds 3 --output outputs/run.json

# Run experiments (each writes JSON + CSV + Markdown sidecars)
python -m experiments.exp1_causal_levels.run --rounds-per-level 20
python -m experiments.exp2_jury_ablation.run --rounds 30 --level 2
python -m experiments.exp3_difficulty.run --rounds 30
python -m experiments.exp4_evolution.run --rounds 10 --level 2

# Run visualization backend (FastAPI + WebSocket)
python -c "import uvicorn; from visualization.api import VisualizationAPI; uvicorn.run(VisualizationAPI({'api_host':'127.0.0.1','api_port':8001,'websocket_path':'/ws/game'}).create_app(), host='127.0.0.1', port=8001)"

# Run visualization frontend
cd visualization/frontend && npm install && npm run dev -- --host 127.0.0.1 --port 5173

# Stream a live game into the frontend
python run_live_game.py --rounds 6 --delay 1.0 --ws ws://127.0.0.1:8001/ws/game

# Tests
pytest
pytest tests/test_specific.py          # single file
pytest tests/test_specific.py::test_fn  # single test
pytest -x                               # stop on first failure
```

## Architecture

4-layer system with information asymmetry at its core:

```
Layer 4: Visualization    — FastAPI REST + WebSocket (visualization/api.py)
Layer 3: Evaluation        — 14 metrics across 4 categories (evaluation/)
Layer 2: Game Engine       — Debate orchestration + difficulty control (game/)
Layer 1: Agents            — A(7B Traitor) → B(14B Scientist) → C(72B Auditor) + Jury (agents/)
Cross-cutting: Causal Tools — Pearl L1/L2/L3 toolchain (causal_tools/)
```

### Debate Protocol Flow
`SETUP → CLAIM (A) → CHALLENGE (B) → REBUTTAL (A) → JURY → AUDIT (C) → COMPLETE`

The Jury votes before the Auditor so Agent C can factor jury consensus into its verdict.

### Agent Roles & Information Asymmetry
- **Agent A** (Traitor, Qwen2.5-7B): Has full SCM + hidden variables. Generates deceptive causal claims.
- **Agent B** (Scientist, Qwen2.5-14B): Sees only observational data. Detects fallacies using causal tools.
- **Agent C** (Auditor, Qwen2.5-72B): Evaluates both sides, executes tools, renders verdict.
- **Jury**: 3-5 models with majority/weighted/Bayesian voting.

### Causal Toolchain (Pearl's Ladder)
- **L1 Association** (`causal_tools/l1_association.py`): Correlation, conditional independence, Simpson's paradox detection
- **L2 Intervention** (`causal_tools/l2_intervention.py`): Backdoor/frontdoor adjustment, IV estimation, propensity matching
- **L3 Counterfactual** (`causal_tools/l3_counterfactual.py`): SCM-based inference, sensitivity analysis, probability of necessity/sufficiency
- **Meta** (`causal_tools/meta_tools.py`): Tool selection and registry

### Key Design Decisions
- **Capability gradient**: Smaller model for creative deception (7B), larger for rigorous detection (14B) and auditing (72B)
- **Dynamic difficulty**: `game/difficulty.py` targets ~0.4 deception success rate via Flow theory
- **Strategy evolution**: `game/evolution.py` feeds round summaries back for adversarial co-evolution
- **LLM-first agents**: `Agent A/B/C` now ask the LLM for a structured JSON decision first, then validate/merge with tool evidence and fallback logic. The core target is no longer “rules first, LLM only adds narration.”
- **Tool-backed guardrails**: `Agent B` and `Agent C` still run the Pearl toolchain and use its outputs as constraints, evidence, and fallback when the model response is missing or malformed.
- **Mock-friendly engine**: `game/debate_engine.py` still ships with built-in mock agents, so the full round loop runs offline. Real agents are initialized first; mocks are only used when initialization or runtime fails.
- **LLMService layering**: `game/llm_service.py` is the single backend adapter. `dashscope`/`api` is wired to Alibaba Bailian's OpenAI-compatible endpoint (default for Qwen2.5-7B/14B/72B). `vllm` and `ollama` remain placeholders and currently degrade to mock. `LLMService.generate_json(...)` is the key entrypoint for structured decisions.
- **On-disk tracking**: `evaluation/tracker.py` writes `logs/<run_id>/` with `config.json`, `metrics.jsonl`, `rounds.jsonl`, and `artifacts/`. `main.py`, exp1, and exp4 all route results through it.

## Configuration

All game parameters in `configs/default.yaml`: model configs, debate rounds, causal levels, difficulty tuning, evaluation weights, and visualization settings.

## Development Status

Runnable end-to-end. `main.py` plus `exp1/exp2/exp3/exp4` all execute full rounds (data generation → debate → jury → audit → evolution → tracking), and the visualization stack can consume a live event stream via `run_live_game.py`. Real agents (A/B/C) and the Pearl toolchain are wired in, and the default configuration targets Qwen2.5-7B/14B/72B on DashScope through the built-in `LLMService`.

Current practical status:

- The framework, experiment entrypoints, tracking, and frontend are all runnable.
- The decision flow is now **LLM-first**, but `Agent B/C` plus the jury still form a strong “audit coalition,” so `Agent A` is currently underpowered in real runs.
- The next tuning target is not infrastructure, but game balance: improving `Agent A` deception quality and/or reducing `Agent C` / jury bias so `DSR` moves closer to the intended range.

## Key Dependencies

Core causal stack: DoWhy, CausalML, causal-learn, EconML. LLM access: `openai` SDK against DashScope's OpenAI-compatible endpoint (local vllm + transformers kept as future options). Backend: FastAPI + WebSocket. Experiment tracking: W&B. See `requirements.txt` for full list.
