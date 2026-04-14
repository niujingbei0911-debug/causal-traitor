# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**The Causal Traitor (因果叛徒)** — A multi-agent adversarial debate system for causal reasoning research. Three LLM agents (Traitor, Scientist, Auditor) + a Jury debate causal claims across Pearl's 3-layer causal hierarchy (Association → Intervention → Counterfactual). The Traitor constructs plausible but flawed causal arguments; the Scientist detects fallacies; the Auditor renders verdicts.

## Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run a single game via the unified entry point
python main.py --rounds 3 --output outputs/run.json

# Run experiments (each writes JSON + CSV + Markdown sidecars)
python -m experiments.exp1_causal_levels.run --rounds-per-level 20
python -m experiments.exp4_evolution.run --rounds 10 --level 2

# Run visualization server (FastAPI + WebSocket)
python -m visualization.api

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
- **Mock-friendly engine**: `game/debate_engine.py` ships with built-in mock agents, so the full round loop runs without any real LLM. Real agents are tried first and fall back to mocks on failure or `NotImplementedError`.
- **LLMService layering**: `game/llm_service.py` is the single backend adapter. `dashscope`/`api` is wired to Alibaba Bailian's OpenAI-compatible endpoint (default for Qwen2.5-7B/14B/72B); `vllm` and `ollama` are declared but fall back to mock until a local server is attached; `mock` is fully offline and deterministic. API key resolution order: explicit `api_key` in config → `DASHSCOPE_API_KEY` / `OPENAI_API_KEY` env var → hardcoded low-cap dev key in `_DEFAULT_DASHSCOPE_API_KEY`. Agents call the service after their deterministic tool-backed reasoning to attach a natural-language extension; `[mock:...]` responses are silently dropped so behavior stays deterministic whenever the real backend is unavailable.
- **On-disk tracking**: `evaluation/tracker.py` writes `logs/<run_id>/` with `config.json`, `metrics.jsonl`, `rounds.jsonl`, and `artifacts/`. `main.py`, exp1, and exp4 all route results through it.

## Configuration

All game parameters in `configs/default.yaml`: model configs, debate rounds, causal levels, difficulty tuning, evaluation weights, and visualization settings.

## Development Status

Runnable end-to-end. `main.py`, exp1, and exp4 all execute full rounds (data generation → debate → evolution → tracking). Real agents (A/B/C) and the Pearl toolchain are wired in, and by default they hit Qwen2.5-7B/14B/72B on DashScope via the built-in LLMService. Tests and any run without the key still work through the mock fallback. The `vllm` / `ollama` backends are placeholders until a local server is attached. See `docs/TASK_ASSIGNMENT.md` for the phased implementation plan and team assignments.

## Key Dependencies

Core causal stack: DoWhy, CausalML, causal-learn, EconML. LLM access: `openai` SDK against DashScope's OpenAI-compatible endpoint (local vllm + transformers kept as future options). Backend: FastAPI + WebSocket. Experiment tracking: W&B. See `requirements.txt` for full list.
