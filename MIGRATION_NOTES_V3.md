# Migration to V3: Benchmark-First Research Direction

## What Changed

The project has **transitioned from a multi-agent adversarial debate system to a benchmark-focused research framework**. The core research question is now:

> How do LLM verifiers handle **causal claim auditing under information asymmetry**? Specifically, do they make **unsafe acceptances** when public evidence is insufficient to identify the causal effect?

## What Was Deleted

### Multi-Agent Debate System (Now Archived)
- `agents/agent_a.py` (Traitor) — ❌ Deleted
- `agents/agent_b.py` (Scientist) — ❌ Deleted
- `agents/agent_c.py` (Auditor) — ❌ Deleted
- `agents/jury.py` — ❌ Deleted
- `agents/prompts/` — ❌ Deleted

### Game Engine Infrastructure
- `game/debate_engine.py` — ❌ Deleted
- `game/difficulty.py` — ❌ Deleted
- `game/evolution.py` — ❌ Deleted
- `main.py` — ❌ Deleted
- `run_live_game.py` — ❌ Deleted

### Old Experiments (Multi-Agent Focused)
- `experiments/exp1_causal_levels/` — ❌ Deleted
- `experiments/exp2_jury_ablation/` — ❌ Deleted
- `experiments/exp3_difficulty/` — ❌ Deleted
- `experiments/exp4_evolution/` — ❌ Deleted
- `experiments/exp_adversarial_robustness/` — ❌ Deleted
- `experiments/exp_cross_model_transfer/` — ❌ Deleted
- `experiments/exp_human_audit/` — ❌ Deleted
- `experiments/exp_identifiability_ablation/` — ❌ Deleted
- `experiments/exp_leakage_study/` — ❌ Deleted
- `experiments/exp_main_benchmark/` — ❌ Deleted
- `experiments/exp_ood_generalization/` — ❌ Deleted
- `experiments/exp_persuasion_robustness/` — ❌ Deleted
- `experiments/exp_real_grounded_subset/` — ❌ Deleted
- `experiments/exp_witness_faithfulness/` — ❌ Deleted

### Dependent Tests
- `tests/test_agents.py` — ❌ Deleted
- `tests/test_integration.py` — ❌ Deleted
- `tests/test_tool_executor.py` — ❌ Deleted
- `tests/test_human_audit.py` — ❌ Deleted
- `tests/test_information_partition.py` — ❌ Deleted

## What Was Preserved

### Core Benchmark Infrastructure ✅
- `benchmark/` — Schema, generator, graph families, witnesses, split builder
- `verifier/` — Claim parser, assumption ledger, countermodel search, decision pipeline
- `evaluation/` — Metrics, scoring, reporting, significance testing
- `causal_tools/` — Pearl's causal ladder (L1/L2/L3)

### New Experiments ✅
- `experiments/exp_api_baseline_smoke/` — Quick validation of API
- `experiments/exp_llm_baseline_matrix/` — Core LLM baseline evaluation
- `experiments/paper_artifacts.py` — Unified artifact generation

### Minimal Agent Runtime ✅
- `agents/tool_executor.py` — Tool execution for claims (still used by benchmark)
- `agents/__init__.py` — Namespace

### Game Package (Refactored) ✅
- `game/config.py` — Configuration loading
- `game/llm_service.py` — LLM backend adapter
- `game/data_generator.py` — Synthetic scenario generation
- `game/types.py` — Shared type definitions

## New Entry Points

Instead of `main.py` and game loops, the new workflow is:

```bash
# Run API baseline smoke test
python -m experiments.exp_api_baseline_smoke.run

# Run LLM baseline matrix
python -m experiments.exp_llm_baseline_matrix.run

# Generate paper artifacts
python -m experiments.paper_artifacts generate_all
```

## Key Files to Read

1. **docs/FINAL_RESEARCH_POSITIONING_BENCHMARK_ROUTE_V3.md** — Full V3 blueprint
2. **docs/FINAL_CONSTRUCTION_BLUEPRINT_V2.md** — Paper structure and task definition
3. **README.md** — Updated repository overview
4. **experiments/README.md** — How to run new experiments

## If You Need Old Code

The multi-agent debate system is **not deleted from git history**. You can restore any deleted file with:

```bash
git checkout HEAD~10 -- agents/agent_a.py  # Adjust commit hash as needed
```

The old code remains in git and can be used for reference, demos, or appendix materials, but is not part of the main benchmark paper repository.

---

**Summary**: The project is now **benchmark-first**. Focus on:
- Benchmark construction (benchmark/, verifier/)
- LLM baseline evaluation (exp_llm_baseline_matrix/)
- Metric reporting and artifacts (evaluation/, paper_artifacts.py)

The multi-agent adversarial debate is no longer the core research narrative.
