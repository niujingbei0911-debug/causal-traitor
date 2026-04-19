# Project Map

This file is the shortest reliable guide to the current repository layout.

## Canonical Baseline

Treat the following two documents as the active project definition:

- [FINAL_CONSTRUCTION_BLUEPRINT.md](FINAL_CONSTRUCTION_BLUEPRINT.md)
- [AGENT_EXECUTION_MANUAL.md](AGENT_EXECUTION_MANUAL.md)
- [ARTIFACT_LAYOUT.md](ARTIFACT_LAYOUT.md)
- [COURSE_REPO_WALKTHROUGH.md](COURSE_REPO_WALKTHROUGH.md)

If any older document conflicts with them, the newer mainline documents win.

## What The Project Is Now

The repository's active research goal is:

> a new task + a leakage-free benchmark + a countermodel-grounded verifier
> for adversarial causal claim oversight under information asymmetry

That means the primary story is no longer:

> a large multi-agent causal deception game with jury, difficulty, and evolution

Those older system elements are preserved, but they are now secondary.

## Mainline Assets

These are the directories and files that directly support the paper's main thesis.

### Task / Benchmark

- [schema.py](C:/Users/njb18/Desktop/causal-traitor/benchmark/schema.py)
- [generator.py](C:/Users/njb18/Desktop/causal-traitor/benchmark/generator.py)
- [graph_families.py](C:/Users/njb18/Desktop/causal-traitor/benchmark/graph_families.py)
- [attacks.py](C:/Users/njb18/Desktop/causal-traitor/benchmark/attacks.py)
- [witnesses.py](C:/Users/njb18/Desktop/causal-traitor/benchmark/witnesses.py)
- [split_builder.py](C:/Users/njb18/Desktop/causal-traitor/benchmark/split_builder.py)

### Verifier

- [claim_parser.py](C:/Users/njb18/Desktop/causal-traitor/verifier/claim_parser.py)
- [assumption_ledger.py](C:/Users/njb18/Desktop/causal-traitor/verifier/assumption_ledger.py)
- [countermodel_search.py](C:/Users/njb18/Desktop/causal-traitor/verifier/countermodel_search.py)
- [decision.py](C:/Users/njb18/Desktop/causal-traitor/verifier/decision.py)
- [pipeline.py](C:/Users/njb18/Desktop/causal-traitor/verifier/pipeline.py)

### Evaluation / Main Experiments

- [metrics.py](C:/Users/njb18/Desktop/causal-traitor/evaluation/metrics.py)
- [scorer.py](C:/Users/njb18/Desktop/causal-traitor/evaluation/scorer.py)
- [reporting.py](C:/Users/njb18/Desktop/causal-traitor/evaluation/reporting.py)
- [significance.py](C:/Users/njb18/Desktop/causal-traitor/evaluation/significance.py)
- [experiments/README.md](C:/Users/njb18/Desktop/causal-traitor/experiments/README.md)
- [benchmark_harness.py](C:/Users/njb18/Desktop/causal-traitor/experiments/benchmark_harness.py)
- [exp_main_benchmark/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp_main_benchmark/run.py)
- [exp_adversarial_robustness/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp_adversarial_robustness/run.py)
- [exp_identifiability_ablation/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp_identifiability_ablation/run.py)
- [exp_leakage_study/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp_leakage_study/run.py)
- [exp_ood_generalization/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp_ood_generalization/run.py)
- [exp_cross_model_transfer/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp_cross_model_transfer/run.py)
- [exp_human_audit/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp_human_audit/run.py)

## Appendix / Demo Assets

These remain useful, but they are not the main paper claim.

- [jury.py](C:/Users/njb18/Desktop/causal-traitor/agents/jury.py)
- [difficulty.py](C:/Users/njb18/Desktop/causal-traitor/game/difficulty.py)
- [evolution.py](C:/Users/njb18/Desktop/causal-traitor/game/evolution.py)
- [api.py](C:/Users/njb18/Desktop/causal-traitor/visualization/api.py)
- [App.tsx](C:/Users/njb18/Desktop/causal-traitor/visualization/frontend/src/App.tsx)
- [run_live_game.py](C:/Users/njb18/Desktop/causal-traitor/run_live_game.py)
- [main.py](C:/Users/njb18/Desktop/causal-traitor/main.py)
- [exp1_causal_levels/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp1_causal_levels/run.py)
- [exp2_jury_ablation/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp2_jury_ablation/run.py)
- [exp3_difficulty/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp3_difficulty/run.py)
- [exp4_evolution/run.py](C:/Users/njb18/Desktop/causal-traitor/experiments/exp4_evolution/run.py)

## Historical Archive

Superseded design and planning materials live under [legacy/](legacy).

Use them only when you need:

- historical context
- appendix writeup background
- evidence of project evolution
- references for earlier system framing

Do not use them as the current source of truth for project scope.

## Practical Reading Order

If you want to understand the active project without getting confused:

1. [FINAL_CONSTRUCTION_BLUEPRINT.md](FINAL_CONSTRUCTION_BLUEPRINT.md)
2. [AGENT_EXECUTION_MANUAL.md](AGENT_EXECUTION_MANUAL.md)
3. This file
4. `benchmark/`
5. `verifier/`
6. `evaluation/`
7. `experiments/benchmark_harness.py`

Only after that should you look at:

- `agents/`
- `game/`
- `visualization/`
- `docs/legacy/`

## Generated Artifacts

Workspace-generated artifacts are now organized with a lightweight convention:

- `outputs/mainline/`
  Main paper experiment outputs.
- `outputs/supplemental/`
  Legacy appendix/demo outputs and game-run dumps.
- `outputs/review/`
  Scratch, audit, verification, and review snapshots.
- `logs/supplemental_runs/`
  Historical runtime logs for legacy/demo runs.

The exact convention is documented in [ARTIFACT_LAYOUT.md](ARTIFACT_LAYOUT.md).
