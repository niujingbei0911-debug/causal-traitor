# Artifact Layout

This file documents the workspace-level organization used for generated artifacts.

## Why This Exists

The repository keeps `outputs/` and `logs/` out of git, which is correct for
large generated files, but it also means those directories can become noisy very
quickly.

To keep the workspace readable without touching code paths, artifacts are grouped
by purpose instead of by script internals.

## Current Convention

### `outputs/mainline/`

Use for paper-facing outputs from the active research track:

- `exp_main_benchmark*`
- `exp_leakage_study*`
- `exp_identifiability_ablation*`
- `exp_ood_generalization*`
- `exp_adversarial_robustness*`
- `exp_cross_model_transfer*`
- `exp_human_audit*`

These correspond to the main experiment family defined in
[FINAL_CONSTRUCTION_BLUEPRINT.md](FINAL_CONSTRUCTION_BLUEPRINT.md).

### `outputs/supplemental/`

Use for appendix/demo-oriented outputs such as:

- `exp1_causal_levels*`
- `exp3_difficulty*`
- `exp4_evolution*`
- `game_run_*`

These files are still useful, but they are not the main paper artifact set.

### `outputs/review/`

Use for scratch and verification-style material, including files with prefixes such
as:

- `_review_*`
- `_verify_*`
- `_audit_*`
- `_bench_*`
- `_cfg_*`
- `_diag_*`
- `_dup_*`
- `_final_*`
- `_tmp_*`
- `tmp_*`
- `__tmp_*`

This bucket is intentionally broad. Anything that is exploratory, diagnostic, or
generated during review should live here instead of cluttering `outputs/` root.

### `logs/supplemental_runs/`

Use for archived runtime tracker directories from legacy/demo runs, including:

- `main_run_*`
- `exp1_*`
- `exp2_*`
- `exp3_*`
- `exp4_*`

## Important Constraint

No code paths were changed to enforce this structure.

That means:

- fresh runs may still write directly to `outputs/` root
- fresh runtime logs may still appear directly under `logs/`

This layout is therefore a **cleanup convention**, not a hard runtime contract.

## Recommended Usage

If you are producing paper-facing results, prefer either:

1. running scripts with explicit `--output` paths under `outputs/mainline/`, or
2. periodically moving newly generated files into the correct subdirectory

If you are doing debugging or review passes, write directly into
`outputs/review/` when practical.
