# Experiments Map

This directory contains two distinct experiment layers:

1. **Mainline paper experiments**
   These are the experiments that support the active research thesis defined in
   `docs/FINAL_CONSTRUCTION_BLUEPRINT.md`.
2. **Supplemental / appendix / demo experiments**
   These are retained from the earlier system framing and are still useful for
   demo, appendix analysis, and historical comparison, but they are not the
   primary paper evidence.

## Shared Infrastructure

- `benchmark_harness.py`
  Shared helper layer for Phase-4-style benchmark runs, scoring, significance,
  and artifact writing.

## Mainline Paper Experiments

- `exp_main_benchmark/`
  Core benchmark comparison across main verifier variants.
- `exp_adversarial_robustness/`
  Attack-strength robustness study.
- `exp_identifiability_ablation/`
  Ablation of ledger, countermodel, abstention, and tools.
- `exp_leakage_study/`
  Clean public partition vs oracle-leaking partition.
- `exp_ood_generalization/`
  OOD behavior across family, lexical, and variable-renaming shifts.
- `exp_cross_model_transfer/`
  Current surrogate transfer study.
  Important: this is not yet the full blueprint cross-model-family experiment.
- `exp_human_audit/`
  Human-audit package generation and agreement interface.

## Supplemental / Appendix / Demo Experiments

- `exp1_causal_levels/`
  Legacy causal-level comparison.
- `exp2_jury_ablation/`
  Jury mechanism study.
- `exp3_difficulty/`
  Dynamic-difficulty study.
- `exp4_evolution/`
  Evolution / arms-race study.

## Recommended Reading Order

If you are reading this directory for the active project:

1. `benchmark_harness.py`
2. `exp_main_benchmark/run.py`
3. `exp_leakage_study/run.py`
4. `exp_identifiability_ablation/run.py`
5. `exp_adversarial_robustness/run.py`
6. `exp_ood_generalization/run.py`
7. `exp_human_audit/run.py`
8. `exp_cross_model_transfer/run.py`

Only after that should you read:

- `exp1_causal_levels/`
- `exp2_jury_ablation/`
- `exp3_difficulty/`
- `exp4_evolution/`

## Artifact Convention

Workspace artifact cleanup currently follows:

- `outputs/mainline/` for paper-facing experiment outputs
- `outputs/supplemental/` for appendix/demo outputs
- `outputs/review/` for scratch and review outputs

See `docs/ARTIFACT_LAYOUT.md` for the full convention.
