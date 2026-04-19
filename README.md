# The Causal Traitor v3

> Current canonical research direction: **Adversarial Causal Oversight under Information Asymmetry**
>
> Repository note: the project has fully transitioned to the paper-oriented plan defined in
> [docs/FINAL_CONSTRUCTION_BLUEPRINT.md](docs/FINAL_CONSTRUCTION_BLUEPRINT.md) and
> [docs/AGENT_EXECUTION_MANUAL.md](docs/AGENT_EXECUTION_MANUAL.md).
> Older multi-agent game materials are retained only for appendix/demo support and
> historical traceability.

## Project Definition

This repository now centers on three paper-level deliverables:

1. `Adversarial Causal Oversight`
   A task where a verifier judges whether a natural-language causal claim is
   `valid`, `invalid`, or `unidentifiable` under information asymmetry.
2. `Leakage-free benchmark`
   Programmatically generated benchmark instances with strict public/gold
   separation, witness annotations, and IID/OOD splits.
3. `Countermodel-Grounded Verification`
   A verifier pipeline built around claim parsing, assumption ledger construction,
   countermodel search, and tool-backed adjudication.

The older "multi-agent deception game" framing is no longer the primary research
story of the repository. It remains only as supplemental demo / appendix
infrastructure.

## Canonical Docs

Read these first:

- [docs/FINAL_CONSTRUCTION_BLUEPRINT.md](docs/FINAL_CONSTRUCTION_BLUEPRINT.md)
  Final paper blueprint, task definition, benchmark scope, verifier design, and
  experiment matrix.
- [docs/AGENT_EXECUTION_MANUAL.md](docs/AGENT_EXECUTION_MANUAL.md)
  Engineering execution handbook aligned to the blueprint phases.
- [docs/PROJECT_MAP.md](docs/PROJECT_MAP.md)
  Repository navigation guide: what is mainline, what is appendix/demo, and what
  is historical archive.
- [docs/ARTIFACT_LAYOUT.md](docs/ARTIFACT_LAYOUT.md)
  Output and log organization guide for experiment artifacts, review snapshots,
  and supplemental runtime traces.
- [docs/COURSE_REPO_WALKTHROUGH.md](docs/COURSE_REPO_WALKTHROUGH.md)
  Recommended repository reading and explanation order for course presentation,
  advisor discussion, and defense-style walkthroughs.

Historical materials:

- [docs/legacy/README.md](docs/legacy/README.md)
  Index for superseded planning, design, review, and progress documents.

## Repository Layout

### Main Paper Assets

- `benchmark/`
  Benchmark schema, graph families, claim generation, witness generation, split
  building, and loading.
- `verifier/`
  Claim parser, assumption ledger, countermodel search, decision rule, and unified
  pipeline.
- `evaluation/`
  Verdict-centric metrics, scoring, reporting, and statistical significance
  helpers.
- `experiments/exp_main_benchmark/`
  Main benchmark experiment.
- `experiments/exp_adversarial_robustness/`
  Attack-strength robustness evaluation.
- `experiments/exp_identifiability_ablation/`
  Component ablations for ledger, countermodel, abstention, and tools.
- `experiments/exp_leakage_study/`
  Clean vs oracle-leaking partition comparison.
- `experiments/exp_ood_generalization/`
  OOD evaluation across family, lexical, and variable-renaming shifts.
- `experiments/exp_cross_model_transfer/`
  Current surrogate transfer study.
- `experiments/exp_human_audit/`
  Human-audit package generation and agreement interface.
- `experiments/README.md`
  Experiment-layer guide that separates paper-facing runs from appendix/demo runs.

### Supplemental / Appendix / Demo Assets

- `agents/`
  Retained runtime shell around attacker / baseline / verifier roles.
- `game/`
  Debate engine and demo orchestration. Some modules are mainline-reused
  infrastructure, while `difficulty.py` and `evolution.py` are appendix/demo
  oriented.
- `visualization/`
  Frontend and API for live demo visualization.
- `run_live_game.py`
  Supplemental live-demo entrypoint.
- `experiments/exp1_causal_levels/`
  Legacy benchmark-style causal-level experiment.
- `experiments/exp2_jury_ablation/`
  Appendix jury study.
- `experiments/exp3_difficulty/`
  Appendix difficulty-control study.
- `experiments/exp4_evolution/`
  Appendix evolution study.

### Historical Archive

- `docs/legacy/`
  Superseded design documents, course-project planning notes, review reports, and
  progress reports kept only for reference.

## Current Implementation Status

The repository already contains substantial implementation aligned with the new
plan:

- information-partitioned benchmark schema in `benchmark/schema.py`
- programmatic graph families and attack templates in `benchmark/`
- verifier-first pipeline in `verifier/`
- verdict-centric evaluation stack in `evaluation/`
- Phase-4-style experiment runners in `experiments/`
- backward-compatible Agent C wrapper and public-only tool execution

The repository still also contains:

- legacy README-era multi-agent game framing
- retained appendix/demo experiments
- historical documents that no longer define the active project

Those materials are intentionally preserved, but they are not the canonical basis
for paper writing or project scoping.

## Quick Start

### Environment

- Python 3.10+
- Node.js 18+ for frontend visualization only
- Optional DashScope / OpenAI-compatible API keys for LLM-backed runtime paths

### Install

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Optional frontend install:

```bash
cd visualization/frontend
npm install
cd ../..
```

### Core Verification

Run the main implementation-focused tests:

```bash
pytest tests/test_information_partition.py tests/test_benchmark.py tests/test_verifier.py tests/test_evaluation.py -q
```

Run the full test suite:

```bash
pytest tests/ -v
```

### Mainline Experiments

Main benchmark:

```bash
python -m experiments.exp_main_benchmark.run --samples-per-family 10
```

Leakage study:

```bash
python -m experiments.exp_leakage_study.run --samples-per-family 10
```

Identifiability ablation:

```bash
python -m experiments.exp_identifiability_ablation.run --samples-per-family 10
```

OOD generalization:

```bash
python -m experiments.exp_ood_generalization.run --samples-per-family 10
```

Human-audit package:

```bash
python -m experiments.exp_human_audit.run --samples-per-family 15
```

### Supplemental Demo Entrypoints

Legacy / demo game run:

```bash
python main.py --rounds 3 --output outputs/run.json
```

Live visualization stream:

```bash
python run_live_game.py --rounds 6 --delay 1.0 --ws ws://127.0.0.1:8001/ws/game
```

These commands are retained for supplemental demonstration value and backward
compatibility. They are not the repository's primary research entrypoints.

## Outputs

- `outputs/`
  Generated artifacts. After cleanup, it is organized into:
  - `outputs/mainline/` for paper-facing main experiments
  - `outputs/supplemental/` for legacy appendix/demo outputs
  - `outputs/review/` for scratch, audit, verification, and review snapshots
- `logs/`
  Runtime tracker artifacts. Current archived runtime/demo logs live under
  `logs/supplemental_runs/`.

Fresh runs may still write into `outputs/` or `logs/` root unless an explicit path
is supplied. See [docs/ARTIFACT_LAYOUT.md](docs/ARTIFACT_LAYOUT.md) for the cleanup
convention used in this workspace.

## License

MIT License
