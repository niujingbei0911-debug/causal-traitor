# Phase 4 Real-Grounded And Witness Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the Phase 4 real-grounded subset and witness faithfulness runners so they satisfy the documented contracts, consume grounded inputs, and emit reproducible artifacts with CI and significance scaffolding.

**Architecture:** Remove runner-local toy data generation, make the real-grounded loader consume explicit observed data from dataset payloads or file paths, and report synthetic versus real-grounded results on distinct evaluation scopes. Rebuild witness faithfulness as an evidence-replay experiment that reruns the verifier decision logic under ablated, corrupted, and shuffled evidence bundles instead of post-hoc verdict JSON edits.

**Tech Stack:** Python, pandas, unittest, existing benchmark harness utilities, verifier decision pipeline, bootstrap/significance helpers.

---

### Task 1: Lock Failing Tests For Real-Grounded Loader And Runner

**Files:**
- Modify: `tests/test_real_grounded_subset.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Write failing loader tests for explicit observed data and missing-data rejection**

Add tests that:
- provide case-level observed data records and assert `load_real_grounded_samples(...)` preserves them exactly
- assert the loader raises when no observed-data source is supplied
- assert the default blueprint no longer forces a synthetic treatment-to-outcome edge

- [ ] **Step 2: Run the focused loader tests and verify they fail for the current synthetic loader**

Run: `python -m pytest tests/test_real_grounded_subset.py -q`
Expected: FAIL in the new explicit-observed-data assertions and missing-data rejection assertions.

- [ ] **Step 3: Write failing integration assertions for the real-grounded runner contract**

Add integration assertions that:
- the runner accepts a dataset path
- the real-grounded partition reports a single `real_grounded` evaluation scope instead of fake `test_iid/test_ood`
- real-grounded raw predictions carry `split == "real_grounded"`

- [ ] **Step 4: Run the focused integration tests and verify they fail**

Run: `python -m pytest tests/test_integration.py -q -k real_grounded`
Expected: FAIL in the new scope and dataset assertions.

### Task 2: Lock Failing Tests For Witness Necessity Replay

**Files:**
- Modify: `tests/test_verifier.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Write failing tests that require replayed condition decisions**

Add tests that:
- build a deterministic benchmark sample with a witness
- run the witness-faithfulness helper
- assert conditioned records include replayed verdicts, non-trivial `verdict_changed`, and condition-specific predictions instead of hardcoded `False`

- [ ] **Step 2: Run the focused witness unit tests and verify they fail**

Run: `python -m pytest tests/test_verifier.py -q -k witness`
Expected: FAIL because the current implementation only edits verdict dictionaries and hardcodes `verdict_changed = False`.

- [ ] **Step 3: Write failing integration assertions for witness artifacts**

Add integration assertions that:
- aggregated witness metrics contain CI-style summary dictionaries
- `significance` is always emitted
- the CI artifact is non-empty

- [ ] **Step 4: Run the focused integration tests and verify they fail**

Run: `python -m pytest tests/test_integration.py -q -k witness`
Expected: FAIL because the current runner emits plain means and omits significance.

### Task 3: Implement Real-Grounded Data Source And Reporting Repair

**Files:**
- Modify: `benchmark/loaders.py`
- Modify: `experiments/exp_real_grounded_subset/run.py`
- Create or modify fixture data as needed under `tests/fixtures` or `benchmark/data`

- [ ] **Step 1: Make the loader require explicit observed-data sources**

Implement loading priority:
- inline observed-data records from case metadata
- file-backed data from `claim.observed_data_path`
- otherwise raise `ValueError`

- [ ] **Step 2: Remove synthetic table generation and synthetic hidden-variable fabrication**

Replace the monotonic synthetic dataframe and `hidden_context_*` defaults with case-provided metadata or safe empty defaults.

- [ ] **Step 3: Stop forcing a fixed treatment-to-outcome DAG**

Use case metadata when available; otherwise build a minimal node-only DAG that preserves variables without inventing edges.

- [ ] **Step 4: Add dataset-path support to the runner**

Extend `run_experiment(...)` and CLI parsing with a dataset input argument and route it through `load_real_grounded_dataset(...)`.

- [ ] **Step 5: Report synthetic and real-grounded on honest scopes**

Keep synthetic on `test_iid/test_ood`, but evaluate the real-grounded partition once per system/seed on a single `real_grounded` scope.

- [ ] **Step 6: Remove dead temporary-directory cleanup**

Delete the unused temp-directory path and any dead cleanup code.

### Task 4: Implement Witness Faithfulness Replay Repair

**Files:**
- Modify: `experiments/exp_witness_faithfulness/run.py`

- [ ] **Step 1: Build reusable evidence bundles per sample**

Capture parsed claim, ledger, countermodel-search output, and tool trace needed to replay `decide_verdict(...)`.

- [ ] **Step 2: Implement replayed `original/drop/corrupt/shuffle` conditions**

For each condition, perturb the evidence bundle instead of editing the final verdict payload:
- `drop_witness`: remove the generating evidence channel
- `corrupt_witness`: mutate the evidence channel so it drives a degraded or contradictory replay
- `shuffle_witness`: replay with a donor bundle from another witness-bearing sample

- [ ] **Step 3: Score replayed predictions with real outcome metrics**

Record per-condition predictions, `verdict_changed`, and explanation-grounding metrics, then aggregate them per seed with `score_prediction_records(...)`.

- [ ] **Step 4: Emit CI and significance artifacts**

Summarize condition metrics with bootstrap CI and compare each condition against `original` with paired seed bootstrap reports.

- [ ] **Step 5: Rewrite markdown summary and conclusion logic**

Base conclusions on actual replay results and significance summaries instead of hardcoded heuristic drops.

### Task 5: Verify End To End

**Files:**
- No new files required

- [ ] **Step 1: Run focused unit and integration tests**

Run:
- `python -m pytest tests/test_real_grounded_subset.py -q`
- `python -m pytest tests/test_verifier.py -q -k witness`
- `python -m pytest tests/test_integration.py -q -k "real_grounded or witness"`

- [ ] **Step 2: Run the repaired experiment smoke tests**

Run:
- `python -m experiments.exp_real_grounded_subset.run --allow-protocol-violations --seeds 0 --samples-per-family 1`
- `python -m experiments.exp_witness_faithfulness.run --allow-protocol-violations --seeds 0 --samples-per-family 1`

- [ ] **Step 3: Confirm emitted artifacts are non-empty and contract-complete**

Check that each runner writes:
- config
- seed list
- raw predictions
- aggregated metrics
- CI
- significance
- markdown summary
