"""Run the configured strong LLM baseline matrix.

This runner reads ``configs/llm_baseline_matrix.yaml`` and executes the same
public-view API baseline prompt for each enabled model, seed, and split.
Secrets stay in environment variables or ``.env``; the config stores only env
var names.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

import yaml

from experiments.exp_api_baseline_smoke.run import run_api_baseline_smoke
from evaluation.metrics import CausalMetrics


DEFAULT_CONFIG = Path("configs/llm_baseline_matrix.yaml")
DEFAULT_MAX_SAMPLES = 1_000_000


RunOne = Callable[..., dict[str, Any]]


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_config(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Matrix config must be a YAML mapping: {config_path}")
    if not isinstance(payload.get("run"), dict):
        raise ValueError("Matrix config requires a 'run' mapping.")
    if not isinstance(payload.get("models"), list):
        raise ValueError("Matrix config requires a 'models' list.")
    return payload


def _enabled_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    models = [dict(model) for model in config.get("models", []) if model.get("enabled", True)]
    if not models:
        raise ValueError("Matrix config has no enabled models.")
    seen: set[str] = set()
    for model in models:
        model_id = str(model.get("id") or "").strip()
        if not model_id:
            raise ValueError("Each enabled model needs a non-empty 'id'.")
        if model_id in seen:
            raise ValueError(f"Duplicate enabled model id: {model_id}")
        seen.add(model_id)
        if not model.get("backend") or not model.get("model"):
            raise ValueError(f"Model {model_id!r} needs both 'backend' and 'model'.")
    return models


def _resolve_output_dir(config_path: Path, run_cfg: dict[str, Any], *, probe: bool) -> Path:
    output_dir = Path(str(run_cfg.get("output_dir") or "outputs/llm_baseline_matrix"))
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    return output_dir / "probe" if probe else output_dir


def run_llm_baseline_matrix(
    *,
    config_path: Path | str = DEFAULT_CONFIG,
    probe: bool = False,
    continue_on_error: bool = False,
    run_one: RunOne = run_api_baseline_smoke,
) -> dict[str, Any]:
    """Execute enabled model/seed/split jobs and write a manifest."""

    resolved_config_path = Path(config_path)
    config = _read_config(resolved_config_path)
    run_cfg = dict(config["run"])
    models = _enabled_models(config)
    output_dir = _resolve_output_dir(resolved_config_path, run_cfg, probe=probe)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(seed) for seed in run_cfg.get("seeds", [0])]
    splits = [str(split) for split in run_cfg.get("splits", ["test_iid"])]
    if probe:
        seeds = seeds[:1]
        splits = splits[:1]

    samples_per_family = 1 if probe else int(run_cfg.get("samples_per_family", 10))
    max_samples = 1 if probe else int(run_cfg.get("max_samples", DEFAULT_MAX_SAMPLES))
    jobs: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    generated_at_utc = _now_utc()

    for model in models:
        model_id = str(model["id"])
        temperature = float(model.get("temperature", run_cfg.get("temperature", 0.0)))
        max_tokens = int(model.get("max_tokens", run_cfg.get("max_tokens", 512)))
        timeout = float(model.get("timeout", run_cfg.get("timeout", 90)))
        for seed in seeds:
            for split_name in splits:
                output_path = output_dir / f"{model_id}_seed{seed}_{split_name}.json"
                job: dict[str, Any] = {
                    "model_id": model_id,
                    "backend": model.get("backend"),
                    "model": model.get("model"),
                    "api_mode": model.get("api_mode", "chat_completions"),
                    "reasoning_effort": model.get("reasoning_effort"),
                    "thinking": model.get("thinking"),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": timeout,
                    "seed": seed,
                    "split": split_name,
                    "output_path": str(output_path),
                }
                try:
                    payload = run_one(
                        output_path=output_path,
                        model=model.get("model"),
                        backend=model.get("backend"),
                        base_url=model.get("base_url"),
                        api_key_env=model.get("api_key_env"),
                        api_mode=model.get("api_mode") or "chat_completions",
                        reasoning_effort=model.get("reasoning_effort"),
                        thinking=model.get("thinking"),
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout,
                        seed=seed,
                        split_name=split_name,
                        max_samples=max_samples,
                        samples_per_family=samples_per_family,
                        difficulty=float(run_cfg.get("difficulty", 0.55)),
                        max_public_rows=int(run_cfg.get("max_public_rows", 5)),
                        reject_mock_fallback=bool(run_cfg.get("reject_mock_fallback", True)),
                        run_status="llm_baseline_matrix_probe_job" if probe else "llm_baseline_matrix_job",
                        run_note=(
                            "Probe-only API-backed public-view LLM baseline matrix component. "
                            "Do not cite as the full baseline matrix."
                            if probe
                            else (
                                "API-backed public-view LLM baseline matrix component. "
                                "Cite via the matrix manifest and aggregate artifacts."
                            )
                        ),
                    )
                    job["ok"] = True
                    job["summary"] = payload.get("summary", {})
                    raw_records.extend(_enrich_records(payload.get("records", []), job))
                except Exception as exc:
                    job["ok"] = False
                    job["error_type"] = type(exc).__name__
                    job["error"] = str(exc)
                    jobs.append(job)
                    if not continue_on_error:
                        artifacts = _write_matrix_artifacts(
                            output_dir=output_dir,
                            config=config,
                            probe=probe,
                            jobs=jobs,
                            records=raw_records,
                            generated_at_utc=generated_at_utc,
                        )
                        manifest = _build_manifest(
                            resolved_config_path,
                            output_dir,
                            config,
                            probe,
                            jobs,
                            generated_at_utc=generated_at_utc,
                            artifacts=artifacts,
                        )
                        _write_manifest(output_dir, manifest)
                        raise
                else:
                    jobs.append(job)

    artifacts = _write_matrix_artifacts(
        output_dir=output_dir,
        config=config,
        probe=probe,
        jobs=jobs,
        records=raw_records,
        generated_at_utc=generated_at_utc,
    )
    manifest = _build_manifest(
        resolved_config_path,
        output_dir,
        config,
        probe,
        jobs,
        generated_at_utc=generated_at_utc,
        artifacts=artifacts,
    )
    return _write_manifest(output_dir, manifest)


def _build_manifest(
    config_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    probe: bool,
    jobs: list[dict[str, Any]],
    *,
    generated_at_utc: str | None = None,
    artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    succeeded = sum(1 for job in jobs if job.get("ok"))
    failed = len(jobs) - succeeded
    total_predictions = sum(int((job.get("summary") or {}).get("total", 0)) for job in jobs if job.get("ok"))
    fallback_records = sum(
        int((job.get("summary") or {}).get("fallback_records", 0)) for job in jobs if job.get("ok")
    )
    parse_errors = sum(int((job.get("summary") or {}).get("parse_errors", 0)) for job in jobs if job.get("ok"))
    return {
        "status": "llm_baseline_matrix_probe" if probe else "llm_baseline_matrix",
        "generated_at_utc": generated_at_utc or _now_utc(),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "probe": bool(probe),
        "summary": {
            "jobs_total": len(jobs),
            "succeeded": succeeded,
            "failed": failed,
            "total_predictions": total_predictions,
            "fallback_records": fallback_records,
            "parse_errors": parse_errors,
        },
        "config_snapshot": config,
        "jobs": jobs,
        "artifacts": artifacts or {},
    }


def _enrich_records(records: Any, job: dict[str, Any]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    if not isinstance(records, list):
        return enriched
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        payload = dict(record)
        payload["matrix_model_id"] = job.get("model_id")
        payload["matrix_backend"] = job.get("backend")
        payload["matrix_model"] = job.get("model")
        payload["matrix_api_mode"] = job.get("api_mode")
        payload["matrix_reasoning_effort"] = job.get("reasoning_effort")
        payload["matrix_thinking"] = job.get("thinking")
        payload["matrix_temperature"] = job.get("temperature")
        payload["matrix_max_tokens"] = job.get("max_tokens")
        payload["matrix_timeout"] = job.get("timeout")
        payload["matrix_seed"] = job.get("seed")
        payload["matrix_split"] = job.get("split")
        payload["matrix_record_index"] = index
        enriched.append(payload)
    return enriched


def _write_matrix_artifacts(
    *,
    output_dir: Path,
    config: dict[str, Any],
    probe: bool,
    jobs: list[dict[str, Any]],
    records: list[dict[str, Any]],
    generated_at_utc: str,
) -> dict[str, str]:
    aggregate = _aggregate_records(
        config=config,
        probe=probe,
        jobs=jobs,
        records=records,
        generated_at_utc=generated_at_utc,
    )
    aggregate_path = output_dir / "llm_baseline_aggregated_metrics.json"
    csv_path = output_dir / "llm_baseline_summary.csv"
    raw_path = output_dir / "llm_baseline_raw_predictions.jsonl"
    markdown_path = output_dir / "llm_baseline_summary.md"

    aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(csv_path, aggregate)
    _write_raw_predictions(raw_path, records)
    markdown_path.write_text(_render_markdown_summary(aggregate), encoding="utf-8")
    return {
        "aggregated_metrics": str(aggregate_path),
        "summary_csv": str(csv_path),
        "raw_predictions": str(raw_path),
        "summary_markdown": str(markdown_path),
    }


def _aggregate_records(
    *,
    config: dict[str, Any],
    probe: bool,
    jobs: list[dict[str, Any]],
    records: list[dict[str, Any]],
    generated_at_utc: str,
) -> dict[str, Any]:
    by_model_split_seed: dict[str, dict[str, dict[int, list[dict[str, Any]]]]] = {}
    for record in records:
        model_id = str(record.get("matrix_model_id") or record.get("model_id") or "unknown_model")
        split_name = str(record.get("split") or record.get("matrix_split") or "unknown_split")
        seed = int(record.get("seed", record.get("matrix_seed", 0)))
        by_model_split_seed.setdefault(model_id, {}).setdefault(split_name, {}).setdefault(seed, []).append(record)

    models_payload: dict[str, Any] = {}
    for model_id in sorted(by_model_split_seed):
        models_payload[model_id] = {}
        for split_name in sorted(by_model_split_seed[model_id]):
            seed_payloads: dict[str, Any] = {}
            all_records: list[dict[str, Any]] = []
            for seed in sorted(by_model_split_seed[model_id][split_name]):
                seed_records = by_model_split_seed[model_id][split_name][seed]
                seed_payload = _record_metrics(seed_records)
                seed_payloads[str(seed)] = seed_payload
                all_records.extend(seed_records)
            metric_names = (
                "accuracy",
                "macro_f1",
                "unsafe_acceptance_rate",
                "wise_refusal_recall",
                "wise_refusal_precision",
                "over_refusal_rate",
                "parse_error_rate",
                "fallback_rate",
            )
            aggregate_metrics = _record_metrics(all_records)
            for metric_name in metric_names:
                values = [float(seed_payloads[str(seed)][metric_name]) for seed in sorted(by_model_split_seed[model_id][split_name])]
                aggregate_metrics[metric_name] = {
                    "mean": round(float(mean(values)), 4) if values else 0.0,
                    "std": round(float(pstdev(values)), 4) if len(values) > 1 else 0.0,
                    "n_seeds": len(values),
                    "seed_values": values,
                }
            aggregate_metrics["seeds"] = seed_payloads
            models_payload[model_id][split_name] = aggregate_metrics

    return {
        "status": "llm_baseline_matrix_probe_aggregate" if probe else "llm_baseline_matrix_aggregate",
        "generated_at_utc": generated_at_utc,
        "probe": bool(probe),
        "summary": {
            "jobs_total": len(jobs),
            "succeeded": sum(1 for job in jobs if job.get("ok")),
            "failed": sum(1 for job in jobs if not job.get("ok")),
            "total_predictions": len(records),
            "parse_errors": sum(1 for record in records if record.get("parse_error")),
            "fallback_records": sum(1 for record in records if record.get("fallback_detected")),
        },
        "config_snapshot": config,
        "models": models_payload,
    }


def _record_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    gold_labels = [record.get("gold_label") for record in records]
    predicted_labels = [record.get("predicted_label") for record in records]
    total = len(records)
    correct = sum(1 for record in records if bool(record.get("correct")))
    parse_errors = sum(1 for record in records if bool(record.get("parse_error")))
    fallback_records = sum(1 for record in records if bool(record.get("fallback_detected")))
    accuracy = CausalMetrics.verdict_accuracy(gold_labels, predicted_labels).value
    macro_f1 = CausalMetrics.verdict_macro_f1(gold_labels, predicted_labels).value
    unsafe_acceptance_rate = CausalMetrics.unsafe_acceptance_rate(gold_labels, predicted_labels).value
    wise_refusal_recall = CausalMetrics.wise_refusal_recall(gold_labels, predicted_labels).value
    wise_refusal_precision = CausalMetrics.wise_refusal_precision(gold_labels, predicted_labels).value
    over_refusal_rate = CausalMetrics.over_refusal_rate(gold_labels, predicted_labels).value
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "unsafe_acceptance_rate": unsafe_acceptance_rate,
        "wise_refusal_recall": wise_refusal_recall,
        "wise_refusal_precision": wise_refusal_precision,
        "over_refusal_rate": over_refusal_rate,
        "parse_errors": parse_errors,
        "parse_error_rate": round(parse_errors / total, 4) if total else 0.0,
        "fallback_records": fallback_records,
        "fallback_rate": round(fallback_records / total, 4) if total else 0.0,
    }


def _metric_mean(value: Any) -> float:
    if isinstance(value, dict):
        return float(value.get("mean", 0.0))
    return float(value)


def _write_summary_csv(path: Path, aggregate: dict[str, Any]) -> None:
    fieldnames = [
        "model_id",
        "split",
        "seed",
        "total",
        "correct",
        "accuracy",
        "macro_f1",
        "unsafe_acceptance_rate",
        "wise_refusal_recall",
        "wise_refusal_precision",
        "over_refusal_rate",
        "parse_errors",
        "parse_error_rate",
        "fallback_records",
        "fallback_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_id, split_payload in aggregate.get("models", {}).items():
            for split_name, metrics in split_payload.items():
                writer.writerow(_csv_row(model_id, split_name, "ALL", metrics))
                for seed, seed_metrics in metrics.get("seeds", {}).items():
                    writer.writerow(_csv_row(model_id, split_name, seed, seed_metrics))


def _csv_row(model_id: str, split_name: str, seed: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "split": split_name,
        "seed": seed,
        "total": int(metrics.get("total", 0)),
        "correct": int(metrics.get("correct", 0)),
        "accuracy": f"{_metric_mean(metrics.get('accuracy', 0.0)):.4f}",
        "macro_f1": f"{_metric_mean(metrics.get('macro_f1', 0.0)):.4f}",
        "unsafe_acceptance_rate": f"{_metric_mean(metrics.get('unsafe_acceptance_rate', 0.0)):.4f}",
        "wise_refusal_recall": f"{_metric_mean(metrics.get('wise_refusal_recall', 0.0)):.4f}",
        "wise_refusal_precision": f"{_metric_mean(metrics.get('wise_refusal_precision', 0.0)):.4f}",
        "over_refusal_rate": f"{_metric_mean(metrics.get('over_refusal_rate', 0.0)):.4f}",
        "parse_errors": int(metrics.get("parse_errors", 0)),
        "parse_error_rate": f"{_metric_mean(metrics.get('parse_error_rate', 0.0)):.4f}",
        "fallback_records": int(metrics.get("fallback_records", 0)),
        "fallback_rate": f"{_metric_mean(metrics.get('fallback_rate', 0.0)):.4f}",
    }


def _write_raw_predictions(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _render_markdown_summary(aggregate: dict[str, Any]) -> str:
    lines = [
        "# Strong LLM Baseline Matrix",
        "",
        f"- Status: {aggregate['status']}",
        f"- Generated UTC: {aggregate['generated_at_utc']}",
        f"- Total predictions: {aggregate['summary']['total_predictions']}",
        f"- Parse errors: {aggregate['summary']['parse_errors']}",
        f"- Fallback records: {aggregate['summary']['fallback_records']}",
        "",
        "| Model | Split | n | Acc. | Macro-F1 | Unsafe Accept | Wise Refusal | Parse Err. | Fallback |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_id, split_payload in aggregate.get("models", {}).items():
        for split_name, metrics in split_payload.items():
            lines.append(
                "| "
                + " | ".join(
                    (
                        model_id,
                        split_name,
                        str(metrics.get("total", 0)),
                        f"{_metric_mean(metrics.get('accuracy', 0.0)):.4f}",
                        f"{_metric_mean(metrics.get('macro_f1', 0.0)):.4f}",
                        f"{_metric_mean(metrics.get('unsafe_acceptance_rate', 0.0)):.4f}",
                        f"{_metric_mean(metrics.get('wise_refusal_recall', 0.0)):.4f}",
                        f"{_metric_mean(metrics.get('parse_error_rate', 0.0)):.4f}",
                        f"{_metric_mean(metrics.get('fallback_rate', 0.0)):.4f}",
                    )
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def _write_manifest(output_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--probe", action="store_true", help="Run one seed, one split, one sample per enabled model.")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    manifest = run_llm_baseline_matrix(
        config_path=args.config,
        probe=args.probe,
        continue_on_error=args.continue_on_error,
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    print(f"manifest: {manifest['manifest_path']}")


if __name__ == "__main__":
    main()
