"""Run the configured strong LLM baseline matrix.

This runner reads ``configs/llm_baseline_matrix.yaml`` and executes the same
public-view API baseline prompt for each enabled model, seed, and split.
Secrets stay in environment variables or ``.env``; the config stores only env
var names.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import yaml

from experiments.exp_api_baseline_smoke.run import run_api_baseline_smoke


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
                    )
                    job["ok"] = True
                    job["summary"] = payload.get("summary", {})
                except Exception as exc:
                    job["ok"] = False
                    job["error_type"] = type(exc).__name__
                    job["error"] = str(exc)
                    jobs.append(job)
                    if not continue_on_error:
                        manifest = _build_manifest(resolved_config_path, output_dir, config, probe, jobs)
                        _write_manifest(output_dir, manifest)
                        raise
                else:
                    jobs.append(job)

    manifest = _build_manifest(resolved_config_path, output_dir, config, probe, jobs)
    return _write_manifest(output_dir, manifest)


def _build_manifest(
    config_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    probe: bool,
    jobs: list[dict[str, Any]],
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
        "generated_at_utc": _now_utc(),
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
    }


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
