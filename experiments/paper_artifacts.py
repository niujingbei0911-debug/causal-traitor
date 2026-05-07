"""Build paper-facing tables from frozen experiment artifacts.

This module is intentionally small and file-based: the paper and poster should
consume generated rows from a single artifact bundle instead of hand-copied
numbers in LaTeX or PowerPoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SOURCE_RUNS: dict[str, str] = {
    "main_benchmark": "review_fixed_exp_main_benchmark",
    "identifiability_ablation": "review_fixed_exp_identifiability_ablation",
    "leakage_study": "review_fixed_exp_leakage_study",
    "ood_generalization": "review_fixed_exp_ood_generalization",
    "adversarial_robustness": "review_fixed_exp_adversarial_robustness",
}

SOURCE_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("aggregated_metrics", "_aggregated_metrics.json"),
    ("config", "_config.json"),
    ("seed_list", "_seed_list.json"),
    ("ci", "_ci.json"),
    ("significance", "_significance.json"),
    ("raw_predictions", "_raw_predictions.jsonl"),
    ("json", ".json"),
    ("markdown", ".md"),
)

MAIN_SYSTEM_ORDER: tuple[tuple[str, str], ...] = (
    ("direct_judge", "Direct Judge"),
    ("cot_judge", "CoT Judge"),
    ("tool_baseline", "Tool Baseline"),
    ("debate_baseline", "Debate Baseline"),
    ("refusal_aware_baseline", "Refusal-Aware"),
    ("countermodel_grounded", "Countermodel-Grounded"),
)

ABLATION_ORDER: tuple[tuple[str, str], ...] = (
    ("countermodel_grounded", "Full"),
    ("no_ledger", "No-Ledger"),
    ("no_countermodel", "No-Countermodel"),
    ("no_abstention", "No-Abstention"),
    ("no_tools", "No-Tools"),
)

OOD_ORDER: tuple[tuple[str, str], ...] = (
    ("graph_family_ood", "Graph-Family"),
    ("mechanism_ood", "Mechanism"),
    ("attack_family_ood", "Attack-Family"),
    ("context_shift_ood", "Context-Shift"),
    ("paired_flip_ood", "Paired-Flip"),
)

ADVERSARIAL_ORDER: tuple[tuple[str, str], ...] = (
    ("weak", "Weak"),
    ("medium", "Medium"),
    ("strong", "Strong"),
    ("hidden_information_aware", "Leak-Aware"),
)

SPLITS: tuple[str, ...] = ("test_iid", "test_ood")
ROW_END = r" \tabularnewline"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_file_map(source_dir: Path, stem: str) -> dict[str, Path]:
    files = {name: source_dir / f"{stem}{suffix}" for name, suffix in SOURCE_SUFFIXES}
    missing = [path for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(str(missing[0]))
    return files


def _metric(metrics: dict[str, Any], metric_name: str) -> dict[str, Any]:
    value = metrics[metric_name]
    return {
        "mean": float(value["mean"]),
        "std": float(value.get("std", 0.0)),
        "ci_lower": float(value.get("ci_lower", value["mean"])),
        "ci_upper": float(value.get("ci_upper", value["mean"])),
        "n": int(value.get("n", 0)),
    }


def _round_metric(metrics: dict[str, Any], metric_name: str) -> dict[str, float | int]:
    item = _metric(metrics, metric_name)
    return {
        "mean": round(float(item["mean"]), 3),
        "std": round(float(item["std"]), 3),
        "ci_lower": round(float(item["ci_lower"]), 3),
        "ci_upper": round(float(item["ci_upper"]), 3),
        "n": int(item["n"]),
    }


def _mean_std(metrics: dict[str, Any], metric_name: str) -> str:
    item = _metric(metrics, metric_name)
    return f"${item['mean']:.3f} \\pm {item['std']:.3f}$"


def _mean(metrics: dict[str, Any], metric_name: str) -> str:
    item = _metric(metrics, metric_name)
    return f"${item['mean']:.3f}$"


def _ci(metrics: dict[str, Any], metric_name: str) -> str:
    item = _metric(metrics, metric_name)
    return f"$[{item['ci_lower']:.3f}, {item['ci_upper']:.3f}]$"


def _delta(left: float, right: float) -> str:
    value = right - left
    sign = "+" if value >= 0 else ""
    return f"${sign}{value:.3f}$"


def _split_label(split_name: str) -> str:
    return split_name.replace("_", r"\_")


def _sc(label: str) -> str:
    return rf"\textsc{{{label}}}"


def _render_main_rows(metrics: dict[str, Any]) -> str:
    rows: list[str] = []
    for system_name, label in MAIN_SYSTEM_ORDER:
        for split_name in SPLITS:
            split_metrics = metrics[system_name][split_name]
            rows.append(
                " & ".join(
                    (
                        _sc(label),
                        _split_label(split_name),
                        _mean_std(split_metrics, "verdict_accuracy"),
                        _mean_std(split_metrics, "unsafe_acceptance_rate"),
                        _mean_std(split_metrics, "wise_refusal_recall"),
                    )
                )
                + ROW_END
            )
    return "\n".join(rows) + "\n"


def _render_ablation_rows(metrics: dict[str, Any]) -> str:
    rows: list[str] = []
    for system_name, label in ABLATION_ORDER:
        for split_name in SPLITS:
            split_metrics = metrics[system_name][split_name]
            rows.append(
                " & ".join(
                    (
                        _sc(label),
                        split_name.replace("test_", "").replace("_", r"\_"),
                        _mean(split_metrics, "verdict_accuracy"),
                        _mean(split_metrics, "wise_refusal_recall"),
                    )
                )
                + ROW_END
            )
    return "\n".join(rows) + "\n"


def _render_leakage_rows(metrics: dict[str, Any]) -> str:
    rows: list[str] = []
    for split_name in SPLITS:
        clean = metrics["countermodel_grounded"][split_name]
        leaking = metrics["oracle_leaking_partition"][split_name]
        clean_acc = _metric(clean, "verdict_accuracy")["mean"]
        leak_acc = _metric(leaking, "verdict_accuracy")["mean"]
        rows.append(
            " & ".join(
                (
                    _split_label(split_name),
                    _mean(clean, "verdict_accuracy"),
                    _mean(leaking, "verdict_accuracy"),
                    _delta(clean_acc, leak_acc),
                    _ci(clean, "verdict_accuracy"),
                    _ci(leaking, "verdict_accuracy"),
                )
            )
            + ROW_END
        )
    return "\n".join(rows) + "\n"


def _render_ood_rows(metrics: dict[str, Any]) -> str:
    rows: list[str] = []
    for bucket_name, label in OOD_ORDER:
        bucket_metrics = metrics[bucket_name]
        rows.append(
            " & ".join(
                (
                    _sc(label),
                    _mean(bucket_metrics, "verdict_accuracy"),
                    _mean(bucket_metrics, "macro_f1"),
                    _mean(bucket_metrics, "unsafe_acceptance_rate"),
                    _mean(bucket_metrics, "wise_refusal_recall"),
                    _ci(bucket_metrics, "verdict_accuracy"),
                )
            )
            + ROW_END
        )
    return "\n".join(rows) + "\n"


def _render_adversarial_rows(metrics: dict[str, Any]) -> str:
    rows: list[str] = []
    for strength_name, label in ADVERSARIAL_ORDER:
        for split_name in SPLITS:
            split_metrics = metrics[strength_name][split_name]
            rows.append(
                " & ".join(
                    (
                        _sc(label),
                        split_name.replace("test_", "").replace("_", r"\_"),
                        _mean_std(split_metrics, "verdict_accuracy"),
                        _mean_std(split_metrics, "wise_refusal_recall"),
                    )
                )
                + ROW_END
            )
    return "\n".join(rows) + "\n"


def _poster_metrics(source_runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    main = source_runs["main_benchmark"]["aggregated_metrics"]
    ood = source_runs["ood_generalization"]["aggregated_metrics"]
    leakage = source_runs["leakage_study"]["aggregated_metrics"]
    main_config = source_runs["main_benchmark"]["config"]["effective"]
    seeds = source_runs["main_benchmark"]["seeds"]
    return {
        "status": "fixed-code exploratory snapshot",
        "setup": {
            "seeds": seeds,
            "seed_count": len(seeds),
            "samples_per_family": int(main_config["samples_per_family"]),
            "difficulty": float(main_config["difficulty"]),
            "ci_level": 0.95,
        },
        "main": {
            system_name: {
                split_name: {
                    "verdict_accuracy": _round_metric(split_metrics, "verdict_accuracy"),
                    "unsafe_acceptance_rate": _round_metric(split_metrics, "unsafe_acceptance_rate"),
                    "wise_refusal_recall": _round_metric(split_metrics, "wise_refusal_recall"),
                }
                for split_name, split_metrics in split_payload.items()
            }
            for system_name, split_payload in main.items()
        },
        "ood": {
            bucket_name: {
                "verdict_accuracy": _round_metric(bucket_metrics, "verdict_accuracy"),
                "macro_f1": _round_metric(bucket_metrics, "macro_f1"),
                "unsafe_acceptance_rate": _round_metric(bucket_metrics, "unsafe_acceptance_rate"),
                "wise_refusal_recall": _round_metric(bucket_metrics, "wise_refusal_recall"),
            }
            for bucket_name, bucket_metrics in ood.items()
        },
        "leakage": {
            split_name: {
                "clean_accuracy": _round_metric(leakage["countermodel_grounded"][split_name], "verdict_accuracy"),
                "leaking_accuracy": _round_metric(leakage["oracle_leaking_partition"][split_name], "verdict_accuracy"),
            }
            for split_name in SPLITS
        },
    }


def _discover_api_smoke_runs(source_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(source_dir.glob("api_baseline_smoke*.json")):
        payload = _read_json(path)
        runs.append(
            {
                "generated_at_utc": payload.get("generated_at_utc"),
                "status": payload.get("status"),
                "note": payload.get("note"),
                "model": payload.get("model", {}),
                "config": payload.get("config", {}),
                "summary": payload.get("summary", {}),
                "files": {
                    "json": {
                        "path": str(path.as_posix()),
                        "sha256": _sha256(path),
                    }
                },
            }
        )
    return runs


def _artifact_path(source_dir: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else source_dir / path


def _discover_llm_baseline_matrix(source_dir: Path) -> dict[str, Any] | None:
    manifest_path = source_dir / "llm_baseline_matrix" / "manifest.json"
    if not manifest_path.exists():
        return None

    manifest = _read_json(manifest_path)
    summary = dict(manifest.get("summary") or {})
    artifacts = dict(manifest.get("artifacts") or {})
    files: dict[str, dict[str, str]] = {
        "manifest": {
            "path": str(manifest_path.as_posix()),
            "sha256": _sha256(manifest_path),
        }
    }
    for artifact_name, artifact_value in artifacts.items():
        artifact_path = _artifact_path(source_dir, artifact_value)
        if artifact_path.exists():
            files[artifact_name] = {
                "path": str(artifact_path.as_posix()),
                "sha256": _sha256(artifact_path),
            }

    completed = (
        manifest.get("status") == "llm_baseline_matrix"
        and int(summary.get("total_predictions", 0)) > 0
        and int(summary.get("failed", 0)) == 0
        and int(summary.get("fallback_records", 0)) == 0
        and int(summary.get("parse_errors", 0)) == 0
    )
    return {
        "status": manifest.get("status"),
        "generated_at_utc": manifest.get("generated_at_utc"),
        "summary": summary,
        "completed": completed,
        "probe": bool(manifest.get("probe", False)),
        "files": files,
    }


def build_paper_facing_package(
    *,
    source_dir: Path | str = Path("outputs"),
    output_dir: Path | str = Path("outputs/paper_facing"),
    generated_at_utc: str | None = None,
) -> dict[str, Path]:
    """Generate the paper-facing artifact bundle and return written paths."""

    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source_runs: dict[str, dict[str, Any]] = {}
    manifest_runs: dict[str, Any] = {}
    for run_name, stem in SOURCE_RUNS.items():
        files = _source_file_map(source_path, stem)
        aggregated_metrics = _read_json(files["aggregated_metrics"])
        config = _read_json(files["config"])
        seed_list = _read_json(files["seed_list"])
        source_runs[run_name] = {
            "aggregated_metrics": aggregated_metrics,
            "config": config,
            "seeds": list(seed_list.get("seeds", [])),
        }
        manifest_runs[run_name] = {
            "stem": stem,
            "config": config,
            "seeds": list(seed_list.get("seeds", [])),
            "files": {
                file_key: {
                    "path": str(path.as_posix()),
                    "sha256": _sha256(path),
                }
                for file_key, path in files.items()
            },
        }

    generated = generated_at_utc or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    written: dict[str, Path] = {}
    table_payloads = {
        "main_table_rows.tex": _render_main_rows(source_runs["main_benchmark"]["aggregated_metrics"]),
        "ablation_table_rows.tex": _render_ablation_rows(source_runs["identifiability_ablation"]["aggregated_metrics"]),
        "leakage_table_rows.tex": _render_leakage_rows(source_runs["leakage_study"]["aggregated_metrics"]),
        "ood_table_rows.tex": _render_ood_rows(source_runs["ood_generalization"]["aggregated_metrics"]),
        "adversarial_table_rows.tex": _render_adversarial_rows(source_runs["adversarial_robustness"]["aggregated_metrics"]),
    }
    for filename, content in table_payloads.items():
        target = output_path / filename
        target.write_text(content, encoding="utf-8")
        written[filename.removesuffix(".tex")] = target

    poster_metrics = _poster_metrics(source_runs)
    poster_metrics_path = output_path / "poster_metrics.json"
    poster_metrics_path.write_text(json.dumps(poster_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    written["poster_metrics"] = poster_metrics_path

    api_smoke_runs = _discover_api_smoke_runs(source_path)
    llm_baseline_matrix = _discover_llm_baseline_matrix(source_path)
    real_api_smoke_completed = any(
        int((run.get("summary") or {}).get("fallback_records", 0)) == 0
        and int((run.get("summary") or {}).get("total", 0)) > 0
        for run in api_smoke_runs
    )
    full_llm_matrix_completed = bool(llm_baseline_matrix and llm_baseline_matrix.get("completed"))
    manifest = {
        "generated_at_utc": generated,
        "generator": "experiments.paper_artifacts.build_paper_facing_package",
        "artifact_status": "fixed-code exploratory snapshot",
        "human_audit_status": "pending_external_dual_annotation",
        "api_model_baseline_status": (
            "full_llm_baseline_matrix_completed"
            if full_llm_matrix_completed
            else "real_api_smoke_completed; full baseline matrix pending"
            if real_api_smoke_completed
            else "smoke_runner_available; full baseline matrix pending"
        ),
        "api_smoke_runs": api_smoke_runs,
        "llm_baseline_matrix": llm_baseline_matrix,
        "source_runs": manifest_runs,
        "tables": {key: str(path.as_posix()) for key, path in written.items() if key.endswith("_rows")},
        "poster_metrics": str(poster_metrics_path.as_posix()),
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    written["manifest"] = manifest_path
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_facing"))
    args = parser.parse_args()
    written = build_paper_facing_package(source_dir=args.source_dir, output_dir=args.output_dir)
    print(json.dumps({key: str(path) for key, path in written.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
