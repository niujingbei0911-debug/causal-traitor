"""Phase 4 human audit subset generation and agreement interface."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evaluation.reporting import summarize_human_audit_agreement
from experiments.benchmark_harness import (
    DEFAULT_SEEDS,
    OOD_SPLITS,
    aggregate_seed_metrics,
    build_seed_benchmark_run,
    evaluate_system_on_samples,
    manifest_metadata,
    write_artifacts,
)

SYSTEM_NAME = "countermodel_grounded"
AUDIT_FIELDS: tuple[str, ...] = (
    "gold_label_reasonable",
    "verifier_label_reasonable",
    "witness_persuasive",
    "explanation_faithful",
)


def _normalize_seeds(seeds: list[int] | tuple[int, ...] | None) -> list[int]:
    if not seeds:
        return list(DEFAULT_SEEDS)
    return [int(seed) for seed in seeds]


def _round_robin_subset(records: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = f"{record['split']}::{record['gold_label']}"
        buckets.setdefault(key, []).append(record)
    ordered_keys = sorted(buckets)
    for key in ordered_keys:
        buckets[key] = sorted(buckets[key], key=lambda item: (item["seed"], item["instance_id"]))

    selected: list[dict[str, Any]] = []
    while len(selected) < size and ordered_keys:
        next_keys: list[str] = []
        for key in ordered_keys:
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.pop(0))
                if len(selected) >= size:
                    break
            if bucket:
                next_keys.append(key)
        ordered_keys = next_keys
    return selected


def _build_annotation_record(record: dict[str, Any]) -> dict[str, Any]:
    verdict = dict(record["verdict"])
    witness = verdict.get("witness") or {}
    return {
        "audit_id": f"{record['split']}::{record['instance_id']}",
        "seed": record["seed"],
        "split": record["split"],
        "instance_id": record["instance_id"],
        "graph_family": record["graph_family"],
        "query_type": record["query_type"],
        "attack_name": record["attack_name"],
        "claim_text": record["claim_text"],
        "gold_label": record["gold_label"],
        "predicted_label": record["predicted_label"],
        "verifier_confidence": record["confidence"],
        "supports_public_only": bool(record.get("supports_public_only", True)),
        "reasoning_summary": verdict.get("reasoning_summary"),
        "witness_description": witness.get("description"),
        "witness_type": witness.get("witness_type"),
        "annotation_questions": {
            "gold_label_reasonable": "Is the gold label itself reasonable for this claim and public evidence?",
            "verifier_label_reasonable": "Is the verifier label reasonable given the public evidence?",
            "witness_persuasive": "Is the witness persuasive enough to justify the verifier's decision?",
            "explanation_faithful": "Is the reasoning summary faithful to the witness and tool evidence?",
        },
        "annotator_a_gold_label_reasonable": None,
        "annotator_a_verifier_label_reasonable": None,
        "annotator_a_witness_persuasive": None,
        "annotator_a_explanation_faithful": None,
        "annotator_a_notes": "",
        "annotator_b_gold_label_reasonable": None,
        "annotator_b_verifier_label_reasonable": None,
        "annotator_b_witness_persuasive": None,
        "annotator_b_explanation_faithful": None,
        "annotator_b_notes": "",
        "arbiter_notes": "",
    }


def _load_annotations(source: str | None) -> list[dict[str, Any]] | None:
    if not source:
        return None
    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Annotation file must contain a JSON list.")
    return [dict(item) for item in payload]


def _write_annotation_sidecars(base_json_path: Path, package: list[dict[str, Any]]) -> dict[str, str]:
    package_json = base_json_path.with_name(f"{base_json_path.stem}_annotation_package.json")
    package_csv = base_json_path.with_name(f"{base_json_path.stem}_annotation_package.csv")
    package_json.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    with package_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(package[0].keys()) if package else ["audit_id"])
        writer.writeheader()
        for row in package:
            writer.writerow(row)
    return {"annotation_package_json": str(package_json), "annotation_package_csv": str(package_csv)}


def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Human Audit",
        "",
        f"- Seeds: {payload['seeds']}",
        f"- Audit subset size: {payload['config']['audit_subset_size']}",
        f"- Generated records: {len(payload['annotation_package'])}",
        "",
        "## Benchmark Snapshot",
        "",
        "| Split | Verdict Acc. | Macro-F1 | Invalid Accept | Unidentifiable Awareness |",
        "| --- | --- | --- | --- | --- |",
    ]
    for split_name in OOD_SPLITS:
        metrics = payload["aggregated_metrics"][split_name]
        lines.append(
            f"| {split_name} | "
            f"{metrics['verdict_accuracy']['formatted']} | "
            f"{metrics['macro_f1']['formatted']} | "
            f"{metrics['invalid_claim_acceptance_rate']['formatted']} | "
            f"{metrics['unidentifiable_awareness']['formatted']} |"
        )
    lines.extend(
        [
            "",
        "## Annotation Fields",
        "",
        ]
    )
    for field_name in AUDIT_FIELDS:
        lines.append(f"- `{field_name}`")
    if payload.get("agreement_stats") is not None:
        lines.append("")
        lines.append("## Agreement Stats")
        lines.append("")
        for field_name, stats in payload["agreement_stats"]["fields"].items():
            lines.append(
                f"- {field_name}: n={stats['n_scored']}, "
                f"agreement={stats['percent_agreement']:.4f}, "
                f"kappa={stats['cohen_kappa']:.4f}"
            )
    else:
        lines.append("")
        lines.append("## Agreement Interface")
        lines.append("")
        lines.append("Provide a JSON list with annotator A/B fields filled in, then rerun with `--annotations` to compute agreement.")
    return "\n".join(lines)


def run_experiment(
    *,
    seeds: list[int] | tuple[int, ...] | None = None,
    samples_per_family: int = 2,
    difficulty: float = 0.55,
    audit_subset_size: int = 12,
    annotations_path: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    resolved_seeds = _normalize_seeds(seeds)
    raw_predictions: list[dict[str, Any]] = []
    manifests: dict[int, dict[str, Any]] = {}
    per_seed_results: dict[int, Any] = {}

    for seed in resolved_seeds:
        run = build_seed_benchmark_run(
            seed=seed,
            difficulty=difficulty,
            samples_per_family=samples_per_family,
        )
        manifests[seed] = manifest_metadata(run)
        ood_reasons = dict(run.manifest.metadata.get("ood_reasons", {}))
        seed_payload: dict[str, Any] = {}
        for split_name in OOD_SPLITS:
            evaluated = evaluate_system_on_samples(
                run.split_samples[split_name],
                seed=seed,
                split_name=split_name,
                system_name=SYSTEM_NAME,
                ood_reasons=ood_reasons,
            )
            raw_predictions.extend(evaluated["predictions"])
            seed_payload[split_name] = evaluated
        per_seed_results[seed] = seed_payload

    subset = _round_robin_subset(raw_predictions, int(audit_subset_size))
    annotation_package = [_build_annotation_record(record) for record in subset]
    aggregated_metrics = {
        split_name: aggregate_seed_metrics(
            {
                seed: per_seed_results[seed]
                for seed in resolved_seeds
            },
            split_name=split_name,
        )
        for split_name in OOD_SPLITS
    }
    loaded_annotations = _load_annotations(annotations_path)
    agreement_stats = (
        summarize_human_audit_agreement(loaded_annotations, fields=AUDIT_FIELDS)
        if loaded_annotations is not None
        else None
    )

    payload = {
        "experiment_id": "exp_human_audit",
        "config": {
            "samples_per_family": int(samples_per_family),
            "difficulty": float(difficulty),
            "audit_subset_size": int(audit_subset_size),
        },
        "system": SYSTEM_NAME,
        "seeds": resolved_seeds,
        "manifests": manifests,
        "per_seed_results": per_seed_results,
        "aggregated_metrics": aggregated_metrics,
        "annotation_package": annotation_package,
        "agreement_fields": list(AUDIT_FIELDS),
        "agreement_stats": agreement_stats,
        "raw_predictions": raw_predictions,
    }

    summary = _markdown_summary(payload)
    artifacts = write_artifacts(
        output_path=output_path or "outputs/exp_human_audit.json",
        payload=payload,
        markdown_summary=summary,
    )
    annotation_artifacts = _write_annotation_sidecars(Path(artifacts["json"]), annotation_package)
    payload["markdown_summary"] = summary
    payload["artifacts"] = {**artifacts, **annotation_artifacts}
    Path(artifacts["json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 4 human audit packaging workflow.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Explicit seed list.")
    parser.add_argument("--samples-per-family", type=int, default=2, help="Samples generated per benchmark family.")
    parser.add_argument("--difficulty", type=float, default=0.55, help="Benchmark generation difficulty.")
    parser.add_argument("--audit-subset-size", type=int, default=12, help="Number of records in the annotation package.")
    parser.add_argument("--annotations", default=None, help="Optional JSON file with completed dual annotations.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_experiment(
        seeds=args.seeds,
        samples_per_family=args.samples_per_family,
        difficulty=args.difficulty,
        audit_subset_size=args.audit_subset_size,
        annotations_path=args.annotations,
        output_path=args.output,
    )
    print(json.dumps(payload.get("agreement_stats") or {"annotation_package_size": len(payload["annotation_package"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
