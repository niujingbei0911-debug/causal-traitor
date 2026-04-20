"""Phase 4 human audit subset generation and agreement interface."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evaluation.reporting import normalize_human_audit_label, summarize_human_audit_agreement
from experiments.benchmark_harness import (
    MIN_FORMAL_SAMPLES_PER_FAMILY,
    MIN_HUMAN_AUDIT_SUBSET_SIZE,
    MIN_FORMAL_SEED_COUNT,
    OOD_SPLITS,
    aggregate_seed_metrics,
    build_seed_benchmark_run,
    evaluate_system_on_samples,
    manifest_metadata,
    normalize_benchmark_difficulty,
    normalize_benchmark_samples_per_family,
    normalize_experiment_seeds,
    summarize_protocol_compliance,
    write_artifacts,
)

SYSTEM_NAME = "countermodel_grounded"
DEFAULT_SAMPLES_PER_FAMILY = 15
AUDIT_FIELDS: tuple[str, ...] = (
    "gold_label_reasonable",
    "verifier_label_reasonable",
    "witness_quality_reasonable",
    "explanation_faithful",
)
ANNOTATION_MUTABLE_FIELDS: tuple[str, ...] = (
    "annotator_a_gold_label_reasonable",
    "annotator_a_verifier_label_reasonable",
    "annotator_a_witness_quality_reasonable",
    "annotator_a_explanation_faithful",
    "annotator_a_notes",
    "annotator_b_gold_label_reasonable",
    "annotator_b_verifier_label_reasonable",
    "annotator_b_witness_quality_reasonable",
    "annotator_b_explanation_faithful",
    "annotator_b_notes",
    "arbiter_notes",
)


def _round_robin_subset(records: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        ood_bucket = "+".join(sorted(record.get("ood_reasons", []))) or "iid"
        _, _, _, primary_witness, _, _ = _resolve_primary_witness(record.get("verdict") or {})
        witness_type = (primary_witness or {}).get("witness_type") or "none"
        attack_name = record.get("attack_name") or "truthful"
        causal_level = record.get("causal_level")
        if causal_level is None:
            causal_level = (record.get("public_evidence_summary") or {}).get("causal_level", 0)
        key = (
            f"{record['split']}::{record['gold_label']}::{record.get('query_type')}::{attack_name}"
            f"::{witness_type}::{ood_bucket}::{causal_level}"
        )
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


def _resolve_primary_witness(
    verdict: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, str, str]:
    witness = dict(verdict.get("witness") or {})
    support_witness = verdict.get("support_witness")
    countermodel_witness = verdict.get("countermodel_witness")
    primary_witness = countermodel_witness or support_witness or witness or None
    primary_witness_role = "none"
    witness_question = "If no witness is present, mark N/A for witness quality."
    if countermodel_witness is not None:
        primary_witness_role = "countermodel"
        witness_question = (
            "If a countermodel witness is present, is it persuasive enough to justify the verifier's decision?"
        )
    elif support_witness is not None:
        primary_witness_role = "support"
        witness_question = (
            "If a support witness is present, is it persuasive enough to justify the verifier's decision?"
        )
    elif primary_witness:
        primary_witness_role = str(primary_witness.get("witness_type") or "generic")
        witness_question = (
            "If a witness is present, is it persuasive enough to justify the verifier's decision?"
        )
    return witness, support_witness, countermodel_witness, primary_witness, primary_witness_role, witness_question


def _build_annotation_record(record: dict[str, Any]) -> dict[str, Any]:
    verdict = dict(record["verdict"])
    _, support_witness, countermodel_witness, primary_witness, primary_witness_role, witness_question = (
        _resolve_primary_witness(verdict)
    )
    public_payload = dict(record.get("public_evidence_summary", {}))
    return {
        "audit_id": f"{record['split']}::{record['instance_id']}",
        "seed": record["seed"],
        "split": record["split"],
        "instance_id": record["instance_id"],
        "causal_level": record.get("causal_level", public_payload.get("causal_level")),
        "graph_family": record["graph_family"],
        "query_type": record["query_type"],
        "attack_name": record["attack_name"],
        "claim_text": record["claim_text"],
        "gold_label": record["gold_label"],
        "predicted_label": record["predicted_label"],
        "verifier_confidence": record["confidence"],
        "verifier_probabilities": dict(record.get("predicted_probabilities", {})),
        "supports_public_only": bool(record.get("supports_public_only", True)),
        "reasoning_summary": verdict.get("reasoning_summary"),
        "witness_description": (primary_witness or {}).get("description"),
        "witness_type": (primary_witness or {}).get("witness_type"),
        "primary_witness_role": primary_witness_role,
        "public_evidence_summary": dict(record.get("public_evidence_summary", {})),
        "observed_data": record.get("observed_data"),
        "supporting_evidence": list(record.get("supporting_evidence", [])),
        "counter_evidence": list(record.get("counter_evidence", [])),
        "tool_trace": list(record.get("tool_trace", [])),
        "support_witness": support_witness,
        "countermodel_witness": countermodel_witness,
        "primary_witness": primary_witness,
        "annotation_questions": {
            "gold_label_reasonable": "Is the gold label itself reasonable for this claim and public evidence?",
            "verifier_label_reasonable": "Is the verifier label reasonable given the public evidence?",
            "witness_quality_reasonable": witness_question,
            "explanation_faithful": "Is the reasoning summary faithful to the witness and tool evidence?",
        },
        "annotator_a_gold_label_reasonable": None,
        "annotator_a_verifier_label_reasonable": None,
        "annotator_a_witness_quality_reasonable": None,
        "annotator_a_explanation_faithful": None,
        "annotator_a_notes": "",
        "annotator_b_gold_label_reasonable": None,
        "annotator_b_verifier_label_reasonable": None,
        "annotator_b_witness_quality_reasonable": None,
        "annotator_b_explanation_faithful": None,
        "annotator_b_notes": "",
        "arbiter_notes": "",
    }


def _load_annotations(source: str | None) -> list[dict[str, Any]] | None:
    if not source:
        return None
    path = Path(source)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Annotation file must contain a JSON list.")
    return [dict(item) for item in payload]


def _canonicalize_annotation_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        lowered = stripped.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null":
            return None
        try:
            return _canonicalize_annotation_value(json.loads(stripped))
        except json.JSONDecodeError:
            return stripped
    if isinstance(value, dict):
        return {
            str(key): _canonicalize_annotation_value(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, list):
        return [_canonicalize_annotation_value(item) for item in value]
    return value


def _validate_loaded_annotations(
    loaded_annotations: list[dict[str, Any]] | None,
    *,
    annotation_package: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    if loaded_annotations is None:
        return None

    expected_by_id = {
        str(record["audit_id"]): record
        for record in annotation_package
    }
    loaded_ids = [str(record.get("audit_id", "")).strip() for record in loaded_annotations]
    duplicate_ids = sorted(
        audit_id
        for audit_id in {item for item in loaded_ids if item}
        if loaded_ids.count(audit_id) > 1
    )
    if duplicate_ids:
        raise ValueError(f"Loaded annotations contain duplicate audit_id values: {duplicate_ids!r}.")
    if set(loaded_ids) != set(expected_by_id):
        raise ValueError("Loaded annotations do not match the current audit package identities.")

    validated: list[dict[str, Any]] = []
    for record in loaded_annotations:
        audit_id = str(record.get("audit_id", "")).strip()
        expected = expected_by_id[audit_id]
        locked_fields = sorted(set(expected) - set(ANNOTATION_MUTABLE_FIELDS))
        unexpected_locked_fields = sorted(
            field_name
            for field_name in record
            if field_name not in expected and field_name not in ANNOTATION_MUTABLE_FIELDS
        )
        if unexpected_locked_fields:
            raise ValueError(
                f"Loaded annotations contain unexpected locked fields for audit_id={audit_id!r}: "
                f"{unexpected_locked_fields!r}."
            )
        for field_name in locked_fields:
            if _canonicalize_annotation_value(record.get(field_name)) != _canonicalize_annotation_value(expected.get(field_name)):
                raise ValueError(
                    f"Loaded annotations do not match the current audit package for audit_id={audit_id!r}, field={field_name!r}."
                )
        validated_record = dict(expected)
        for field_name in ANNOTATION_MUTABLE_FIELDS:
            if field_name in record:
                validated_record[field_name] = record[field_name]
        validated.append(validated_record)
    return validated


def _conflict_summary(records: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if records is None:
        return None
    conflicts: list[dict[str, Any]] = []
    for record in records:
        conflicting_fields: list[str] = []
        for field_name in AUDIT_FIELDS:
            left = normalize_human_audit_label(record.get(f"annotator_a_{field_name}"))
            right = normalize_human_audit_label(record.get(f"annotator_b_{field_name}"))
            if left and right and left != right:
                conflicting_fields.append(field_name)
        if conflicting_fields:
            conflicts.append(
                {
                    "audit_id": record["audit_id"],
                    "conflicting_fields": conflicting_fields,
                    "arbiter_notes": str(record.get("arbiter_notes", "")).strip(),
                }
            )
    return {
        "n_conflicts": len(conflicts),
        "n_conflicts_with_arbiter_notes": sum(1 for item in conflicts if item["arbiter_notes"]),
        "records": conflicts,
    }


def _write_annotation_sidecars(base_json_path: Path, package: list[dict[str, Any]]) -> dict[str, str]:
    package_json = base_json_path.with_name(f"{base_json_path.stem}_annotation_package.json")
    package_csv = base_json_path.with_name(f"{base_json_path.stem}_annotation_package.csv")
    package_json.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    with package_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(package[0].keys()) if package else ["audit_id"])
        writer.writeheader()
        for row in package:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )
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
            f"{metrics['unsafe_acceptance_rate']['formatted']} | "
            f"{metrics['wise_refusal_recall']['formatted']} |"
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
        if payload.get("conflict_summary") is not None:
            lines.append(
                f"- conflicts: {payload['conflict_summary']['n_conflicts']} "
                f"(with arbiter notes: {payload['conflict_summary']['n_conflicts_with_arbiter_notes']})"
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
    samples_per_family: int = DEFAULT_SAMPLES_PER_FAMILY,
    difficulty: float = 0.55,
    audit_subset_size: int = MIN_HUMAN_AUDIT_SUBSET_SIZE,
    annotations_path: str | None = None,
    allow_protocol_violations: bool = False,
    output_path: str | None = None,
) -> dict[str, Any]:
    resolved_seeds = normalize_experiment_seeds(
        seeds,
        minimum_count=MIN_FORMAL_SEED_COUNT,
        allow_protocol_violations=allow_protocol_violations,
    )
    resolved_samples_per_family = normalize_benchmark_samples_per_family(samples_per_family)
    resolved_difficulty = normalize_benchmark_difficulty(difficulty)
    raw_predictions: list[dict[str, Any]] = []
    manifests: dict[int, dict[str, Any]] = {}
    per_seed_results: dict[int, Any] = {}

    for seed in resolved_seeds:
        run = build_seed_benchmark_run(
            seed=seed,
            difficulty=resolved_difficulty,
            samples_per_family=resolved_samples_per_family,
        )
        manifests[seed] = manifest_metadata(run)
        ood_reasons = dict(run.manifest.metadata.get("ood_reasons", {}))
        seed_payload: dict[str, Any] = {}
        for split_name in ("test_iid", "test_ood"):
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

    resolved_audit_subset_size = min(int(audit_subset_size), len(raw_predictions))
    protocol = summarize_protocol_compliance(
        resolved_seeds,
        minimum_count=MIN_FORMAL_SEED_COUNT,
        minimum_samples_per_family=max(MIN_FORMAL_SAMPLES_PER_FAMILY, DEFAULT_SAMPLES_PER_FAMILY),
        observed_samples_per_family=resolved_samples_per_family,
        minimum_audit_subset_size=MIN_HUMAN_AUDIT_SUBSET_SIZE,
        observed_audit_subset_size=resolved_audit_subset_size,
        allow_protocol_violations=allow_protocol_violations,
    )
    subset = _round_robin_subset(raw_predictions, resolved_audit_subset_size)
    annotation_package = [_build_annotation_record(record) for record in subset]
    aggregated_metrics = {
        split_name: aggregate_seed_metrics(
            {
                seed: per_seed_results[seed]
                for seed in resolved_seeds
            },
            split_name=split_name,
        )
        for split_name in ("test_iid", "test_ood")
    }
    loaded_annotations = _validate_loaded_annotations(
        _load_annotations(annotations_path),
        annotation_package=annotation_package,
    )
    agreement_stats = (
        summarize_human_audit_agreement(loaded_annotations, fields=AUDIT_FIELDS)
        if loaded_annotations is not None
        else None
    )
    conflict_summary = _conflict_summary(loaded_annotations)

    payload = {
        "experiment_id": "exp_human_audit",
        "config": {
            "samples_per_family": int(resolved_samples_per_family),
            "difficulty": float(resolved_difficulty),
            "audit_subset_size": int(resolved_audit_subset_size),
        },
        "requested_config": {
            "samples_per_family": int(samples_per_family),
            "difficulty": float(difficulty),
            "audit_subset_size": int(audit_subset_size),
            "allow_protocol_violations": bool(allow_protocol_violations),
        },
        "system": SYSTEM_NAME,
        "seeds": resolved_seeds,
        "protocol": protocol,
        "manifests": manifests,
        "per_seed_results": per_seed_results,
        "aggregated_metrics": aggregated_metrics,
        "annotation_package": annotation_package,
        "agreement_fields": list(AUDIT_FIELDS),
        "agreement_stats": agreement_stats,
        "conflict_summary": conflict_summary,
        "raw_predictions": raw_predictions,
    }

    summary = _markdown_summary(payload)
    artifacts = write_artifacts(
        output_path=output_path or "outputs/mainline/exp_human_audit.json",
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
    parser.add_argument(
        "--samples-per-family",
        type=int,
        default=DEFAULT_SAMPLES_PER_FAMILY,
        help="Samples generated per benchmark family.",
    )
    parser.add_argument("--difficulty", type=float, default=0.55, help="Benchmark generation difficulty.")
    parser.add_argument(
        "--audit-subset-size",
        type=int,
        default=MIN_HUMAN_AUDIT_SUBSET_SIZE,
        help="Number of records in the annotation package.",
    )
    parser.add_argument("--annotations", default=None, help="Optional JSON file with completed dual annotations.")
    parser.add_argument(
        "--allow-protocol-violations",
        action="store_true",
        help="Allow exploratory runs that violate the formal >=3-seed Phase 4 protocol.",
    )
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
        allow_protocol_violations=args.allow_protocol_violations,
        output_path=args.output,
    )
    print(json.dumps(payload.get("agreement_stats") or {"annotation_package_size": len(payload["annotation_package"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
