"""Loaders for benchmark split manifests and split-indexed instances."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark.generator import BenchmarkSample
from benchmark.graph_families import GraphFamilyBlueprint, IdentifiabilityStatus
from benchmark.real_grounded import (
    RealGroundedCase,
    RealGroundedDataset,
    ensure_real_grounded_dataset,
)
from benchmark.schema import BenchmarkSplitManifest, ClaimInstance, GoldCausalInstance, PublicCausalInstance


def _normalize_manifest_source(
    source: BenchmarkSplitManifest | dict[str, Any] | str | Path,
) -> BenchmarkSplitManifest:
    if isinstance(source, BenchmarkSplitManifest):
        return source
    if isinstance(source, dict):
        return BenchmarkSplitManifest.from_dict(source)

    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return BenchmarkSplitManifest.from_dict(payload)


def _normalize_instances(
    instances: list[ClaimInstance | dict[str, Any]],
) -> list[ClaimInstance]:
    normalized: list[ClaimInstance] = []
    seen_ids: set[str] = set()
    for item in instances:
        instance = (
            ClaimInstance.from_dict(item.to_dict())
            if isinstance(item, ClaimInstance)
            else ClaimInstance.from_dict(dict(item))
        )
        if instance.instance_id in seen_ids:
            raise ValueError(f"Duplicate instance_id detected while loading splits: {instance.instance_id!r}.")
        seen_ids.add(instance.instance_id)
        normalized.append(instance)
    return normalized


def load_split_manifest(
    source: BenchmarkSplitManifest | dict[str, Any] | str | Path,
) -> BenchmarkSplitManifest:
    """Load a split manifest from an in-memory object or JSON file."""

    return _normalize_manifest_source(source)


def load_split_ids(
    source: BenchmarkSplitManifest | dict[str, Any] | str | Path,
) -> dict[str, list[str]]:
    """Load split ids keyed by `train/dev/test_iid/test_ood`."""

    manifest = _normalize_manifest_source(source)
    return manifest.split_map()


def load_split_instances(
    instances: list[ClaimInstance | dict[str, Any]],
    manifest_source: BenchmarkSplitManifest | dict[str, Any] | str | Path,
) -> dict[str, list[ClaimInstance]]:
    """Resolve split ids from a manifest into actual ClaimInstance objects."""

    manifest = _normalize_manifest_source(manifest_source)
    normalized_instances = _normalize_instances(instances)
    instance_by_id = {instance.instance_id: instance for instance in normalized_instances}
    ood_reasons = manifest.metadata.get("ood_reasons", {})
    ood_reasons = ood_reasons if isinstance(ood_reasons, dict) else {}

    splits: dict[str, list[ClaimInstance]] = {}
    for split_name, ids in manifest.split_map().items():
        resolved: list[ClaimInstance] = []
        for instance_id in ids:
            try:
                instance = ClaimInstance.from_dict(instance_by_id[instance_id].to_dict())
            except KeyError as exc:
                raise KeyError(
                    f"Manifest references unknown instance_id {instance_id!r} in split {split_name!r}."
                ) from exc
            instance.meta.pop("ood_split", None)
            instance.meta.pop("ood_reasons", None)
            instance.meta["ood_split"] = split_name
            if instance_id in ood_reasons:
                instance.meta["ood_reasons"] = list(ood_reasons[instance_id])
            resolved.append(instance)
        splits[split_name] = resolved
    return splits


def _normalize_real_grounded_source(
    source: RealGroundedDataset | RealGroundedCase | dict[str, Any] | list[Any] | str | Path,
) -> RealGroundedDataset:
    if isinstance(source, RealGroundedCase):
        return RealGroundedDataset(cases=[source])
    if isinstance(source, RealGroundedDataset):
        return ensure_real_grounded_dataset(source)
    if isinstance(source, (dict, list)):
        return ensure_real_grounded_dataset(source)

    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset = ensure_real_grounded_dataset(payload)
    dataset.metadata = {
        **dict(dataset.metadata),
        "_source_path": str(path.resolve()),
        "_source_root": str(path.resolve().parent),
    }
    return dataset


def load_real_grounded_dataset(
    source: RealGroundedDataset | RealGroundedCase | dict[str, Any] | list[Any] | str | Path,
) -> RealGroundedDataset:
    """Load a real-grounded dataset from memory or a JSON file."""

    return _normalize_real_grounded_source(source)


def load_real_grounded_cases(
    source: RealGroundedDataset | RealGroundedCase | dict[str, Any] | list[Any] | str | Path,
) -> list[RealGroundedCase]:
    """Resolve a real-grounded dataset source into case objects."""

    return list(_normalize_real_grounded_source(source).cases)


def load_real_grounded_claims(
    source: RealGroundedDataset | RealGroundedCase | dict[str, Any] | list[Any] | str | Path,
) -> list[ClaimInstance]:
    """Extract ClaimInstance objects from a real-grounded dataset source."""

    return _normalize_real_grounded_source(source).claim_instances()


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(item).strip() for item in value if str(item).strip()]


def _coerce_frame_from_payload(value: Any, *, field_name: str) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        frame = value.copy(deep=True)
    elif isinstance(value, list):
        frame = pd.DataFrame(list(value))
    elif isinstance(value, dict):
        frame = pd.DataFrame(value)
    else:
        raise ValueError(f"{field_name} must be a DataFrame-compatible payload, got {type(value)!r}.")
    if frame.empty:
        raise ValueError(f"{field_name} must contain at least one row.")
    return frame.copy(deep=True)


def _read_real_grounded_data_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".js"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "records" in payload:
            payload = payload["records"]
        return _coerce_frame_from_payload(payload, field_name=f"observed_data_path={path}")
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(
        f"Unsupported observed_data_path suffix {suffix!r} for real-grounded case file {str(path)!r}."
    )


def _resolve_real_grounded_data_path(dataset: RealGroundedDataset, observed_data_path: str) -> Path:
    path = Path(observed_data_path)
    if path.is_absolute():
        return path
    source_root = dataset.metadata.get("_source_root")
    if source_root:
        return Path(str(source_root)) / path
    return path


def _real_grounded_observed_data(
    case: RealGroundedCase,
    *,
    dataset: RealGroundedDataset,
) -> pd.DataFrame:
    claim = case.claim
    metadata = dict(case.metadata)
    payload = metadata.get("observed_data_records", metadata.get("observed_data"))
    if payload is not None:
        frame = _coerce_frame_from_payload(
            payload,
            field_name=f"RealGroundedCase[{case.case_id!r}].metadata['observed_data_records']",
        )
    elif claim.observed_data_path:
        data_path = _resolve_real_grounded_data_path(dataset, claim.observed_data_path)
        if not data_path.exists():
            raise ValueError(
                f"Real-grounded observed_data_path does not exist for case {case.case_id!r}: {str(data_path)!r}."
            )
        frame = _read_real_grounded_data_file(data_path)
    else:
        raise ValueError(
            "Real-grounded samples require an explicit observed-data source via "
            "case.metadata['observed_data_records'] / case.metadata['observed_data'] "
            f"or ClaimInstance.observed_data_path. Missing for case {case.case_id!r}."
        )

    expected_columns = list(claim.observed_variables)
    for role in ("treatment", "outcome"):
        variable_name = str(claim.target_variables[role])
        if variable_name not in expected_columns:
            expected_columns.append(variable_name)
    missing_columns = [column for column in expected_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"Real-grounded observed data for case {case.case_id!r} is missing required columns: {missing_columns!r}."
        )
    return frame.loc[:, expected_columns].copy(deep=True)


def _real_grounded_hidden_variables(case: RealGroundedCase) -> list[str]:
    return _normalize_string_list(case.metadata.get("hidden_variables", []))


def _real_grounded_blueprint(case: RealGroundedCase) -> GraphFamilyBlueprint:
    claim = case.claim
    treatment = str(claim.target_variables["treatment"])
    outcome = str(claim.target_variables["outcome"])
    observed_variables = list(claim.observed_variables)
    if treatment not in observed_variables:
        observed_variables.append(treatment)
    if outcome not in observed_variables:
        observed_variables.append(outcome)
    hidden_variables = _real_grounded_hidden_variables(case)
    all_variables = [*observed_variables, *[variable for variable in hidden_variables if variable not in observed_variables]]
    raw_true_dag = case.metadata.get("true_dag")
    if isinstance(raw_true_dag, dict):
        true_dag = {
            str(parent): _normalize_string_list(children)
            for parent, children in raw_true_dag.items()
        }
    else:
        true_dag = {variable_name: [] for variable_name in all_variables}
    for variable_name in all_variables:
        true_dag.setdefault(variable_name, [])
    role_bindings = {
        **{
            str(key): str(value)
            for key, value in dict(case.metadata.get("role_bindings", {})).items()
        },
        "treatment": treatment,
        "outcome": outcome,
    }
    return GraphFamilyBlueprint(
        family_name=claim.graph_family,
        causal_level=claim.causal_level,
        identifiability=IdentifiabilityStatus.POTENTIALLY_UNIDENTIFIABLE,
        description=case.public_evidence_summary,
        role_bindings=role_bindings,
        target_variables={
            "treatment": treatment,
            "outcome": outcome,
        },
        true_dag=true_dag,
        hidden_variables=hidden_variables,
        observed_variables=observed_variables,
        seed=0,
        proxy_variables=list(claim.proxy_variables),
        selection_variables=_normalize_string_list(case.metadata.get("selection_variables", [])),
        query_types=[claim.query_type],
        supported_gold_labels=[claim.gold_label.value],
        generator_hints={
            "source": "real_grounded_subset_loader",
            "grounding_type": case.grounding_type.value,
        },
        family_tags=["real_grounded", case.grounding_type.value],
    )


def _real_grounded_gold_instance(
    case: RealGroundedCase,
    *,
    observed_data: pd.DataFrame,
    blueprint: GraphFamilyBlueprint,
) -> GoldCausalInstance:
    claim = case.claim
    treatment = str(claim.target_variables["treatment"])
    outcome = str(claim.target_variables["outcome"])
    return GoldCausalInstance(
        scenario_id=f"real_grounded::{case.case_id}",
        description=case.public_evidence_summary,
        true_dag=blueprint.true_dag,
        variables=list(observed_data.columns),
        hidden_variables=list(blueprint.hidden_variables),
        ground_truth={
            "label": claim.gold_label.value,
            "treatment": treatment,
            "outcome": outcome,
            "answer": claim.gold_answer,
        },
        observed_data=observed_data.copy(deep=True),
        full_data=observed_data.copy(deep=True),
        data=observed_data.copy(deep=True),
        causal_level=int(str(claim.causal_level).lstrip("L")),
        difficulty=0.5,
        true_scm={"source": "real_grounded_subset_loader"},
        gold_label=claim.gold_label,
        metadata={
            "public_scenario_id": f"public_real_grounded::{case.case_id}",
            "public_description": case.public_evidence_summary,
            "source_citation": case.source_citation.to_dict(),
            "information_contract": case.information_contract.to_dict(),
            "identifying_assumptions": list(case.identifying_assumptions),
            "witness_note": case.witness_note,
        },
    )


def _real_grounded_public_instance(
    case: RealGroundedCase,
    *,
    observed_data: pd.DataFrame,
) -> PublicCausalInstance:
    claim = case.claim
    return PublicCausalInstance(
        scenario_id=f"public_real_grounded::{case.case_id}",
        description=case.public_evidence_summary,
        variables=list(observed_data.columns),
        proxy_variables=list(claim.proxy_variables),
        selection_mechanism=claim.selection_mechanism or claim.meta.get("selection_mechanism"),
        observed_data=observed_data.copy(deep=True),
        data=observed_data.copy(deep=True),
        causal_level=int(str(claim.causal_level).lstrip("L")),
        metadata={
            "context_shift_group": case.grounding_type.value,
        },
    )


def load_real_grounded_samples(
    source: RealGroundedDataset | RealGroundedCase | dict[str, Any] | list[Any] | str | Path,
) -> list[BenchmarkSample]:
    """Resolve real-grounded cases into benchmark-style samples for experiment runners."""

    dataset = _normalize_real_grounded_source(source)
    samples: list[BenchmarkSample] = []
    for case in dataset.cases:
        claim = ClaimInstance.from_dict(case.claim.to_dict())
        claim.meta = {
            **dict(claim.meta),
            "data_origin": "real_grounded",
            "dataset_partition": "real_grounded",
            "grounding_type": case.grounding_type.value,
            "real_grounded_case_id": case.case_id,
        }
        observed_data = _real_grounded_observed_data(case, dataset=dataset)
        case_for_sample = RealGroundedCase.from_dict(
            {
                **case.to_dict(),
                "claim": claim.to_dict(),
            }
        )
        blueprint = _real_grounded_blueprint(case_for_sample)
        gold = _real_grounded_gold_instance(
            case_for_sample,
            observed_data=observed_data,
            blueprint=blueprint,
        )
        public = _real_grounded_public_instance(case_for_sample, observed_data=observed_data)
        samples.append(
            BenchmarkSample(
                claim=claim,
                gold=gold,
                public=public,
                blueprint=blueprint,
            )
        )
    return samples


def save_real_grounded_dataset(
    dataset: RealGroundedDataset | dict[str, Any] | list[Any],
    destination: str | Path,
) -> Path:
    """Serialize a real-grounded dataset into the canonical JSON export format."""

    normalized = ensure_real_grounded_dataset(dataset)
    path = Path(destination)
    payload = normalized.to_dict()
    payload["metadata"] = {
        key: value
        for key, value in dict(payload.get("metadata", {})).items()
        if not str(key).startswith("_")
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
