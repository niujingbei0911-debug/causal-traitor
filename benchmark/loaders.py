"""Loaders for benchmark split manifests and split-indexed instances."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.real_grounded import (
    RealGroundedCase,
    RealGroundedDataset,
    ensure_real_grounded_dataset,
)
from benchmark.schema import BenchmarkSplitManifest, ClaimInstance


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
    return [
        item if isinstance(item, ClaimInstance) else ClaimInstance.from_dict(dict(item))
        for item in instances
    ]


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
                instance = instance_by_id[instance_id]
            except KeyError as exc:
                raise KeyError(
                    f"Manifest references unknown instance_id {instance_id!r} in split {split_name!r}."
                ) from exc
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
    return ensure_real_grounded_dataset(payload)


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


def save_real_grounded_dataset(
    dataset: RealGroundedDataset | dict[str, Any] | list[Any],
    destination: str | Path,
) -> Path:
    """Serialize a real-grounded dataset into the canonical JSON export format."""

    normalized = ensure_real_grounded_dataset(dataset)
    path = Path(destination)
    path.write_text(
        json.dumps(normalized.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
