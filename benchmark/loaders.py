"""Loaders for benchmark split manifests and split-indexed instances."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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

    splits: dict[str, list[ClaimInstance]] = {}
    for split_name, ids in manifest.split_map().items():
        resolved: list[ClaimInstance] = []
        for instance_id in ids:
            try:
                resolved.append(instance_by_id[instance_id])
            except KeyError as exc:
                raise KeyError(
                    f"Manifest references unknown instance_id {instance_id!r} in split {split_name!r}."
                ) from exc
        splits[split_name] = resolved
    return splits
