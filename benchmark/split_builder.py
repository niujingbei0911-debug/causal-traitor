"""Split builder for the causal oversight benchmark."""

from __future__ import annotations

from collections.abc import Iterable
import random
from typing import Any

from benchmark.schema import BenchmarkSplitManifest, ClaimInstance


def _normalize_instances(
    instances: Iterable[ClaimInstance | dict[str, Any]],
) -> list[ClaimInstance]:
    normalized: list[ClaimInstance] = []
    seen_ids: set[str] = set()
    for item in instances:
        instance = item if isinstance(item, ClaimInstance) else ClaimInstance.from_dict(dict(item))
        if instance.instance_id in seen_ids:
            raise ValueError(f"Duplicate instance_id detected while building splits: {instance.instance_id!r}.")
        seen_ids.add(instance.instance_id)
        normalized.append(instance)
    return normalized


def _is_variable_renamed(instance: ClaimInstance) -> bool:
    meta = dict(instance.meta)
    if any(
        bool(meta.get(key))
        for key in ("variable_renaming", "variable_renaming_holdout", "variable_renamed")
    ):
        return True
    rename_map = meta.get("rename_map")
    if isinstance(rename_map, dict) and bool(rename_map):
        return True
    original_variables = meta.get("original_variables")
    if isinstance(original_variables, (list, tuple)):
        return [str(value) for value in original_variables] != list(instance.observed_variables)
    renamed_variables = meta.get("renamed_variables")
    if isinstance(renamed_variables, (list, tuple)) and len(renamed_variables) > 0:
        return True
    return False


def _shuffle_ids(instance_ids: list[str], *, seed: int) -> list[str]:
    rng = random.Random(int(seed))
    shuffled = list(instance_ids)
    rng.shuffle(shuffled)
    return shuffled


def _slice_in_domain_ids(
    candidate_ids: list[str],
    *,
    seed: int,
    dev_ratio: float,
    test_iid_ratio: float,
) -> dict[str, list[str]]:
    shuffled = _shuffle_ids(candidate_ids, seed=seed)
    total = len(shuffled)
    if total == 0:
        return {"train": [], "dev": [], "test_iid": []}

    n_dev = int(round(total * dev_ratio))
    n_test_iid = int(round(total * test_iid_ratio))

    if total >= 3:
        n_dev = max(1, n_dev)
    if total >= 4:
        n_test_iid = max(1, n_test_iid)

    # Keep at least one train example whenever there are in-domain candidates.
    while n_dev + n_test_iid >= total and n_test_iid > 0:
        n_test_iid -= 1
    while n_dev + n_test_iid >= total and n_dev > 0:
        n_dev -= 1

    dev_ids = shuffled[:n_dev]
    test_iid_ids = shuffled[n_dev : n_dev + n_test_iid]
    train_ids = shuffled[n_dev + n_test_iid :]
    return {
        "train": train_ids,
        "dev": dev_ids,
        "test_iid": test_iid_ids,
    }


def _collect_unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values})


def build_split_manifest(
    instances: Iterable[ClaimInstance | dict[str, Any]],
    *,
    dataset_name: str = "causal_oversight_benchmark",
    version: str = "v1",
    family_holdout: Iterable[str] | None = None,
    lexical_holdout: Iterable[str] | None = None,
    variable_renaming_holdout: bool | None = None,
    dev_ratio: float = 0.2,
    test_iid_ratio: float = 0.2,
    seed: int = 0,
) -> BenchmarkSplitManifest:
    """Build a deterministic manifest with IID and OOD splits."""

    normalized_instances = _normalize_instances(instances)
    if not normalized_instances:
        raise ValueError("Cannot build split manifest from an empty benchmark instance set.")

    families = _collect_unique(instance.graph_family for instance in normalized_instances)
    templates = _collect_unique(instance.language_template_id for instance in normalized_instances)
    renamed_available = any(_is_variable_renamed(instance) for instance in normalized_instances)

    selected_family_holdout = _collect_unique(family_holdout or [])
    selected_lexical_holdout = _collect_unique(lexical_holdout or [])
    selected_variable_renaming_holdout = (
        renamed_available if variable_renaming_holdout is None else bool(variable_renaming_holdout)
    )

    if not selected_family_holdout and len(families) > 1:
        selected_family_holdout = [families[-1]]
    if not selected_lexical_holdout and len(templates) > 1:
        selected_lexical_holdout = [templates[-1]]

    ood_reasons: dict[str, list[str]] = {}
    for instance in normalized_instances:
        reasons: list[str] = []
        if instance.graph_family in selected_family_holdout:
            reasons.append("family_holdout")
        if instance.language_template_id in selected_lexical_holdout:
            reasons.append("lexical_holdout")
        if selected_variable_renaming_holdout and _is_variable_renamed(instance):
            reasons.append("variable_renaming_holdout")
        if reasons:
            ood_reasons[instance.instance_id] = reasons

    if not ood_reasons:
        raise ValueError(
            "Unable to produce a test_ood split from the provided instances and holdout settings."
        )

    test_ood_ids = sorted(ood_reasons)
    in_domain_ids = sorted(
        instance.instance_id
        for instance in normalized_instances
        if instance.instance_id not in ood_reasons
    )
    split_map = _slice_in_domain_ids(
        in_domain_ids,
        seed=seed,
        dev_ratio=dev_ratio,
        test_iid_ratio=test_iid_ratio,
    )

    metadata = {
        "builder": "split_builder_v1",
        "seed": int(seed),
        "n_instances": len(normalized_instances),
        "split_counts": {
            "train": len(split_map["train"]),
            "dev": len(split_map["dev"]),
            "test_iid": len(split_map["test_iid"]),
            "test_ood": len(test_ood_ids),
        },
        "ood_reasons": ood_reasons,
    }

    return BenchmarkSplitManifest(
        dataset_name=dataset_name,
        version=version,
        train=split_map["train"],
        dev=split_map["dev"],
        test_iid=split_map["test_iid"],
        test_ood=test_ood_ids,
        family_holdout=selected_family_holdout,
        lexical_holdout=selected_lexical_holdout,
        variable_renaming_holdout=selected_variable_renaming_holdout,
        metadata=metadata,
    )


def build_benchmark_splits(
    instances: Iterable[ClaimInstance | dict[str, Any]],
    **kwargs: Any,
) -> BenchmarkSplitManifest:
    """Alias kept for readability in benchmark runner call sites."""

    return build_split_manifest(instances, **kwargs)
