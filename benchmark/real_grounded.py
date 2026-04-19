"""Schema contract for literature-grounded and semi-real benchmark cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from benchmark.schema import ClaimInstance, ensure_claim_instance


def _normalize_str_list(values: list[Any] | tuple[Any, ...] | None) -> list[str]:
    if values is None:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _require_non_empty_string(value: Any, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


class GroundingType(str, Enum):
    """Supported real-grounded subset provenance types."""

    LITERATURE_GROUNDED = "literature_grounded"
    SEMI_REAL = "semi_real"


GROUNDING_TYPE_SPACE: tuple[str, ...] = tuple(item.value for item in GroundingType)


def _coerce_grounding_type(value: GroundingType | str) -> GroundingType:
    if isinstance(value, GroundingType):
        return value
    try:
        return GroundingType(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"grounding_type must be one of {GROUNDING_TYPE_SPACE}, got {value!r}."
        ) from exc


@dataclass(slots=True)
class SourceCitation:
    """Bibliographic source metadata for one real-grounded case."""

    citation_text: str
    title: str = ""
    source_id: str = ""
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    year: int | None = None
    url: str | None = None
    doi: str | None = None

    def __post_init__(self) -> None:
        self.citation_text = _require_non_empty_string(
            self.citation_text,
            field_name="source_citation.citation_text",
        )
        self.title = str(self.title).strip()
        self.source_id = str(self.source_id).strip()
        self.authors = _normalize_str_list(self.authors)
        self.venue = str(self.venue).strip()
        self.year = None if self.year is None else int(self.year)
        self.url = _coerce_optional_string(self.url)
        self.doi = _coerce_optional_string(self.doi)

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation_text": self.citation_text,
            "title": self.title,
            "source_id": self.source_id,
            "authors": list(self.authors),
            "venue": self.venue,
            "year": self.year,
            "url": self.url,
            "doi": self.doi,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SourceCitation":
        return cls(
            citation_text=payload.get("citation_text", payload.get("citation", "")),
            title=str(payload.get("title", "")),
            source_id=str(payload.get("source_id", payload.get("id", ""))),
            authors=list(payload.get("authors", [])),
            venue=str(payload.get("venue", "")),
            year=payload.get("year"),
            url=payload.get("url"),
            doi=payload.get("doi"),
        )


def _normalize_source_citation(value: SourceCitation | dict[str, Any] | None) -> SourceCitation:
    if value is None:
        raise ValueError("source_citation is required for RealGroundedCase.")
    if isinstance(value, SourceCitation):
        return SourceCitation.from_dict(value.to_dict())
    if isinstance(value, dict):
        return SourceCitation.from_dict(value)
    raise TypeError(f"Unsupported source_citation type: {type(value)!r}.")


@dataclass(slots=True)
class InformationContract:
    """Visible-versus-hidden information contract for a real-grounded case."""

    visible_information: list[str] = field(default_factory=list)
    hidden_information: list[str] = field(default_factory=list)
    note: str = ""

    def __post_init__(self) -> None:
        self.visible_information = _normalize_str_list(self.visible_information)
        self.hidden_information = _normalize_str_list(self.hidden_information)
        self.note = str(self.note).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "visible_information": list(self.visible_information),
            "hidden_information": list(self.hidden_information),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InformationContract":
        return cls(
            visible_information=list(payload.get("visible_information", [])),
            hidden_information=list(payload.get("hidden_information", [])),
            note=str(payload.get("note", "")),
        )


def _normalize_information_contract(
    value: InformationContract | dict[str, Any] | None,
) -> InformationContract:
    if value is None:
        return InformationContract()
    if isinstance(value, InformationContract):
        return InformationContract.from_dict(value.to_dict())
    if isinstance(value, dict):
        return InformationContract.from_dict(value)
    raise TypeError(f"Unsupported information_contract type: {type(value)!r}.")


@dataclass(slots=True)
class RealGroundedCase:
    """One literature-grounded or semi-real benchmark case."""

    case_id: str
    grounding_type: GroundingType | str
    claim: ClaimInstance | dict[str, Any]
    source_citation: SourceCitation | dict[str, Any]
    public_evidence_summary: str
    information_contract: InformationContract | dict[str, Any] | None = None
    identifying_assumptions: list[str] = field(default_factory=list)
    witness_note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    grounding_type_space: ClassVar[tuple[str, ...]] = GROUNDING_TYPE_SPACE

    def __post_init__(self) -> None:
        self.case_id = _require_non_empty_string(self.case_id, field_name="case_id")
        self.grounding_type = _coerce_grounding_type(self.grounding_type)
        self.claim = ensure_claim_instance(self.claim)
        self.source_citation = _normalize_source_citation(self.source_citation)
        self.public_evidence_summary = _require_non_empty_string(
            self.public_evidence_summary,
            field_name="public_evidence_summary",
        )
        self.information_contract = _normalize_information_contract(self.information_contract)
        self.identifying_assumptions = _normalize_str_list(self.identifying_assumptions)
        self.witness_note = _require_non_empty_string(self.witness_note, field_name="witness_note")
        self.metadata = dict(self.metadata)

        claim_meta = dict(self.claim.meta)
        claim_meta.setdefault("data_origin", "real_grounded")
        claim_meta.setdefault("grounding_type", self.grounding_type.value)
        claim_meta.setdefault("real_grounded_case_id", self.case_id)
        claim_meta.setdefault("source_citation", self.source_citation.to_dict())
        claim_meta.setdefault("public_evidence_summary", self.public_evidence_summary)
        claim_meta.setdefault("information_contract", self.information_contract.to_dict())
        claim_meta.setdefault("witness_note", self.witness_note)
        self.claim.meta = claim_meta

        merged_assumptions = [
            *self.claim.gold_assumptions,
            *[
                item
                for item in self.identifying_assumptions
                if item not in set(self.claim.gold_assumptions)
            ],
        ]
        self.claim.gold_assumptions = merged_assumptions

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "grounding_type": self.grounding_type.value,
            "claim": self.claim.to_dict(),
            "source_citation": self.source_citation.to_dict(),
            "public_evidence_summary": self.public_evidence_summary,
            "information_contract": self.information_contract.to_dict(),
            "identifying_assumptions": list(self.identifying_assumptions),
            "witness_note": self.witness_note,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RealGroundedCase":
        if "source_citation" not in payload:
            raise ValueError("RealGroundedCase payload must include source_citation.")
        return cls(
            case_id=payload.get("case_id", payload.get("id", "")),
            grounding_type=payload.get("grounding_type", "literature_grounded"),
            claim=payload.get("claim"),
            source_citation=payload.get("source_citation"),
            public_evidence_summary=payload.get("public_evidence_summary", ""),
            information_contract=payload.get("information_contract"),
            identifying_assumptions=list(payload.get("identifying_assumptions", [])),
            witness_note=payload.get("witness_note", ""),
            metadata=dict(payload.get("metadata", {})),
        )


def ensure_real_grounded_case(value: RealGroundedCase | dict[str, Any]) -> RealGroundedCase:
    if isinstance(value, RealGroundedCase):
        return RealGroundedCase.from_dict(value.to_dict())
    if isinstance(value, dict):
        return RealGroundedCase.from_dict(value)
    raise TypeError(f"Unsupported real-grounded case type: {type(value)!r}.")


@dataclass(slots=True)
class RealGroundedDataset:
    """Serializable dataset wrapper for the real-grounded subset."""

    dataset_name: str = "real_grounded_subset"
    version: str = "v1"
    cases: list[RealGroundedCase | dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dataset_name = _require_non_empty_string(self.dataset_name, field_name="dataset_name")
        self.version = _require_non_empty_string(self.version, field_name="version")
        self.cases = [ensure_real_grounded_case(case) for case in self.cases]
        self.metadata = dict(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "version": self.version,
            "cases": [case.to_dict() for case in self.cases],
            "metadata": dict(self.metadata),
        }

    def claim_instances(self) -> list[ClaimInstance]:
        return [ensure_claim_instance(case.claim) for case in self.cases]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RealGroundedDataset":
        return cls(
            dataset_name=payload.get("dataset_name", "real_grounded_subset"),
            version=payload.get("version", "v1"),
            cases=list(payload.get("cases", [])),
            metadata=dict(payload.get("metadata", {})),
        )


def ensure_real_grounded_dataset(
    value: RealGroundedDataset | dict[str, Any] | list[dict[str, Any]] | list[RealGroundedCase],
) -> RealGroundedDataset:
    if isinstance(value, RealGroundedDataset):
        return RealGroundedDataset.from_dict(value.to_dict())
    if isinstance(value, dict):
        if "cases" in value:
            return RealGroundedDataset.from_dict(value)
        return RealGroundedDataset(cases=[value])
    if isinstance(value, list):
        return RealGroundedDataset(cases=list(value))
    raise TypeError(f"Unsupported real-grounded dataset type: {type(value)!r}.")
