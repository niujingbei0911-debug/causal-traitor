"""Shared game-side data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class GamePhase(Enum):
    """Lifecycle stages for a single debate round."""

    SETUP = "setup"
    CLAIM = "claim"
    CHALLENGE = "challenge"
    REBUTTAL = "rebuttal"
    AUDIT = "audit"
    JURY = "jury"
    COMPLETE = "complete"


@dataclass(slots=True)
class CausalScenario:
    """A single causal reasoning scenario for one round."""

    scenario_id: str
    description: str
    true_dag: dict[str, list[str]]
    variables: list[str]
    hidden_variables: list[str]
    ground_truth: dict[str, Any] = field(default_factory=dict)
    observed_data: pd.DataFrame | None = None
    full_data: pd.DataFrame | None = None
    data: pd.DataFrame | None = None
    causal_level: int = 1
    difficulty: float = 0.5
    difficulty_config: dict[str, Any] = field(default_factory=dict)
    true_scm: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.observed_data is None and self.data is not None:
            self.observed_data = self.data.copy()
        if self.observed_data is None:
            self.observed_data = pd.DataFrame()
        if self.full_data is None:
            self.full_data = self.observed_data.copy()
        if self.data is None:
            self.data = self.observed_data
        if self.true_scm is None:
            self.true_scm = {"graph": self.true_dag}


@dataclass(slots=True)
class DebateTurn:
    """One utterance or system event in the debate transcript."""

    speaker: str
    phase: GamePhase
    content: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "speaker":
            return self.speaker
        if key == "phase":
            return self.phase.value
        if key == "content":
            return self.content
        if key == "metadata":
            return self.metadata
        return self.metadata.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "phase": self.phase.value,
            "content": self.content,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class DebateContext:
    """Mutable context carried across one round."""

    scenario: CausalScenario
    round_number: int = 0
    turns: list[DebateTurn] = field(default_factory=list)
    current_phase: GamePhase = GamePhase.SETUP
    evolution_context: dict[str, Any] | None = None
    jury_verdict: dict[str, Any] | None = None
    jury_result: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: list[DebateTurn] = []
        for turn in self.turns:
            if isinstance(turn, DebateTurn):
                normalized.append(turn)
                continue
            if isinstance(turn, dict):
                phase_value = turn.get("phase", self.current_phase.value)
                phase = phase_value if isinstance(phase_value, GamePhase) else GamePhase(str(phase_value))
                normalized.append(
                    DebateTurn(
                        speaker=str(turn.get("speaker", "unknown")),
                        phase=phase,
                        content=str(turn.get("content", "")),
                        metadata=dict(turn.get("metadata", {})),
                    )
                )
                continue
            raise TypeError(f"Unsupported turn type: {type(turn)!r}")
        self.turns = normalized
