"""Shared game-side data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class GamePhase(Enum):
    """Lifecycle stages for a single debate round."""

    SETUP = "setup"
    CLAIM = "claim"
    CHALLENGE = "challenge"
    REBUTTAL = "rebuttal"
    AUDIT = "audit"
    JURY = "jury"
    COMPLETE = "complete"

from benchmark.schema import (
    GoldCausalInstance,
    PublicCausalInstance,
    VerifierScenario,
    VerifierVerdict,
    VerdictLabel,
)

# ``CausalScenario`` is retained only as a legacy constructor/export name for
# the attacker-side gold view. Verifier-side runtime paths should use
# ``VerifierScenario`` / ``PublicCausalInstance`` instead.
# Legacy game code still imports and instantiates ``CausalScenario`` directly.
# During the schema transition, that name continues to mean the gold view.
# A debate/game ``winner`` is still a protocol artifact, but the paper-level
# supervision target is ``gold_label`` / ``verdict.label`` in the frozen
# ``valid`` / ``invalid`` / ``unidentifiable`` space.
CausalScenario = GoldCausalInstance


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

    scenario: VerifierScenario
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
