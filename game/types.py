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
    observed_data: pd.DataFrame
    full_data: pd.DataFrame
    ground_truth: dict[str, Any]
    causal_level: int = 1
    difficulty: float = 0.5
    difficulty_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DebateTurn:
    """One utterance or system event in the debate transcript."""

    speaker: str
    phase: GamePhase
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DebateContext:
    """Mutable context carried across one round."""

    scenario: CausalScenario
    round_number: int = 0
    turns: list[DebateTurn] = field(default_factory=list)
    current_phase: GamePhase = GamePhase.SETUP
    evolution_context: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

