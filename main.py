"""CLI entrypoint for running the causal debate game."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from game.config import ConfigLoader
from game.debate_engine import DebateEngine


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


async def run_game(
    *,
    config_path: str | None = None,
    rounds: int | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    loader = ConfigLoader(config_path=config_path)
    config = loader.load()
    engine = DebateEngine(config)
    await engine.initialize()
    num_rounds = rounds or config.get("game", {}).get("max_rounds", 5)
    results = await engine.run_game(num_rounds=num_rounds)
    summary = {
        "n_rounds": len(results),
        "agent_a_wins": sum(result["winner"] == "agent_a" for result in results),
        "agent_b_wins": sum(result["winner"] == "agent_b" for result in results),
        "final_difficulty": engine.difficulty_controller.get_difficulty(),
        "arms_race_index": engine.evolution_tracker.get_arms_race_index(),
    }
    payload = {
        "summary": summary,
        "results": _json_ready(results),
    }

    target = Path(output_path) if output_path else _default_output_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _default_output_path(config: dict[str, Any]) -> Path:
    output_dir = Path(config.get("data", {}).get("output_dir", "outputs"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"game_run_{timestamp}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Causal Traitor debate engine.")
    parser.add_argument("--config", dest="config_path", default=None, help="Path to YAML config file.")
    parser.add_argument("--rounds", type=int, default=None, help="Number of rounds to run.")
    parser.add_argument("--output", dest="output_path", default=None, help="Path to JSON output.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = asyncio.run(
        run_game(
            config_path=args.config_path,
            rounds=args.rounds,
            output_path=args.output_path,
        )
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
