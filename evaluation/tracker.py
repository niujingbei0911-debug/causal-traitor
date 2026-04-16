"""Experiment tracking utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for one tracked run."""

    experiment_id: str
    name: str
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class ExperimentTracker:
    """
    Track experiment metrics, round logs, and artifacts.

    This class intentionally stays lightweight. Statistical reporting is kept in
    the dedicated evaluation/reporting.py module rather than being hard-wired
    here.
    """

    def __init__(self, config: ExperimentConfig, log_dir: str = "logs", use_wandb: bool = False):
        self.config = config
        self.log_dir = Path(log_dir)
        self.use_wandb = use_wandb
        self.metrics_history: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        self.run_dir: Path | None = None
        self.metrics_path: Path | None = None
        self.rounds_path: Path | None = None
        self.artifacts_dir: Path | None = None
        self._initialized = False

    def init(self) -> None:
        """Create run directories and initialize on-disk logs."""

        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.config.experiment_id}_{timestamp}"
        self.run_dir = self.log_dir / run_name
        self.artifacts_dir = self.run_dir / "artifacts"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        (self.run_dir / "config.json").write_text(
            json.dumps(self._json_ready(asdict(self.config)), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.rounds_path = self.run_dir / "rounds.jsonl"
        self._initialized = True

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log one metric event."""

        self._ensure_initialized()
        record = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "metrics": self._json_ready(metrics),
        }
        self.metrics_history.append(record)
        self._append_jsonl(self.metrics_path, record)

    def log_artifact(self, name: str, data: Any, artifact_type: str = "json") -> None:
        """Persist an artifact payload."""

        self._ensure_initialized()
        safe_name = name.replace("/", "_")
        target = self.artifacts_dir / f"{safe_name}.{artifact_type}"

        if artifact_type == "json":
            target.write_text(
                json.dumps(self._json_ready(data), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return

        if artifact_type in {"txt", "md"}:
            target.write_text(str(data), encoding="utf-8")
            return

        target.write_text(
            json.dumps({"data": self._json_ready(data)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def log_round(self, round_id: int, round_data: Dict[str, Any]) -> None:
        """Log one round payload."""

        self._ensure_initialized()
        record = {
            "timestamp": datetime.now().isoformat(),
            "round_id": round_id,
            "round_data": self._json_ready(round_data),
        }
        self._append_jsonl(self.rounds_path, record)

    def save_checkpoint(self, game_state: Dict[str, Any]) -> str:
        """Persist a checkpoint and return its path."""

        self._ensure_initialized()
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        target = checkpoint_dir / f"checkpoint_{datetime.now().strftime('%H%M%S')}.json"
        target.write_text(
            json.dumps(self._json_ready(game_state), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(target)

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load a saved checkpoint."""

        path = Path(checkpoint_path)
        return json.loads(path.read_text(encoding="utf-8"))

    def finish(self) -> Dict[str, Any]:
        """Finalize the run and emit a summary."""

        self._ensure_initialized()
        end_time = datetime.now()
        summary = {
            "experiment_id": self.config.experiment_id,
            "name": self.config.name,
            "started_at": self.start_time.isoformat(),
            "finished_at": end_time.isoformat(),
            "duration_seconds": (end_time - self.start_time).total_seconds(),
            "n_metric_events": len(self.metrics_history),
            "run_dir": str(self.run_dir),
            "use_wandb": self.use_wandb,
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.init()

    def _append_jsonl(self, path: Path | None, record: Dict[str, Any]) -> None:
        if path is None:
            raise RuntimeError("Tracker path not initialized")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _json_ready(self, value: Any) -> Any:
        if is_dataclass(value):
            return self._json_ready(asdict(value))
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: self._json_ready(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_ready(item) for item in value]
        try:
            import pandas as pd

            if isinstance(value, pd.DataFrame):
                return {
                    "__type__": "DataFrame",
                    "shape": list(value.shape),
                    "columns": list(value.columns),
                    "head": value.head(5).to_dict(orient="records"),
                }
            if isinstance(value, pd.Series):
                return value.to_dict()
        except ImportError:
            pass
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
        except ImportError:
            pass
        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                return self._json_ready(value.to_dict())
            except Exception:
                return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
