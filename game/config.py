"""Configuration loading helpers for the game stack."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class ConfigLoader:
    """Lightweight YAML config loader with section access."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> None:
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.overrides = overrides or {}
        self._config: dict[str, Any] | None = None

    def load(self, force: bool = False) -> dict[str, Any]:
        if self._config is not None and not force:
            return deepcopy(self._config)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        if not isinstance(data, dict):
            raise ValueError(f"Config file must contain a mapping: {self.config_path}")

        if self.overrides:
            data = _deep_merge(data, self.overrides)

        self._config = data
        return deepcopy(data)

    def get(self, section: str, default: Any | None = None) -> Any:
        config = self.load()
        return deepcopy(config.get(section, default))

    def require(self, section: str) -> Any:
        config = self.load()
        if section not in config:
            raise KeyError(f"Missing required config section: {section}")
        return deepcopy(config[section])

    def as_dict(self) -> dict[str, Any]:
        return self.load()


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience function for one-shot config loading."""

    return ConfigLoader(config_path=config_path, overrides=overrides).load()

