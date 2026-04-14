"""Model backend adapter with a mock-friendly default path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class LLMResponse:
    text: str
    backend: str
    model_name: str
    metadata: dict[str, Any]


class LLMService:
    """Unified interface for vLLM, Ollama, or a local mock backend."""

    def __init__(self, config: dict[str, Any], allow_mock_fallback: bool = True) -> None:
        self.config = config
        self.allow_mock_fallback = allow_mock_fallback
        self.backend = config.get("backend", "mock")
        self.model_name = config.get("name", "mock-model")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 512)
        self.initialized = False

    async def initialize(self) -> None:
        self.initialized = True

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if not self.initialized:
            await self.initialize()

        del system_prompt
        used_temperature = self.temperature if temperature is None else temperature
        used_max_tokens = self.max_tokens if max_tokens is None else max_tokens

        if self.backend == "mock":
            return self._mock_response(prompt, used_temperature, used_max_tokens)

        if self.backend == "vllm":
            return self._mock_response(
                prompt,
                used_temperature,
                used_max_tokens,
                note="vLLM backend not wired yet; using mock fallback.",
            )

        if self.backend == "ollama":
            return self._mock_response(
                prompt,
                used_temperature,
                used_max_tokens,
                note="Ollama backend not wired yet; using mock fallback.",
            )

        if self.allow_mock_fallback:
            return self._mock_response(
                prompt,
                used_temperature,
                used_max_tokens,
                note=f"Unknown backend '{self.backend}', using mock fallback.",
            )

        raise ValueError(f"Unsupported backend: {self.backend}")

    def _mock_response(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        note: str | None = None,
    ) -> LLMResponse:
        snippet = " ".join(prompt.strip().split())[: max(40, min(140, max_tokens // 8))]
        text = f"[mock:{self.model_name}] {snippet}"
        metadata = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if note:
            metadata["note"] = note
        return LLMResponse(
            text=text,
            backend=self.backend,
            model_name=self.model_name,
            metadata=metadata,
        )
