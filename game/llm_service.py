"""Model backend adapter for the debate agents.

Supported backends:

* ``mock``      — deterministic, no network; used for tests and offline runs.
* ``dashscope`` / ``api`` — Alibaba Bailian (Qwen) via the OpenAI-compatible
  endpoint ``https://dashscope.aliyuncs.com/compatible-mode/v1``. Reads the
  API key from the ``DASHSCOPE_API_KEY`` environment variable (falling back
  to ``OPENAI_API_KEY`` / an explicit ``api_key`` in the config).
* ``vllm`` / ``ollama`` — kept as declared options, but currently route to
  the mock fallback until a local server is connected.

The service never embeds a real API key in code or config defaults. If the
env variable is missing, requests silently degrade to the mock response so
the rest of the pipeline keeps running.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

try:
    from openai import AsyncOpenAI  # type: ignore
except ImportError:  # pragma: no cover - openai is declared in requirements.txt
    AsyncOpenAI = None  # type: ignore


DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Friendly aliases → DashScope model ids.
_MODEL_ALIAS = {
    "qwen/qwen2.5-7b-instruct": "qwen2.5-7b-instruct",
    "qwen/qwen2.5-14b-instruct": "qwen2.5-14b-instruct",
    "qwen/qwen2.5-32b-instruct": "qwen2.5-32b-instruct",
    "qwen/qwen2.5-72b-instruct": "qwen2.5-72b-instruct",
}


@dataclass(slots=True)
class LLMResponse:
    text: str
    backend: str
    model_name: str
    metadata: dict[str, Any]


class LLMService:
    """Unified interface for mock, DashScope (Qwen), or local vLLM/Ollama."""

    def __init__(self, config: dict[str, Any], allow_mock_fallback: bool = True) -> None:
        self.config = config
        self.allow_mock_fallback = allow_mock_fallback
        self.backend = (config.get("backend") or "mock").lower()
        self.model_name = config.get("name", "mock-model")
        self.temperature = float(config.get("temperature", 0.3))
        self.max_tokens = int(config.get("max_tokens", 512))
        self.base_url = config.get("base_url") or DASHSCOPE_BASE_URL
        self._explicit_api_key = config.get("api_key")  # allow override, but prefer env var
        self._client: Any | None = None
        self.initialized = False

    async def initialize(self) -> None:
        if self.backend in {"dashscope", "api"}:
            api_key = self._resolve_api_key()
            if api_key and AsyncOpenAI is not None:
                self._client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)
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

        used_temperature = self.temperature if temperature is None else float(temperature)
        used_max_tokens = self.max_tokens if max_tokens is None else int(max_tokens)

        if self.backend in {"dashscope", "api"}:
            return await self._call_dashscope(prompt, system_prompt, used_temperature, used_max_tokens)

        if self.backend in {"vllm", "ollama"}:
            return self._mock_response(
                prompt,
                used_temperature,
                used_max_tokens,
                note=f"{self.backend} backend not wired; using mock fallback.",
            )

        if self.backend == "mock":
            return self._mock_response(prompt, used_temperature, used_max_tokens)

        if self.allow_mock_fallback:
            return self._mock_response(
                prompt,
                used_temperature,
                used_max_tokens,
                note=f"Unknown backend '{self.backend}', using mock fallback.",
            )

        raise ValueError(f"Unsupported backend: {self.backend}")

    # ------------------------------------------------------------------
    # backends
    # ------------------------------------------------------------------

    async def _call_dashscope(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        if self._client is None:
            return self._mock_response(
                prompt,
                temperature,
                max_tokens,
                note="DASHSCOPE_API_KEY missing or openai SDK unavailable; using mock fallback.",
            )

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        model_id = _MODEL_ALIAS.get(self.model_name.lower(), self.model_name)
        try:
            completion = await self._client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # network / auth / quota errors
            if not self.allow_mock_fallback:
                raise
            return self._mock_response(
                prompt,
                temperature,
                max_tokens,
                note=f"DashScope call failed ({type(exc).__name__}): {exc}; using mock fallback.",
            )

        choice = completion.choices[0] if completion.choices else None
        text = (choice.message.content if choice and choice.message else "") or ""
        usage = getattr(completion, "usage", None)
        metadata: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
        }
        if usage is not None:
            metadata["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        return LLMResponse(
            text=text,
            backend="dashscope",
            model_name=model_id,
            metadata=metadata,
        )

    def _mock_response(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        note: str | None = None,
    ) -> LLMResponse:
        snippet = " ".join(prompt.strip().split())[: max(40, min(140, max_tokens // 8))]
        text = f"[mock:{self.model_name}] {snippet}"
        metadata: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if note:
            metadata["note"] = note
        return LLMResponse(
            text=text,
            backend="mock" if self.backend not in {"vllm", "ollama"} else self.backend,
            model_name=self.model_name,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    # No hardcoded key — set DASHSCOPE_API_KEY / OPENAI_API_KEY env var,
    # or provide api_key in configs/default.yaml → llm section.
    _DEFAULT_DASHSCOPE_API_KEY: str | None = None

    def _resolve_api_key(self) -> str | None:
        if self._explicit_api_key:
            return str(self._explicit_api_key)
        for env_name in ("DASHSCOPE_API_KEY", "OPENAI_API_KEY"):
            value = os.environ.get(env_name)
            if value:
                return value
        return self._DEFAULT_DASHSCOPE_API_KEY or None
