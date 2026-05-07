"""Model backend adapter for the debate agents.

Supported backends:

* ``mock``      — deterministic, no network; used for tests and offline runs.
* ``dashscope`` / ``api`` — Alibaba Bailian (Qwen) via the OpenAI-compatible
  endpoint ``https://dashscope.aliyuncs.com/compatible-mode/v1``. Reads the
  API key from the ``DASHSCOPE_API_KEY`` environment variable (falling back
  to ``OPENAI_API_KEY`` / an explicit ``api_key`` in the config).
* ``vllm`` / ``ollama`` — kept as declared options, but currently route to
  the mock fallback until a local server is connected.

The service never embeds a real API key in code or config defaults. Offline
agents may opt into mock fallback for local demos, but evaluation runners can
disable fallback so missing credentials or backend failures are surfaced as
errors rather than being recorded as model evidence.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any

try:
    from openai import AsyncOpenAI  # type: ignore
except ImportError:  # pragma: no cover - openai is declared in requirements.txt
    AsyncOpenAI = None  # type: ignore

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is declared in requirements.txt
    find_dotenv = None  # type: ignore
    load_dotenv = None  # type: ignore


DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
_DOTENV_LOADED_PATHS: set[str] = set()


def _looks_like_api_key(value: Any) -> bool:
    """Reject placeholders like '在这里粘贴你的API_KEY' that would break header encoding."""
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if not candidate.startswith("sk-"):
        return False
    try:
        candidate.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True

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
        self.api_key_env = config.get("api_key_env")
        self.api_mode = str(config.get("api_mode") or "chat_completions").lower()
        self.reasoning_effort = config.get("reasoning_effort")
        self._explicit_api_key = config.get("api_key")  # allow override, but prefer env var
        self._client: Any | None = None
        self.initialized = False

    async def initialize(self) -> None:
        if self.backend in {"dashscope", "api", "openai"}:
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

        if self.backend in {"dashscope", "api", "openai"}:
            if self.api_mode == "responses":
                return await self._call_responses(prompt, system_prompt, used_temperature, used_max_tokens)
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

    async def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[LLMResponse, dict[str, Any] | None]:
        """Generate text and parse the first JSON object found in the response."""

        response = await self.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response, self.extract_json_object(response.text)

    async def close(self) -> None:
        """Close the underlying async API client when one was created."""
        client = self._client
        self._client = None
        if client is None:
            return
        close = getattr(client, "close", None)
        if close is None:
            return
        result = close()
        if asyncio.iscoroutine(result):
            await result

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
            if not self.allow_mock_fallback:
                raise RuntimeError(
                    "DASHSCOPE_API_KEY missing or openai SDK unavailable; mock fallback is disabled."
                )
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
        timeout_sec = float(self.config.get("timeout", 60))
        request: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.reasoning_effort:
            request["reasoning_effort"] = str(self.reasoning_effort)
        thinking = self.config.get("thinking")
        if thinking:
            request["extra_body"] = {"thinking": {"type": str(thinking)}}

        try:
            completion = await asyncio.wait_for(
                self._client.chat.completions.create(**request),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            if not self.allow_mock_fallback:
                raise
            return self._mock_response(
                prompt,
                temperature,
                max_tokens,
                note=f"DashScope call timed out after {timeout_sec}s for model {model_id}; using mock fallback.",
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
            "api_mode": "chat_completions",
            "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
        }
        if self.reasoning_effort:
            metadata["reasoning_effort"] = str(self.reasoning_effort)
        if thinking:
            metadata["thinking"] = str(thinking)
        if usage is not None:
            metadata["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        return LLMResponse(
            text=text,
            backend=self.backend,
            model_name=model_id,
            metadata=metadata,
        )

    async def _call_responses(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        if self._client is None:
            if not self.allow_mock_fallback:
                raise RuntimeError(
                    "API key missing or openai SDK unavailable; mock fallback is disabled."
                )
            return self._mock_response(
                prompt,
                temperature,
                max_tokens,
                note="API key missing or openai SDK unavailable; using mock fallback.",
            )

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        model_id = _MODEL_ALIAS.get(self.model_name.lower(), self.model_name)
        timeout_sec = float(self.config.get("timeout", 60))
        request: dict[str, Any] = {
            "model": model_id,
            "input": messages,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if self.reasoning_effort:
            request["reasoning"] = {"effort": str(self.reasoning_effort)}
        try:
            completion = await asyncio.wait_for(
                self._client.responses.create(**request),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            if not self.allow_mock_fallback:
                raise
            return self._mock_response(
                prompt,
                temperature,
                max_tokens,
                note=f"Responses API call timed out after {timeout_sec}s for model {model_id}; using mock fallback.",
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self.allow_mock_fallback:
                raise
            return self._mock_response(
                prompt,
                temperature,
                max_tokens,
                note=f"Responses API call failed ({type(exc).__name__}): {exc}; using mock fallback.",
            )

        text = self._extract_responses_text(completion)
        usage = getattr(completion, "usage", None)
        metadata: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_mode": "responses",
        }
        if self.reasoning_effort:
            metadata["reasoning_effort"] = str(self.reasoning_effort)
        if usage is not None:
            metadata["usage"] = {
                "prompt_tokens": getattr(usage, "input_tokens", None),
                "completion_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        return LLMResponse(
            text=text,
            backend=self.backend,
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

    @staticmethod
    def _extract_responses_text(completion: Any) -> str:
        output_text = getattr(completion, "output_text", None)
        if output_text:
            return str(output_text)

        chunks: list[str] = []
        for item in getattr(completion, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(str(text))
        return "\n".join(chunks)

    @staticmethod
    def extract_json_object(text: str | None) -> dict[str, Any] | None:
        """Extract the first valid JSON object from a model response."""

        if not text:
            return None

        candidates: list[str] = []
        stripped = text.strip()
        candidates.append(stripped)

        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", stripped, flags=re.IGNORECASE)
        candidates.extend(block.strip() for block in fenced if block.strip())

        first = stripped.find("{")
        last = stripped.rfind("}")
        if first != -1 and last != -1 and first < last:
            candidates.append(stripped[first : last + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    # No hardcoded API keys. Callers should provide a valid env var or explicit
    # config key; otherwise the service falls back to mock mode.
    _DEFAULT_DASHSCOPE_API_KEY: str | None = None

    @staticmethod
    def _load_dotenv_if_available() -> None:
        if find_dotenv is None or load_dotenv is None:
            return
        dotenv_path = find_dotenv(usecwd=True)
        if not dotenv_path or dotenv_path in _DOTENV_LOADED_PATHS:
            return
        load_dotenv(dotenv_path, override=False)
        _DOTENV_LOADED_PATHS.add(dotenv_path)

    def _resolve_api_key(self) -> str | None:
        self._load_dotenv_if_available()
        if self.api_key_env:
            value = os.environ.get(str(self.api_key_env))
            return value if value and _looks_like_api_key(value) else None
        for env_name in ("DASHSCOPE_API_KEY", "OPENAI_API_KEY"):
            value = os.environ.get(env_name)
            if value and _looks_like_api_key(value):
                return value
        if self._explicit_api_key and _looks_like_api_key(self._explicit_api_key):
            return str(self._explicit_api_key)
        return self._DEFAULT_DASHSCOPE_API_KEY or None
