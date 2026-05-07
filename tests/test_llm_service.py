import os
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from game.llm_service import LLMService


class LLMServiceApiKeyTests(unittest.TestCase):
    def test_resolve_api_key_loads_dotenv_from_current_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, ".env").write_text(
                "DASHSCOPE_API_KEY=sk-test-dotenv-key\n",
                encoding="ascii",
            )
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                with patch.dict(os.environ, {}, clear=True):
                    service = LLMService({"backend": "dashscope", "name": "qwen2.5-7b-instruct"})

                    self.assertEqual(service._resolve_api_key(), "sk-test-dotenv-key")
            finally:
                os.chdir(cwd)

    def test_resolve_api_key_keeps_existing_environment_key_over_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, ".env").write_text(
                "DASHSCOPE_API_KEY=sk-dotenv-key\n",
                encoding="ascii",
            )
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-env-key"}, clear=True):
                    service = LLMService({"backend": "dashscope", "name": "qwen2.5-7b-instruct"})

                    self.assertEqual(service._resolve_api_key(), "sk-env-key")
            finally:
                os.chdir(cwd)

    def test_resolve_api_key_prefers_environment_over_explicit_config_key(self) -> None:
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-env-key"}, clear=True):
            service = LLMService(
                {
                    "backend": "dashscope",
                    "name": "qwen2.5-7b-instruct",
                    "api_key": "sk-config-key",
                }
            )

            self.assertEqual(service._resolve_api_key(), "sk-env-key")

    def test_resolve_api_key_honors_configured_env_name(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DASHSCOPE_API_KEY": "sk-dashscope-key",
                "OPENAI_API_KEY": "sk-openai-key",
            },
            clear=True,
        ):
            service = LLMService(
                {
                    "backend": "api",
                    "name": "gpt-5.5",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "OPENAI_API_KEY",
                }
            )

            self.assertEqual(service._resolve_api_key(), "sk-openai-key")

    def test_resolve_api_key_does_not_fallback_when_configured_env_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-dashscope-key"}, clear=True):
                    service = LLMService(
                        {
                            "backend": "api",
                            "name": "gpt-5.5",
                            "base_url": "https://api.openai.com/v1",
                            "api_key_env": "OPENAI_API_KEY",
                        }
                    )

                    self.assertIsNone(service._resolve_api_key())
            finally:
                os.chdir(cwd)

    def test_dashscope_missing_key_raises_when_mock_fallback_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                with patch.dict(os.environ, {}, clear=True):
                    service = LLMService(
                        {"backend": "dashscope", "name": "qwen2.5-7b-instruct"},
                        allow_mock_fallback=False,
                    )

                    with self.assertRaisesRegex(RuntimeError, "DASHSCOPE_API_KEY"):
                        asyncio.run(service.generate("Use only public evidence."))
            finally:
                os.chdir(cwd)

    def test_generate_uses_chat_completions_reasoning_options_when_configured(self) -> None:
        class FakeChatCompletionsEndpoint:
            def __init__(self) -> None:
                self.kwargs = None

            async def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content='{"verdict":"valid"}'),
                            finish_reason="stop",
                        )
                    ],
                    usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10),
                )

        endpoint = FakeChatCompletionsEndpoint()
        service = LLMService(
            {
                "backend": "openai",
                "name": "deepseek-v4-pro",
                "base_url": "https://api.deepseek.com",
                "api_key_env": "CUSTOM_API_KEY",
                "api_mode": "chat_completions",
                "reasoning_effort": "max",
                "thinking": "enabled",
            },
            allow_mock_fallback=False,
        )
        service.initialized = True
        service._client = SimpleNamespace(
            chat=SimpleNamespace(completions=endpoint)
        )

        response = asyncio.run(
            service.generate(
                "Use only public evidence.",
                system_prompt="You are a careful causal verifier.",
                max_tokens=64,
            )
        )

        self.assertEqual(response.text, '{"verdict":"valid"}')
        self.assertEqual(response.backend, "openai")
        self.assertEqual(response.model_name, "deepseek-v4-pro")
        self.assertEqual(endpoint.kwargs["model"], "deepseek-v4-pro")
        self.assertEqual(endpoint.kwargs["max_tokens"], 64)
        self.assertEqual(endpoint.kwargs["reasoning_effort"], "max")
        self.assertEqual(endpoint.kwargs["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertEqual(endpoint.kwargs["messages"][0]["role"], "system")
        self.assertEqual(response.metadata["api_mode"], "chat_completions")
        self.assertEqual(response.metadata["reasoning_effort"], "max")
        self.assertEqual(response.metadata["thinking"], "enabled")

    def test_generate_uses_responses_api_when_configured(self) -> None:
        class FakeResponsesEndpoint:
            def __init__(self) -> None:
                self.kwargs = None

            async def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    output_text='{"verdict":"valid"}',
                    usage=SimpleNamespace(input_tokens=11, output_tokens=5, total_tokens=16),
                )

        endpoint = FakeResponsesEndpoint()
        service = LLMService(
            {
                "backend": "openai",
                "name": "gpt-5.5",
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
                "api_mode": "responses",
                "reasoning_effort": "high",
            },
            allow_mock_fallback=False,
        )
        service.initialized = True
        service._client = SimpleNamespace(responses=endpoint)

        response = asyncio.run(
            service.generate(
                "Use only public evidence.",
                system_prompt="You are a careful causal verifier.",
                max_tokens=64,
            )
        )

        self.assertEqual(response.text, '{"verdict":"valid"}')
        self.assertEqual(response.backend, "openai")
        self.assertEqual(response.model_name, "gpt-5.5")
        self.assertEqual(endpoint.kwargs["model"], "gpt-5.5")
        self.assertEqual(endpoint.kwargs["max_output_tokens"], 64)
        self.assertEqual(endpoint.kwargs["reasoning"], {"effort": "high"})
        self.assertEqual(endpoint.kwargs["input"][0]["role"], "system")
