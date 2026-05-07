import os
import tempfile
import unittest
from pathlib import Path
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
