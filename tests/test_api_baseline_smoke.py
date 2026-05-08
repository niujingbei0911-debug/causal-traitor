import json
from pathlib import Path
from typing import Any

import pytest

from experiments.exp_api_baseline_smoke.run import _build_service, run_api_baseline_smoke
from game.llm_service import LLMResponse


class FakeAPIService:
    backend = "dashscope"
    model_name = "qwen2.5-7b-instruct"
    temperature = 0.0
    max_tokens = 128

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.closed = False

    async def generate_json(self, prompt: str, **_: Any) -> tuple[LLMResponse, dict[str, Any]]:
        self.prompts.append(prompt)
        return (
            LLMResponse(
                text='{"verdict": "unidentifiable", "confidence": 0.77, "reasoning_summary": "public evidence is insufficient"}',
                backend="dashscope",
                model_name=self.model_name,
                metadata={"temperature": 0.0, "max_tokens": 128, "finish_reason": "stop"},
            ),
            {"verdict": "unidentifiable", "confidence": 0.77, "reasoning_summary": "public evidence is insufficient"},
        )

    async def close(self) -> None:
        self.closed = True


class FakeMockService(FakeAPIService):
    backend = "mock"
    model_name = "mock-model"

    async def generate_json(self, prompt: str, **_: Any) -> tuple[LLMResponse, dict[str, Any]]:
        self.prompts.append(prompt)
        return (
            LLMResponse(
                text="[mock:mock-model] {}",
                backend="mock",
                model_name=self.model_name,
                metadata={"note": "using mock fallback"},
            ),
            {},
        )


class FlakyAPIService(FakeAPIService):
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        super().__init__()
        self.fail_on_call = fail_on_call

    async def generate_json(self, prompt: str, **_: Any) -> tuple[LLMResponse, dict[str, Any]]:
        if self.fail_on_call is not None and len(self.prompts) + 1 == self.fail_on_call:
            self.prompts.append(prompt)
            raise RuntimeError("transient API failure")
        return await super().generate_json(prompt, **_)


def test_api_baseline_smoke_records_public_prompt_and_raw_response(tmp_path: Path) -> None:
    service = FakeAPIService()
    output_path = tmp_path / "api_smoke.json"

    payload = run_api_baseline_smoke(
        output_path=output_path,
        service=service,
        seed=0,
        split_name="test_iid",
        max_samples=1,
        generated_at_utc="2026-05-08T00:00:00Z",
    )

    assert output_path.exists()
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == payload
    assert payload["status"] == "api_smoke"
    assert payload["model"]["backend"] == "dashscope"
    assert payload["model"]["name"] == "qwen2.5-7b-instruct"
    assert payload["summary"]["total"] == 1
    assert payload["records"][0]["raw_response"].startswith('{"verdict"')
    assert payload["records"][0]["parsed_payload"]["verdict"] == "unidentifiable"
    assert payload["records"][0]["predicted_label"] == "unidentifiable"
    assert payload["records"][0]["fallback_detected"] is False
    assert payload["records"][0]["prompt_sha256"]
    assert service.closed is True

    prompt = service.prompts[0]
    assert "Public scenario:" in prompt
    assert "gold_label" not in prompt
    assert "gold_answer" not in prompt
    assert "attacker_rationale" not in prompt
    assert "countermodel_witness" not in prompt
    assert "true_scm" not in prompt


def test_api_baseline_smoke_checkpoints_and_resumes_partial_records(tmp_path: Path) -> None:
    output_path = tmp_path / "api_smoke.json"
    failing_service = FlakyAPIService(fail_on_call=2)

    with pytest.raises(RuntimeError, match="transient API failure"):
        run_api_baseline_smoke(
            output_path=output_path,
            service=failing_service,
            seed=0,
            split_name="test_ood",
            max_samples=2,
            checkpoint_records=True,
            resume_existing_records=True,
        )

    partial = json.loads(output_path.read_text(encoding="utf-8"))
    assert partial["complete"] is False
    assert partial["summary"]["total"] == 1
    assert len(partial["records"]) == 1

    resumed_service = FlakyAPIService()
    payload = run_api_baseline_smoke(
        output_path=output_path,
        service=resumed_service,
        seed=0,
        split_name="test_ood",
        max_samples=2,
        checkpoint_records=True,
        resume_existing_records=True,
    )

    assert payload["complete"] is True
    assert payload["summary"]["total"] == 2
    assert len(payload["records"]) == 2
    assert len(resumed_service.prompts) == 1


def test_api_baseline_smoke_rejects_mock_fallback(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="mock fallback"):
        run_api_baseline_smoke(
            output_path=tmp_path / "api_smoke.json",
            service=FakeMockService(),
            seed=0,
            split_name="test_iid",
            max_samples=1,
        )


def test_api_baseline_service_accepts_custom_base_url_and_key_env() -> None:
    service = _build_service(
        model="gpt-5.5",
        backend="api",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        api_mode="responses",
        reasoning_effort="high",
        temperature=0.0,
        max_tokens=256,
        timeout=30,
    )

    assert service.base_url == "https://api.openai.com/v1"
    assert service.config["api_key_env"] == "OPENAI_API_KEY"
    assert service.config["api_mode"] == "responses"
    assert service.config["reasoning_effort"] == "high"
    assert service.allow_mock_fallback is False
