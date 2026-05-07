import json
from pathlib import Path
from typing import Any

from experiments.exp_llm_baseline_matrix.run import run_llm_baseline_matrix


def test_llm_baseline_matrix_runs_enabled_model_seed_split_grid(tmp_path: Path) -> None:
    config_path = tmp_path / "matrix.yaml"
    output_dir = (tmp_path / "out").as_posix()
    config_path.write_text(
        f"""
run:
  output_dir: {output_dir}
  samples_per_family: 10
  difficulty: 0.55
  seeds: [0, 1]
  splits: [test_iid, test_ood]
  max_public_rows: 5
  temperature: 0.0
  max_tokens: 128
  timeout: 30
  reject_mock_fallback: true
models:
  - id: live_model
    enabled: true
    backend: openai
    model: gpt-5.5
    api_mode: responses
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    reasoning_effort: high
  - id: disabled_model
    enabled: false
    backend: dashscope
    model: qwen2.5-7b-instruct
    api_key_env: DASHSCOPE_API_KEY
""",
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []

    def fake_run_one(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": {"total": 2, "correct": 1, "parse_errors": 0, "fallback_records": 0},
            "records": [],
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    manifest = run_llm_baseline_matrix(config_path=config_path, run_one=fake_run_one)

    assert len(calls) == 4
    assert {call["split_name"] for call in calls} == {"test_iid", "test_ood"}
    assert {call["seed"] for call in calls} == {0, 1}
    assert calls[0]["model"] == "gpt-5.5"
    assert calls[0]["api_mode"] == "responses"
    assert calls[0]["reasoning_effort"] == "high"
    assert calls[0]["max_samples"] == 1_000_000
    assert manifest["summary"]["jobs_total"] == 4
    assert manifest["summary"]["succeeded"] == 4
    assert manifest["summary"]["failed"] == 0
    assert Path(manifest["manifest_path"]).exists()


def test_llm_baseline_matrix_probe_overrides_size(tmp_path: Path) -> None:
    config_path = tmp_path / "matrix.yaml"
    output_dir = (tmp_path / "out").as_posix()
    config_path.write_text(
        f"""
run:
  output_dir: {output_dir}
  samples_per_family: 10
  difficulty: 0.55
  seeds: [0, 1]
  splits: [test_iid]
models:
  - id: qwen
    enabled: true
    backend: dashscope
    model: qwen2.5-7b-instruct
    api_key_env: DASHSCOPE_API_KEY
""",
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []

    def fake_run_one(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text(
            json.dumps({"summary": {"total": 1, "correct": 1, "parse_errors": 0, "fallback_records": 0}}),
            encoding="utf-8",
        )
        return {"summary": {"total": 1, "correct": 1, "parse_errors": 0, "fallback_records": 0}}

    run_llm_baseline_matrix(config_path=config_path, probe=True, run_one=fake_run_one)

    assert len(calls) == 1
    assert calls[0]["samples_per_family"] == 1
    assert calls[0]["max_samples"] == 1


def test_llm_baseline_matrix_allows_model_level_runtime_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "matrix.yaml"
    output_dir = (tmp_path / "out").as_posix()
    config_path.write_text(
        f"""
run:
  output_dir: {output_dir}
  samples_per_family: 10
  difficulty: 0.55
  seeds: [0]
  splits: [test_iid]
  temperature: 0.0
  max_tokens: 512
  timeout: 90
models:
  - id: deepseek_v4
    enabled: true
    backend: openai
    model: deepseek-v4-pro
    api_mode: chat_completions
    base_url: https://api.deepseek.com
    api_key_env: CUSTOM_API_KEY
    reasoning_effort: high
    thinking: enabled
    temperature: 0.1
    max_tokens: 1024
    timeout: 120
""",
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []

    def fake_run_one(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text(
            json.dumps({"summary": {"total": 1, "correct": 1, "parse_errors": 0, "fallback_records": 0}}),
            encoding="utf-8",
        )
        return {"summary": {"total": 1, "correct": 1, "parse_errors": 0, "fallback_records": 0}}

    manifest = run_llm_baseline_matrix(config_path=config_path, probe=True, run_one=fake_run_one)

    assert calls[0]["temperature"] == 0.1
    assert calls[0]["max_tokens"] == 1024
    assert calls[0]["timeout"] == 120
    assert calls[0]["thinking"] == "enabled"
    assert manifest["jobs"][0]["temperature"] == 0.1
    assert manifest["jobs"][0]["max_tokens"] == 1024
    assert manifest["jobs"][0]["timeout"] == 120
    assert manifest["jobs"][0]["thinking"] == "enabled"
