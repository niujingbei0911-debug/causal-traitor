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


def test_llm_baseline_matrix_writes_paper_facing_aggregate_artifacts(tmp_path: Path) -> None:
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
""",
        encoding="utf-8",
    )

    def fake_run_one(**kwargs: Any) -> dict[str, Any]:
        seed = int(kwargs["seed"])
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "instance_id": f"sample-{seed}-valid",
                "split": kwargs["split_name"],
                "seed": seed,
                "graph_family": "chain",
                "query_type": "ate",
                "claim_text": "Claim",
                "gold_label": "valid",
                "predicted_label": "valid",
                "correct": True,
                "parse_error": False,
                "fallback_detected": False,
                "prompt_sha256": "a" * 64,
                "raw_response": '{"verdict":"valid"}',
                "parsed_payload": {"verdict": "valid"},
                "response_metadata": {"finish_reason": "stop"},
                "backend": kwargs["backend"],
                "model_name": kwargs["model"],
            },
            {
                "instance_id": f"sample-{seed}-invalid",
                "split": kwargs["split_name"],
                "seed": seed,
                "graph_family": "fork",
                "query_type": "ate",
                "claim_text": "Claim",
                "gold_label": "invalid",
                "predicted_label": "valid" if seed == 0 else "invalid",
                "correct": seed == 1,
                "parse_error": False,
                "fallback_detected": False,
                "prompt_sha256": "b" * 64,
                "raw_response": '{"verdict":"valid"}',
                "parsed_payload": {"verdict": "valid"},
                "response_metadata": {"finish_reason": "stop"},
                "backend": kwargs["backend"],
                "model_name": kwargs["model"],
            },
        ]
        payload = {
            "status": "llm_baseline_matrix_job",
            "summary": {
                "total": len(records),
                "correct": sum(1 for record in records if record["correct"]),
                "parse_errors": 0,
                "fallback_records": 0,
            },
            "records": records,
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    manifest = run_llm_baseline_matrix(config_path=config_path, run_one=fake_run_one)

    artifacts = manifest["artifacts"]
    aggregate_path = Path(artifacts["aggregated_metrics"])
    csv_path = Path(artifacts["summary_csv"])
    raw_path = Path(artifacts["raw_predictions"])
    markdown_path = Path(artifacts["summary_markdown"])
    assert aggregate_path.exists()
    assert csv_path.exists()
    assert raw_path.exists()
    assert markdown_path.exists()

    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    metrics = aggregate["models"]["live_model"]["test_iid"]
    assert metrics["total"] == 4
    assert metrics["accuracy"]["mean"] == 0.75
    assert metrics["unsafe_acceptance_rate"]["mean"] == 0.5
    assert metrics["parse_error_rate"]["mean"] == 0.0
    assert metrics["seeds"]["0"]["accuracy"] == 0.5
    assert metrics["seeds"]["1"]["accuracy"] == 1.0
    assert "live_model,test_iid,ALL,4,3,0.7500" in csv_path.read_text(encoding="utf-8")
    assert len(raw_path.read_text(encoding="utf-8").strip().splitlines()) == 4
    assert "Strong LLM Baseline Matrix" in markdown_path.read_text(encoding="utf-8")


def test_llm_baseline_matrix_can_reuse_existing_successful_jobs(tmp_path: Path) -> None:
    config_path = tmp_path / "matrix.yaml"
    output_dir = tmp_path / "out"
    config_path.write_text(
        f"""
run:
  output_dir: {output_dir.as_posix()}
  samples_per_family: 10
  difficulty: 0.55
  seeds: [0]
  splits: [test_iid]
  max_public_rows: 5
models:
  - id: existing_model
    enabled: true
    backend: openai
    model: gpt-5.5
    api_mode: responses
    api_key_env: OPENAI_API_KEY
  - id: missing_model
    enabled: true
    backend: openai
    model: deepseek-v4-pro
    api_mode: chat_completions
    api_key_env: CUSTOM_API_KEY
""",
        encoding="utf-8",
    )
    existing_path = output_dir / "existing_model_seed0_test_iid.json"
    existing_path.parent.mkdir(parents=True)
    existing_path.write_text(
        json.dumps(
            {
                "status": "llm_baseline_matrix_job",
                "summary": {"total": 1, "correct": 1, "parse_errors": 0, "fallback_records": 0},
                "records": [
                    {
                        "instance_id": "existing-1",
                        "split": "test_iid",
                        "seed": 0,
                        "gold_label": "valid",
                        "predicted_label": "valid",
                        "correct": True,
                        "parse_error": False,
                        "fallback_detected": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_run_one(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["model"])
        output_path = Path(kwargs["output_path"])
        payload = {
            "status": "llm_baseline_matrix_job",
            "summary": {"total": 1, "correct": 0, "parse_errors": 0, "fallback_records": 0},
            "records": [
                {
                    "instance_id": "missing-1",
                    "split": "test_iid",
                    "seed": 0,
                    "gold_label": "invalid",
                    "predicted_label": "valid",
                    "correct": False,
                    "parse_error": False,
                    "fallback_detected": False,
                }
            ],
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    manifest = run_llm_baseline_matrix(config_path=config_path, reuse_existing=True, run_one=fake_run_one)

    assert calls == ["deepseek-v4-pro"]
    assert manifest["summary"]["jobs_total"] == 2
    assert manifest["summary"]["succeeded"] == 2
    assert manifest["summary"]["total_predictions"] == 2
    assert manifest["jobs"][0]["reused_existing"] is True
    assert manifest["jobs"][1].get("reused_existing") is not True


def test_llm_baseline_matrix_accepts_job_level_parallelism(tmp_path: Path) -> None:
    config_path = tmp_path / "matrix.yaml"
    output_dir = (tmp_path / "out").as_posix()
    config_path.write_text(
        f"""
run:
  output_dir: {output_dir}
  seeds: [0, 1]
  splits: [test_iid]
models:
  - id: model_a
    enabled: true
    backend: openai
    model: gpt-5.5
    api_mode: responses
    api_key_env: OPENAI_API_KEY
  - id: model_b
    enabled: true
    backend: openai
    model: deepseek-v4-pro
    api_mode: chat_completions
    api_key_env: CUSTOM_API_KEY
""",
        encoding="utf-8",
    )

    def fake_run_one(**kwargs: Any) -> dict[str, Any]:
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": {"total": 1, "correct": 1, "parse_errors": 0, "fallback_records": 0},
            "records": [
                {
                    "instance_id": f"{kwargs['model']}-{kwargs['seed']}",
                    "split": kwargs["split_name"],
                    "seed": kwargs["seed"],
                    "gold_label": "valid",
                    "predicted_label": "valid",
                    "correct": True,
                    "parse_error": False,
                    "fallback_detected": False,
                }
            ],
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    manifest = run_llm_baseline_matrix(config_path=config_path, parallel_jobs=2, run_one=fake_run_one)

    assert manifest["summary"]["jobs_total"] == 4
    assert manifest["summary"]["succeeded"] == 4
    assert manifest["summary"]["parallel_jobs"] == 2
    assert [job["model_id"] for job in manifest["jobs"]] == ["model_a", "model_a", "model_b", "model_b"]
