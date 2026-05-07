"""Run a tiny API-backed verifier baseline with auditable raw responses.

This is not the full paper baseline matrix. It is a plumbing and schema smoke
test that proves real API calls can be recorded without silently accepting mock
fallbacks as model evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark.schema import VERDICT_LABEL_SPACE
from experiments.benchmark_harness import build_seed_benchmark_run
from game.llm_service import DASHSCOPE_BASE_URL, OPENAI_BASE_URL, LLMResponse, LLMService


DEFAULT_MODEL = "qwen2.5-7b-instruct"
DEFAULT_BACKEND = "dashscope"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 192
DEFAULT_TIMEOUT = 60

FORBIDDEN_PROMPT_KEYS = (
    "gold_label",
    "gold_answer",
    "attacker_rationale",
    "countermodel_witness",
    "support_witness",
    "assumption_witness",
    "true_scm",
    "true_dag",
    "ground_truth",
    "hidden_variables",
)


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _compact_public_instance(public: Any, *, max_rows: int = 5) -> dict[str, Any]:
    payload = public.to_dict()
    observed_rows = list(payload.get("observed_data") or [])
    data_rows = list(payload.get("data") or [])
    payload["observed_data_n_rows"] = len(observed_rows)
    payload["observed_data_preview"] = observed_rows[:max_rows]
    payload["data_n_rows"] = len(data_rows)
    payload.pop("observed_data", None)
    payload.pop("data", None)
    return payload


def build_api_prompt(sample: Any, *, max_rows: int = 5) -> str:
    public_payload = _compact_public_instance(sample.public, max_rows=max_rows)
    public_json = json.dumps(public_payload, ensure_ascii=False, sort_keys=True, indent=2)
    prompt = f"""You are evaluating a causal claim using only public evidence.

Allowed verdict labels: valid, invalid, unidentifiable.

Public scenario:
{public_json}

Claim:
{sample.claim.claim_text}

Return only JSON with keys:
{{"verdict": "valid|invalid|unidentifiable", "confidence": 0.0, "reasoning_summary": "one public-evidence reason"}}
"""
    lowered = prompt.lower()
    leaked_keys = [key for key in FORBIDDEN_PROMPT_KEYS if key in lowered]
    if leaked_keys:
        raise ValueError(f"API baseline prompt contains forbidden key(s): {leaked_keys!r}.")
    return prompt


def parse_api_verdict(parsed_payload: dict[str, Any] | None, raw_text: str) -> str | None:
    payload = parsed_payload or {}
    for key in ("verdict", "label", "predicted_label", "answer"):
        if key in payload:
            normalized = _normalize_label(payload[key])
            if normalized is not None:
                return normalized

    text = str(raw_text or "").strip().lower()
    for label in ("unidentifiable", "not identifiable", "underidentified", "invalid", "valid"):
        if re.search(rf"\b{re.escape(label)}\b", text):
            return _normalize_label(label)
    return None


def _normalize_label(value: Any) -> str | None:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    synonym_map = {
        "not_identifiable": "unidentifiable",
        "nonidentifiable": "unidentifiable",
        "underidentified": "unidentifiable",
        "underdetermined": "unidentifiable",
        "refute": "invalid",
        "refuted": "invalid",
        "false": "invalid",
        "support": "valid",
        "supported": "valid",
        "true": "valid",
    }
    normalized = synonym_map.get(normalized, normalized)
    return normalized if normalized in VERDICT_LABEL_SPACE else None


def _fallback_detected(response: LLMResponse) -> bool:
    note = str(response.metadata.get("note", "")).lower()
    return (
        response.backend.lower() == "mock"
        or response.text.strip().lower().startswith("[mock:")
        or "mock fallback" in note
        or "using mock" in note
    )


async def _run_api_baseline_smoke_async(
    *,
    output_path: Path,
    service: Any,
    seed: int,
    split_name: str,
    max_samples: int,
    samples_per_family: int,
    difficulty: float,
    max_public_rows: int,
    generated_at_utc: str,
    reject_mock_fallback: bool,
    run_status: str,
    run_note: str | None,
) -> dict[str, Any]:
    try:
        run = build_seed_benchmark_run(
            seed=int(seed),
            samples_per_family=int(samples_per_family),
            difficulty=float(difficulty),
        )
        if split_name not in run.split_samples:
            raise ValueError(f"Unsupported split_name {split_name!r}; available splits: {sorted(run.split_samples)}")
        samples = sorted(run.split_samples[split_name], key=lambda item: item.claim.instance_id)[: int(max_samples)]
        if not samples:
            raise ValueError(f"No samples selected for split {split_name!r}.")

        records: list[dict[str, Any]] = []
        for sample in samples:
            prompt = build_api_prompt(sample, max_rows=max_public_rows)
            response, parsed_payload = await service.generate_json(
                prompt,
                system_prompt="You are a careful causal verifier. Use only the public scenario.",
                temperature=getattr(service, "temperature", DEFAULT_TEMPERATURE),
                max_tokens=getattr(service, "max_tokens", DEFAULT_MAX_TOKENS),
            )
            fallback = _fallback_detected(response)
            if fallback and reject_mock_fallback:
                raise RuntimeError(
                    "Detected mock fallback during API baseline smoke; refusing to write model evidence."
                )
            predicted_label = parse_api_verdict(parsed_payload, response.text)
            gold_label = sample.claim.gold_label.value
            records.append(
                {
                    "instance_id": sample.claim.instance_id,
                    "split": split_name,
                    "seed": int(seed),
                    "graph_family": sample.claim.graph_family,
                    "query_type": sample.claim.query_type,
                    "claim_text": sample.claim.claim_text,
                    "gold_label": gold_label,
                    "predicted_label": predicted_label,
                    "correct": predicted_label == gold_label if predicted_label is not None else False,
                    "parse_error": predicted_label is None,
                    "fallback_detected": fallback,
                    "prompt": prompt,
                    "prompt_sha256": _sha256_text(prompt),
                    "raw_response": response.text,
                    "parsed_payload": parsed_payload,
                    "response_metadata": dict(response.metadata),
                    "backend": response.backend,
                    "model_name": response.model_name,
                }
            )

        total = len(records)
        correct = sum(1 for record in records if record["correct"])
        parse_errors = sum(1 for record in records if record["parse_error"])
        payload = {
            "status": run_status,
            "generated_at_utc": generated_at_utc,
            "note": run_note
            or (
                "Tiny API-backed plumbing smoke. Do not cite as a strong LLM baseline "
                "or as the paper's full model matrix."
            ),
            "model": {
                "backend": getattr(service, "backend", None),
                "name": getattr(service, "model_name", None),
                "temperature": getattr(service, "temperature", None),
                "max_tokens": getattr(service, "max_tokens", None),
            },
            "config": {
                "seed": int(seed),
                "split_name": split_name,
                "max_samples": int(max_samples),
                "samples_per_family": int(samples_per_family),
                "difficulty": float(difficulty),
                "max_public_rows": int(max_public_rows),
                "reject_mock_fallback": bool(reject_mock_fallback),
            },
            "summary": {
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total else 0.0,
                "parse_errors": parse_errors,
                "fallback_records": sum(1 for record in records if record["fallback_detected"]),
            },
            "records": records,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
    finally:
        close = getattr(service, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result


def _build_service(
    *,
    model: str,
    backend: str,
    base_url: str | None = None,
    api_key_env: str | None = None,
    api_mode: str = "chat_completions",
    reasoning_effort: str | None = None,
    thinking: str | None = None,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> LLMService:
    resolved_backend = backend.lower()
    resolved_base_url = base_url or (OPENAI_BASE_URL if resolved_backend == "openai" else DASHSCOPE_BASE_URL)
    resolved_api_key_env = api_key_env or ("OPENAI_API_KEY" if resolved_backend == "openai" else None)
    config: dict[str, Any] = {
        "backend": resolved_backend,
        "name": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "base_url": resolved_base_url,
        "api_key_env": resolved_api_key_env,
        "api_mode": api_mode,
        "reasoning_effort": reasoning_effort,
    }
    if thinking:
        config["thinking"] = thinking
    return LLMService(
        config,
        allow_mock_fallback=False,
    )


def run_api_baseline_smoke(
    *,
    output_path: Path | str = Path("outputs/api_baseline_smoke.json"),
    service: Any | None = None,
    model: str = DEFAULT_MODEL,
    backend: str = DEFAULT_BACKEND,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT,
    base_url: str | None = None,
    api_key_env: str | None = None,
    api_mode: str = "chat_completions",
    reasoning_effort: str | None = None,
    thinking: str | None = None,
    seed: int = 0,
    split_name: str = "test_iid",
    max_samples: int = 3,
    samples_per_family: int = 1,
    difficulty: float = 0.55,
    max_public_rows: int = 5,
    generated_at_utc: str | None = None,
    reject_mock_fallback: bool = True,
    run_status: str = "api_smoke",
    run_note: str | None = None,
) -> dict[str, Any]:
    resolved_service = service or _build_service(
        model=model,
        backend=backend,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        base_url=base_url,
        api_key_env=api_key_env,
        api_mode=api_mode,
        reasoning_effort=reasoning_effort,
        thinking=thinking,
    )
    return asyncio.run(
        _run_api_baseline_smoke_async(
            output_path=Path(output_path),
            service=resolved_service,
            seed=seed,
            split_name=split_name,
            max_samples=max_samples,
            samples_per_family=samples_per_family,
            difficulty=difficulty,
            max_public_rows=max_public_rows,
            generated_at_utc=generated_at_utc or _now_utc(),
            reject_mock_fallback=reject_mock_fallback,
            run_status=run_status,
            run_note=run_note,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/api_baseline_smoke.json"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL, e.g. https://api.openai.com/v1.")
    parser.add_argument("--api-key-env", default=None, help="Environment variable to read the API key from.")
    parser.add_argument("--api-mode", default="chat_completions", choices=["chat_completions", "responses"])
    parser.add_argument("--reasoning-effort", default=None, help="Optional Responses API reasoning effort, e.g. low, medium, high.")
    parser.add_argument("--thinking", default=None, help="Optional chat-completions thinking mode, e.g. enabled.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-name", default="test_iid")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--samples-per-family", type=int, default=1)
    parser.add_argument("--difficulty", type=float, default=0.55)
    parser.add_argument("--max-public-rows", type=int, default=5)
    parser.add_argument("--allow-mock-fallback", action="store_true")
    args = parser.parse_args()
    payload = run_api_baseline_smoke(
        output_path=args.output,
        model=args.model,
        backend=args.backend,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        api_mode=args.api_mode,
        reasoning_effort=args.reasoning_effort,
        thinking=args.thinking,
        seed=args.seed,
        split_name=args.split_name,
        max_samples=args.max_samples,
        samples_per_family=args.samples_per_family,
        difficulty=args.difficulty,
        max_public_rows=args.max_public_rows,
        reject_mock_fallback=not args.allow_mock_fallback,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
