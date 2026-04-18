"""Unified verifier pipeline entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmark.schema import PublicCausalInstance, require_public_instance
from verifier.assumption_ledger import build_assumption_ledger
from verifier.claim_parser import parse_claim
from verifier.countermodel_search import search_countermodels
from verifier.decision import VerifierDecision, decide_verdict


@dataclass(slots=True)
class VerifierPipeline:
    """One-click verifier pipeline: parse -> ledger -> countermodel -> decide."""

    tool_runner: Any | None = None

    def _resolve_claim_text(
        self,
        claim_text: str | None,
        transcript: str | list[Any] | tuple[Any, ...] | None,
    ) -> str:
        normalized_claim = (claim_text or "").strip()
        if normalized_claim:
            return normalized_claim
        if isinstance(transcript, str):
            return transcript.strip()
        if transcript:
            return "\n".join(str(item) for item in transcript).strip()
        raise ValueError("VerifierPipeline requires a non-empty claim_text or transcript.")

    def run(
        self,
        claim_text: str | None,
        *,
        scenario: PublicCausalInstance | None = None,
        transcript: str | list[Any] | tuple[Any, ...] | None = None,
        tool_trace: list[Any] | tuple[Any, ...] | None = None,
        tool_context: dict[str, Any] | None = None,
    ) -> VerifierDecision:
        public_scenario = None if scenario is None else require_public_instance(scenario)
        resolved_claim = self._resolve_claim_text(claim_text, transcript)
        parsed_claim = parse_claim(resolved_claim, transcript=transcript)
        ledger = build_assumption_ledger(parsed_claim)
        search_context = {
            **(tool_context or {}),
            "claim_text": resolved_claim,
            "transcript": transcript,
        }
        if public_scenario is not None:
            search_context.update(
                {
                    "public_instance": public_scenario,
                    "observed_data": public_scenario.observed_data.copy(deep=True),
                    "proxy_variables": list(getattr(public_scenario, "proxy_variables", [])),
                    "selection_variables": list(getattr(public_scenario, "selection_variables", [])),
                    "selection_mechanism": getattr(public_scenario, "selection_mechanism", None),
                }
            )
        countermodel_result = search_countermodels(
            parsed_claim,
            ledger,
            scenario=public_scenario,
            context=search_context,
        )
        resolved_tool_trace = []
        if not countermodel_result.found_countermodel:
            resolved_tool_trace = self.run_tools(
                parsed_claim,
                ledger,
                scenario=public_scenario,
                claim_text=resolved_claim,
                transcript=transcript,
                tool_trace=tool_trace,
                tool_context=tool_context,
            )
        return decide_verdict(
            parsed_claim,
            ledger,
            countermodel_result,
            tool_trace=resolved_tool_trace,
        )

    def run_tools(
        self,
        parsed_claim,
        ledger,
        *,
        scenario: PublicCausalInstance | None,
        claim_text: str,
        transcript: str | list[Any] | tuple[Any, ...] | None,
        tool_trace: list[Any] | tuple[Any, ...] | None = None,
        tool_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if tool_trace is not None:
            return self._normalize_tool_trace(tool_trace)
        if self.tool_runner is None:
            return []

        runner = self.tool_runner
        if callable(runner):
            raw_trace = runner(
                parsed_claim=parsed_claim,
                ledger=ledger,
                scenario=scenario,
                claim_text=claim_text,
                transcript=transcript,
                tool_context=tool_context or {},
            )
        elif hasattr(runner, "run"):
            raw_trace = runner.run(
                parsed_claim=parsed_claim,
                ledger=ledger,
                scenario=scenario,
                claim_text=claim_text,
                transcript=transcript,
                tool_context=tool_context or {},
            )
        elif hasattr(runner, "execute"):
            raw_trace = runner.execute(
                parsed_claim=parsed_claim,
                ledger=ledger,
                scenario=scenario,
                claim_text=claim_text,
                transcript=transcript,
                tool_context=tool_context or {},
            )
        else:
            raise TypeError(f"Unsupported tool_runner type: {type(runner)!r}")

        if isinstance(raw_trace, dict):
            raw_trace = raw_trace.get("tool_trace", raw_trace.get("results", []))
        return self._normalize_tool_trace(raw_trace)

    def _normalize_tool_trace(
        self,
        tool_trace: list[Any] | tuple[Any, ...] | Any,
    ) -> list[dict[str, Any]]:
        if tool_trace is None:
            return []
        if isinstance(tool_trace, dict):
            return [dict(tool_trace)]

        normalized: list[dict[str, Any]] = []
        for item in tool_trace:
            if isinstance(item, dict):
                normalized.append(dict(item))
            else:
                normalized.append({"summary": str(item)})
        return normalized


def run_verifier_pipeline(
    claim_text: str | None,
    *,
    scenario: PublicCausalInstance | None = None,
    transcript: str | list[Any] | tuple[Any, ...] | None = None,
    tool_trace: list[Any] | tuple[Any, ...] | None = None,
    tool_context: dict[str, Any] | None = None,
    tool_runner: Any | None = None,
) -> VerifierDecision:
    """Functional wrapper for the verifier pipeline."""

    return VerifierPipeline(tool_runner=tool_runner).run(
        claim_text,
        scenario=scenario,
        transcript=transcript,
        tool_trace=tool_trace,
        tool_context=tool_context,
    )
