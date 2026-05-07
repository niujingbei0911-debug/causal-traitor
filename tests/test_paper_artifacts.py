import json
import inspect
from pathlib import Path

import pytest

from experiments.paper_artifacts import build_paper_facing_package, _discover_llm_baseline_matrix
from poster import build_poster


def test_build_paper_facing_package_freezes_traceable_tables(tmp_path: Path) -> None:
    package = build_paper_facing_package(
        source_dir=Path("outputs"),
        output_dir=tmp_path,
        generated_at_utc="2026-05-08T00:00:00Z",
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    poster_metrics = json.loads((tmp_path / "poster_metrics.json").read_text(encoding="utf-8"))
    main_rows = (tmp_path / "main_table_rows.tex").read_text(encoding="utf-8")
    ood_rows = (tmp_path / "ood_table_rows.tex").read_text(encoding="utf-8")

    assert package["manifest"] == tmp_path / "manifest.json"
    assert manifest["artifact_status"] == "fixed-code exploratory snapshot"
    assert manifest["human_audit_status"] == "pending_external_dual_annotation"
    assert manifest["source_runs"]["main_benchmark"]["config"]["effective"]["difficulty"] == 0.55
    assert manifest["source_runs"]["main_benchmark"]["seeds"] == [0, 1, 2]
    assert manifest["source_runs"]["main_benchmark"]["files"]["aggregated_metrics"]["sha256"]
    assert len(manifest["source_runs"]["main_benchmark"]["files"]["aggregated_metrics"]["sha256"]) == 64
    assert manifest["api_smoke_runs"]
    assert manifest["api_smoke_runs"][0]["summary"]["fallback_records"] == 0
    assert manifest["api_smoke_runs"][0]["files"]["json"]["sha256"]

    assert (
        r"\textsc{Direct Judge} & test\_iid & $0.504 \pm 0.090$ & "
        r"$0.889 \pm 0.192$ & $0.500 \pm 0.500$ \tabularnewline"
    ) in main_rows
    assert (
        r"\textsc{Countermodel-Grounded} & test\_ood & $0.891 \pm 0.005$ & "
        r"$0.000 \pm 0.000$ & $0.825 \pm 0.018$ \tabularnewline"
    ) in main_rows
    assert r"\textsc{Paired-Flip} & $0.439$ & $0.485$ & $0.000$ & $0.714$ & $[0.409, 0.455]$ \tabularnewline" in ood_rows

    assert poster_metrics["setup"]["difficulty"] == 0.55
    assert poster_metrics["setup"]["seeds"] == [0, 1, 2]
    assert poster_metrics["main"]["countermodel_grounded"]["test_ood"]["verdict_accuracy"]["mean"] == pytest.approx(0.891)
    assert poster_metrics["ood"]["paired_flip_ood"]["verdict_accuracy"]["mean"] == pytest.approx(0.439)

    combined_generated_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in tmp_path.iterdir()
        if path.suffix in {".tex", ".json"}
    )
    assert "difficulty $0.70" not in combined_generated_text
    assert "0.83/0.85" not in combined_generated_text
    assert "leakage-free benchmark" not in combined_generated_text.lower()


def test_build_paper_facing_package_rejects_missing_sources(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="review_fixed_exp_main_benchmark_aggregated_metrics.json"):
        build_paper_facing_package(source_dir=tmp_path / "missing", output_dir=tmp_path / "out")


def test_discover_llm_baseline_matrix_freezes_manifest_and_artifacts(tmp_path: Path) -> None:
    matrix_dir = tmp_path / "llm_baseline_matrix"
    matrix_dir.mkdir()
    aggregate_path = matrix_dir / "llm_baseline_aggregated_metrics.json"
    csv_path = matrix_dir / "llm_baseline_summary.csv"
    aggregate_path.write_text('{"status":"llm_baseline_matrix_aggregate"}', encoding="utf-8")
    csv_path.write_text("model_id,split\n", encoding="utf-8")
    manifest_path = matrix_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "llm_baseline_matrix",
                "summary": {
                    "jobs_total": 30,
                    "succeeded": 30,
                    "failed": 0,
                    "total_predictions": 1260,
                    "fallback_records": 0,
                    "parse_errors": 0,
                },
                "artifacts": {
                    "aggregated_metrics": str(aggregate_path),
                    "summary_csv": str(csv_path),
                },
            }
        ),
        encoding="utf-8",
    )

    matrix = _discover_llm_baseline_matrix(tmp_path)

    assert matrix is not None
    assert matrix["status"] == "llm_baseline_matrix"
    assert matrix["summary"]["total_predictions"] == 1260
    assert matrix["completed"] is True
    assert matrix["files"]["manifest"]["sha256"]
    assert matrix["files"]["aggregated_metrics"]["sha256"]
    assert matrix["files"]["summary_csv"]["sha256"]


def test_paper_uses_tabular_safe_row_inputs() -> None:
    paper = Path("poster/icml2026/example_paper.tex").read_text(encoding="utf-8")

    assert r"\input{../../outputs/paper_facing" not in paper
    assert r"\input ../../outputs/paper_facing/main_table_rows.tex" in paper
    assert r"\input ../../outputs/paper_facing/ood_table_rows.tex" in paper


def test_poster_builder_exports_pdf_artifact() -> None:
    source = inspect.getsource(build_poster.build_poster)

    assert callable(build_poster.export_poster_pdf)
    assert "poster_template.pptx" in source
    assert "poster_template.pdf" in source
    assert "export_poster_pdf(" in source


def test_poster_builder_formats_api_smoke_status_from_json() -> None:
    text = build_poster._api_smoke_text(
        {
            "model": {"name": "qwen2.5-7b-instruct"},
            "summary": {"total": 9, "fallback_records": 0},
        },
        {
            "summary": {"total": 12, "fallback_records": 0},
        },
    )

    assert text == "API smoke: qwen2.5-7b, IID n=9/OOD n=12, raw response, fallback=0; full matrix pending."
