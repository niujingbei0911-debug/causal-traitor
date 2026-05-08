"""Build the conference-style poster and chart assets from frozen metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
POSTER_DIR = ROOT / "poster"
OUTPUTS_DIR = ROOT / "outputs"
PAPER_FACING_DIR = OUTPUTS_DIR / "paper_facing"

FONT = "Aptos"
INK = RGBColor(31, 35, 40)
MUTED = RGBColor(91, 99, 110)
LINE = RGBColor(210, 216, 224)
BLUE = "#2F6FED"
TEAL = "#12805C"
RED = "#C7362F"
GOLD = "#B7791F"
GRAY = "#697386"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(metrics: dict[str, Any], name: str) -> dict[str, float]:
    item = metrics[name]
    return {
        "mean": float(item["mean"]),
        "ci_lower": float(item.get("ci_lower", item["mean"])),
        "ci_upper": float(item.get("ci_upper", item["mean"])),
        "std": float(item.get("std", 0.0)),
    }


def _setup_text(poster_metrics: dict[str, Any]) -> str:
    setup = poster_metrics["setup"]
    return (
        f"Exploratory snapshot | seeds={setup['seeds']} | "
        f"10/family | diff={setup['difficulty']:.2f} | "
        "95% CI where shown"
    )


def _llm_baseline_text(llm_payload: dict[str, Any]) -> str:
    summary = llm_payload.get("summary", {})
    models = llm_payload.get("models", {})
    best_model = "n/a"
    best_acc = -1.0
    best_unsafe = 0.0
    for model_id, split_payload in models.items():
        ood = split_payload.get("test_ood", {})
        accuracy = float((ood.get("accuracy") or {}).get("mean", -1.0))
        if accuracy > best_acc:
            best_model = "GPT-5.5" if model_id == "gpt55_xhigh" else str(model_id)
            best_acc = accuracy
            best_unsafe = float((ood.get("unsafe_acceptance_rate") or {}).get("mean", 0.0))
    return (
        f"LLM matrix: 5 models, n={int(summary.get('total_predictions', 0))}, "
        f"failed={int(summary.get('failed', 0))}, parse={int(summary.get('parse_errors', 0))}, "
        f"fallback={int(summary.get('fallback_records', 0))}; "
        f"best OOD {best_model} acc={best_acc:.3f}, unsafe={best_unsafe:.3f}."
    )


def _llm_metric(llm_payload: dict[str, Any], model_id: str, split: str, metric: str) -> float:
    model_payload = llm_payload.get("models", {}).get(model_id, {})
    split_payload = model_payload.get(split, {})
    metric_payload = split_payload.get(metric, {})
    return float(metric_payload.get("mean", 0.0))


def _add_textbox(
    slide: Any,
    text: str,
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    font_size: int = 15,
    bold: bool = False,
    color: RGBColor = INK,
    align: int | None = None,
) -> Any:
    shape = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = shape.text_frame
    frame.clear()
    frame.margin_left = Inches(0.04)
    frame.margin_right = Inches(0.04)
    frame.margin_top = Inches(0.02)
    frame.margin_bottom = Inches(0.02)
    frame.vertical_anchor = MSO_ANCHOR.TOP
    paragraph = frame.paragraphs[0]
    if align is not None:
        paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.name = FONT
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    return shape


def _add_section_title(slide: Any, title: str, left: float, top: float, width: float) -> None:
    _add_textbox(slide, title, left, top, width, 0.25, font_size=16, bold=True, color=RGBColor(20, 83, 136))
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top + 0.28), Inches(width), Inches(0.015))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(20, 83, 136)
    line.line.fill.background()


def _add_panel(slide: Any, left: float, top: float, width: float, height: float) -> None:
    panel = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    panel.fill.solid()
    panel.fill.fore_color.rgb = RGBColor(250, 252, 255)
    panel.line.color.rgb = LINE
    panel.line.width = Pt(0.75)


def _format_bullets(frame: Any, lines: list[str], *, font_size: int = 13, color: RGBColor = INK) -> None:
    frame.clear()
    for idx, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if idx == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.level = 0
        paragraph.font.name = FONT
        paragraph.font.size = Pt(font_size)
        paragraph.font.color.rgb = color
        paragraph.space_after = Pt(4)


def _add_bullets(slide: Any, lines: list[str], left: float, top: float, width: float, height: float, *, font_size: int = 13) -> None:
    shape = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    shape.text_frame.margin_left = Inches(0.05)
    shape.text_frame.margin_right = Inches(0.05)
    shape.text_frame.margin_top = Inches(0.02)
    _format_bullets(shape.text_frame, lines, font_size=font_size)


def build_charts() -> None:
    poster_metrics = _read_json(PAPER_FACING_DIR / "poster_metrics.json")
    main = poster_metrics["main"]
    ood = poster_metrics["ood"]
    llm_baseline = poster_metrics.get("llm_baseline", {})
    ablation = _read_json(OUTPUTS_DIR / "review_fixed_exp_identifiability_ablation_aggregated_metrics.json")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#23272f",
        "xtick.color": "#23272f",
        "ytick.color": "#23272f",
    })

    # Main OOD tradeoff chart. Include the strongest API-backed prompt baseline
    # so the poster text and visual comparison refer to the same object.
    systems = [
        ("direct_judge", "Direct"),
        ("tool_baseline", "Tool"),
        ("refusal_aware_baseline", "Refusal-aware"),
        ("gpt55_xhigh", "GPT-5.5"),
        ("countermodel_grounded", "Ours"),
    ]
    x = np.arange(len(systems))
    acc = []
    acc_low = []
    acc_high = []
    unsafe = []
    for name, _ in systems:
        if name == "gpt55_xhigh":
            accuracy = _llm_metric(llm_baseline, name, "test_ood", "accuracy")
            acc.append(accuracy)
            acc_low.append(0.0)
            acc_high.append(0.0)
            unsafe.append(_llm_metric(llm_baseline, name, "test_ood", "unsafe_acceptance_rate"))
        else:
            accuracy = main[name]["test_ood"]["verdict_accuracy"]["mean"]
            acc.append(accuracy)
            acc_low.append(accuracy - main[name]["test_ood"]["verdict_accuracy"]["ci_lower"])
            acc_high.append(main[name]["test_ood"]["verdict_accuracy"]["ci_upper"] - accuracy)
            unsafe.append(main[name]["test_ood"]["unsafe_acceptance_rate"]["mean"])
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=220)
    ax.bar(x - 0.18, acc, width=0.36, color=BLUE, label="OOD accuracy", yerr=[acc_low, acc_high], capsize=3)
    ax.bar(x + 0.18, unsafe, width=0.36, color=RED, label="Unsafe accept")
    ax.set_xticks(x, [label for _, label in systems], rotation=12, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Main OOD Tradeoff")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    fig.text(0.01, 0.01, _setup_text(poster_metrics), fontsize=8, color="#5B6370")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(POSTER_DIR / "main_chart.png", transparent=False, facecolor="white")
    plt.close(fig)

    # OOD bucket chart.
    buckets = [
        ("graph_family_ood", "Graph"),
        ("mechanism_ood", "Mechanism"),
        ("attack_family_ood", "Attack"),
        ("context_shift_ood", "Context"),
        ("paired_flip_ood", "Paired flip"),
    ]
    x = np.arange(len(buckets))
    values = [ood[name]["verdict_accuracy"]["mean"] for name, _ in buckets]
    lows = [value - ood[name]["verdict_accuracy"]["ci_lower"] for value, (name, _) in zip(values, buckets)]
    highs = [ood[name]["verdict_accuracy"]["ci_upper"] - value for value, (name, _) in zip(values, buckets)]
    colors = [TEAL, TEAL, GOLD, BLUE, RED]
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=220)
    ax.bar(x, values, color=colors, yerr=[lows, highs], capsize=3)
    ax.set_xticks(x, [label for _, label in buckets], rotation=12, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Verdict accuracy")
    ax.set_title("OOD Buckets")
    fig.text(0.01, 0.01, _setup_text(poster_metrics), fontsize=8, color="#5B6370")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(POSTER_DIR / "ood_chart.png", transparent=False, facecolor="white")
    plt.close(fig)

    # Ablation tradeoff chart.
    variants = [
        ("countermodel_grounded", "Full"),
        ("no_countermodel", "No CM"),
        ("no_abstention", "No abstain"),
        ("no_tools", "No tools"),
    ]
    recall = [_metric(ablation[name]["test_ood"], "wise_refusal_recall")["mean"] for name, _ in variants]
    over_refusal = [_metric(ablation[name]["test_ood"], "over_refusal_rate")["mean"] for name, _ in variants]
    accuracy = [_metric(ablation[name]["test_ood"], "verdict_accuracy")["mean"] for name, _ in variants]
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=220)
    sizes = [150 + value * 450 for value in accuracy]
    colors = [BLUE, GOLD, RED, GRAY]
    ax.scatter(over_refusal, recall, s=sizes, color=colors, alpha=0.9, edgecolor="white", linewidth=1.2)
    for x_value, y_value, (_, label) in zip(over_refusal, recall, variants):
        ax.annotate(label, (x_value, y_value), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlim(-0.02, max(over_refusal) + 0.10)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Over-refusal rate")
    ax.set_ylabel("Wise-refusal recall")
    ax.set_title("Ablation Tradeoff")
    ax.grid(axis="both", alpha=0.18)
    fig.text(0.01, 0.01, _setup_text(poster_metrics), fontsize=8, color="#5B6370")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(POSTER_DIR / "ablation_chart.png", transparent=False, facecolor="white")
    plt.close(fig)


def export_poster_pdf(pptx_path: Path, pdf_path: Path) -> None:
    """Export the poster PPTX to PDF using PowerPoint COM on Windows."""
    try:
        import win32com.client  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise RuntimeError("PDF export requires pywin32 and Microsoft PowerPoint on Windows.") from exc

    pptx_path = pptx_path.resolve()
    pdf_path = pdf_path.resolve()
    if pdf_path.exists():
        pdf_path.unlink()

    powerpoint = win32com.client.Dispatch("PowerPoint.Application")
    presentation = None
    try:
        presentation = powerpoint.Presentations.Open(str(pptx_path), True, False, False)
        presentation.SaveAs(str(pdf_path), 32)  # 32 = ppSaveAsPDF
    finally:
        if presentation is not None:
            presentation.Close()
        powerpoint.Quit()


def build_poster() -> None:
    build_charts()
    poster_metrics = _read_json(PAPER_FACING_DIR / "poster_metrics.json")
    manifest = _read_json(PAPER_FACING_DIR / "manifest.json")
    llm_baseline = poster_metrics.get("llm_baseline", {})

    prs = Presentation()
    prs.slide_width = Inches(16)
    prs.slide_height = Inches(9)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = RGBColor(255, 255, 255)

    # Header
    header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(16), Inches(0.82))
    header.fill.solid()
    header.fill.fore_color.rgb = RGBColor(245, 248, 252)
    header.line.fill.background()
    _add_textbox(
        slide,
        "Selective Adversarial Causal Oversight",
        0.45,
        0.12,
        8.7,
        0.38,
        font_size=27,
        bold=True,
        color=INK,
    )
    _add_textbox(
        slide,
        "Benchmark prototype + countermodel-grounded reference verifier",
        0.48,
        0.52,
        8.7,
        0.24,
        font_size=13,
        color=MUTED,
    )
    _add_textbox(slide, "Jingbei Niu   Zeyuan Tong   Yuhang Li | Zhejiang University", 10.5, 0.18, 4.9, 0.24, font_size=12, color=INK, align=PP_ALIGN.RIGHT)
    _add_textbox(slide, _setup_text(poster_metrics), 8.8, 0.52, 6.6, 0.22, font_size=9, color=MUTED, align=PP_ALIGN.RIGHT)

    # Left column.
    _add_section_title(slide, "Task & Contract", 0.45, 1.05, 4.65)
    _add_bullets(
        slide,
        [
            "Verifier sees public scenario + claim only.",
            "Gold SCM, labels, hidden variables stay evaluator-only.",
            "Verdict space: VALID / INVALID / UNIDENTIFIABLE.",
            "Score includes unsafe acceptance and wise refusal.",
        ],
        0.45,
        1.45,
        4.45,
        1.0,
        font_size=13,
    )
    _add_panel(slide, 0.45, 2.55, 4.65, 1.35)
    _add_textbox(slide, "Generated Case", 0.65, 2.72, 1.7, 0.22, font_size=13, bold=True, color=RGBColor(20, 83, 136))
    _add_textbox(
        slide,
        "Claim: observed association between therapy_flag and yield_score is enough for a causal conclusion.\nCorrect label: UNIDENTIFIABLE, because public evidence leaves reverse causality/confounding open.",
        0.65,
        3.0,
        4.25,
        0.72,
        font_size=11,
        color=INK,
    )
    _add_section_title(slide, "Reference Verifier", 0.45, 4.2, 4.65)
    _add_bullets(
        slide,
        [
            "Parse claim -> build identifying-assumption ledger.",
            "Search public-compatible counter-SCMs.",
            "Run Pearl-style tools as supporting evidence.",
            "Return verdict + refusal reason + witness.",
        ],
        0.45,
        4.6,
        4.55,
        1.05,
        font_size=13,
    )
    slide.shapes.add_picture(str(POSTER_DIR / "method_pipeline.png"), Inches(0.45), Inches(5.85), width=Inches(4.65))

    # Middle column.
    _add_section_title(slide, "Main Result", 5.45, 1.05, 5.0)
    slide.shapes.add_picture(str(POSTER_DIR / "main_chart.png"), Inches(5.35), Inches(1.42), width=Inches(5.25))
    _add_bullets(
        slide,
        [
            "Ours: OOD acc 0.891, unsafe accept 0.000.",
            "GPT-5.5 prompt baseline: OOD acc 0.545, unsafe accept 0.071.",
            "Interpretation: current prompt baselines expose the selective-refusal gap.",
        ],
        5.5,
        4.48,
        4.85,
        0.85,
        font_size=12,
    )
    _add_section_title(slide, "Ablation", 5.45, 5.65, 5.0)
    slide.shapes.add_picture(str(POSTER_DIR / "ablation_chart.png"), Inches(5.35), Inches(6.0), width=Inches(5.25))

    # Right column.
    _add_section_title(slide, "OOD Stress", 10.85, 1.05, 4.7)
    slide.shapes.add_picture(str(POSTER_DIR / "ood_chart.png"), Inches(10.75), Inches(1.42), width=Inches(5.0))
    _add_bullets(
        slide,
        [
            "Paired-flip is the clearest current failure slice: acc 0.439.",
            "Graph/mechanism buckets are saturated smoke tests, not headline robustness evidence.",
        ],
        10.95,
        4.48,
        4.35,
        0.7,
        font_size=12,
    )
    _add_section_title(slide, "Validity Gates", 10.85, 5.55, 4.7)
    _add_panel(slide, 10.85, 5.95, 4.7, 2.38)
    _add_bullets(
        slide,
        [
            "Leakage control: oracle-positive reaches 1.000/1.000; score gain is contamination.",
            _llm_baseline_text(llm_baseline),
            "Release gates: larger synthetic runs, real-grounded dual audit, frozen model versions.",
            "Current status: prototype evidence, not a mature benchmark release.",
        ],
        11.05,
        6.12,
        4.4,
        2.0,
        font_size=8,
    )
    _add_textbox(
        slide,
        f"Manifest: outputs/paper_facing/manifest.json | LLM matrix: {manifest['api_model_baseline_status']} | Poster generated from frozen JSON artifacts",
        0.45,
        8.55,
        15.1,
        0.24,
        font_size=9,
        color=MUTED,
        align=PP_ALIGN.CENTER,
    )

    pptx_path = POSTER_DIR / "poster_template.pptx"
    pdf_path = POSTER_DIR / "poster_template.pdf"
    prs.save(pptx_path)
    export_poster_pdf(pptx_path, pdf_path)


if __name__ == "__main__":
    build_poster()
