"""Generate the method pipeline figure for the ICML poster."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")


def _reexec_with_matplotlib() -> None:
    """Re-exec with a local interpreter that can import matplotlib, if needed."""

    try:
        import matplotlib  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    if os.environ.get("MAKE_PIPELINE_REEXEC") == "1":
        raise

    candidates = [
        os.environ.get("MATPLOTLIB_PYTHON"),
        "/opt/homebrew/Caskroom/miniconda/base/envs/lab01/bin/python",
        "/opt/homebrew/Caskroom/miniconda/base/bin/python",
        "python3",
    ]
    for candidate in [c for c in candidates if c]:
        try:
            probe = subprocess.run(
                [candidate, "-c", "import matplotlib"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            continue
        if probe.returncode == 0:
            env = os.environ.copy()
            env["MAKE_PIPELINE_REEXEC"] = "1"
            os.execve(candidate, [candidate, *sys.argv], env)

    raise ModuleNotFoundError(
        "matplotlib is required; install it or set MATPLOTLIB_PYTHON"
    )


_reexec_with_matplotlib()

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


def draw_box(ax, x, y, w, h, text, fill):
    border = "#1f3b6b"
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=2.0,
        edgecolor=border,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=11,
        color="#14243c",
        wrap=True,
        fontweight="bold" if "Stage" in text else "normal",
    )
    return patch


def main() -> None:
    out_path = Path("poster/method_pipeline.png")
    fig, ax = plt.subplots(figsize=(16, 5), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Keep the saved tight bounding box aligned with the requested 16:5 canvas.
    ax.add_patch(Rectangle((0, 0), 16, 5, facecolor="white", edgecolor="none"))

    stage_fill = "#e6edf7"
    io_fill = "#fbf6e6"
    border = "#1f3b6b"
    y = 2.45
    h = 1.12
    specs = [
        (0.25, 1.8, "Claim +\nPublic Scenario", io_fill),
        (2.45, 1.95, "Stage 1:\nParser", stage_fill),
        (4.8, 1.95, "Stage 2:\nLedger", stage_fill),
        (7.15, 2.2, "Stage 3:\nCountermodel\nSearch", stage_fill),
        (9.9, 2.55, "Stage 4:\nTool-backed\nAdjudication", stage_fill),
        (13.15, 2.55, "{valid / invalid /\nunidentifiable}", io_fill),
    ]

    boxes = []
    for x, w, label, fill in specs:
        boxes.append(draw_box(ax, x, y, w, h, label, fill))

    for left, right in zip(boxes, boxes[1:]):
        x1 = left.get_x() + left.get_width()
        x2 = right.get_x()
        yy = y + h / 2
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 0.08, yy),
                (x2 - 0.08, yy),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=2.0,
                color=border,
            )
        )

    ax.text(
        11.18,
        1.72,
        "+ assumption ledger / refusal reason / missing-info spec / witnesses",
        ha="center",
        va="center",
        fontsize=10,
        color="#3e4c61",
    )

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
