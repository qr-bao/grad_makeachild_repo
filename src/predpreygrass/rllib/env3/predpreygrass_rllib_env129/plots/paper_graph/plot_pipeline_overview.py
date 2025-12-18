from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


@dataclass(frozen=True)
class PipelineStyle:
    title: str = "PredPreyGrass: PBT → Training → Data → Analysis"
    figsize: Tuple[float, float] = (14.5, 6.2)

    # Palette (soft, paper-friendly)
    c_train: str = "#E0F3FF"
    c_eval: str = "#FFF7C1"
    c_analysis: str = "#FEE4E8"
    c_box: str = "#FFFFFF"
    c_edge: str = "#2B2B2B"
    c_arrow: str = "#2B2B2B"
    c_accent: str = "#7CA3B8"

    lane_alpha: float = 0.55
    box_alpha: float = 1.0
    font: int = 10


def plot_pipeline_overview(
    *,
    style: PipelineStyle = PipelineStyle(),
    out_path: Optional[str | Path] = None,
    show: bool = False,
):
    """
    Paper-style pipeline diagram:
    PBT (hyperparam search) -> Final training (fixed best config) -> Rollout/logging -> Analysis/plots.
    """
    out_path = Path(out_path) if out_path is not None else None

    fig, ax = plt.subplots(figsize=style.figsize)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    @dataclass(frozen=True)
    class Node:
        x: float
        y: float
        w: float
        h: float

        @property
        def cx(self) -> float:
            return self.x + self.w / 2

        @property
        def cy(self) -> float:
            return self.y + self.h / 2

        @property
        def left(self) -> float:
            return self.x

        @property
        def right(self) -> float:
            return self.x + self.w

        @property
        def bottom(self) -> float:
            return self.y

        @property
        def top(self) -> float:
            return self.y + self.h

    def _anchor(n: Node, side: str) -> Tuple[float, float]:
        if side == "left":
            return (n.left, n.cy)
        if side == "right":
            return (n.right, n.cy)
        if side == "top":
            return (n.cx, n.top)
        if side == "bottom":
            return (n.cx, n.bottom)
        raise ValueError(f"unknown side: {side}")

    # Helpers
    def box(x, y, w, h, text, *, fc=style.c_box, ec=style.c_edge, lw=1.2, r=0.012, fontsize=None) -> Node:
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.012,rounding_size={r}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            alpha=style.box_alpha,
            zorder=2,
        )
        ax.add_patch(p)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=style.font if fontsize is None else fontsize,
            color=style.c_edge,
            zorder=3,
        )
        return Node(x=float(x), y=float(y), w=float(w), h=float(h))

    def lane(y0, y1, label, fc):
        ax.add_patch(
            Rectangle(
                (0.07, y0),
                0.90,
                y1 - y0,
                facecolor=fc,
                edgecolor="none",
                alpha=style.lane_alpha,
                zorder=0,
            )
        )
        ax.text(0.02, (y0 + y1) / 2, label, fontsize=style.font + 1, va="center", ha="left", color=style.c_edge)

    def arrow(
        x1,
        y1,
        x2,
        y2,
        *,
        text=None,
        connectionstyle: str = "arc3",
        text_dx: float = 0.0,
        text_dy: float = 0.018,
    ):
        a = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.2,
            color=style.c_arrow,
            connectionstyle=connectionstyle,
            zorder=4,
        )
        ax.add_patch(a)
        if text:
            ax.text(
                (x1 + x2) / 2 + text_dx,
                (y1 + y2) / 2 + text_dy,
                text,
                fontsize=style.font - 1,
                ha="center",
                va="bottom",
                color=style.c_edge,
            )

    def doc(x, y, w, h, label) -> Node:
        ax.add_patch(
            Rectangle((x, y), w, h, facecolor="#FFFFFF", edgecolor=style.c_edge, linewidth=1.0, zorder=2)
        )
        # folded corner
        ax.add_patch(
            Polygon(
                [(x + w * 0.75, y + h), (x + w, y + h), (x + w, y + h * 0.75)],
                closed=True,
                facecolor=style.c_eval,
                edgecolor=style.c_edge,
                linewidth=0.8,
                zorder=3,
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=style.font - 1, zorder=4)
        return Node(x=float(x), y=float(y), w=float(w), h=float(h))

    def diamond(cx, cy, w, h, text):
        verts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
        ax.add_patch(Polygon(verts, closed=True, facecolor="#FFFFFF", edgecolor=style.c_edge, linewidth=1.2, zorder=2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=style.font - 1, zorder=3)

    # Lanes
    lane(0.68, 0.93, "Training", style.c_train)
    lane(0.38, 0.64, "Evaluation / Logging", style.c_eval)
    lane(0.06, 0.35, "Analysis", style.c_analysis)

    ax.text(0.5, 0.975, style.title, ha="center", va="top", fontsize=style.font + 6, color=style.c_edge)

    # --- Layout (paper-clean, minimal elements, no overlaps) ---
    # Training row (left -> right)
    env_cfg = box(0.10, 0.815, 0.18, 0.075, "Env config\n+ search space")
    env_cfg_file = doc(0.12, 0.74, 0.14, 0.05, "config_env*.json")

    pbt = box(0.32, 0.815, 0.16, 0.075, "PBT\n(parallel trials)")
    pbt_summary = doc(0.34, 0.74, 0.12, 0.05, "pbt_summary.json")

    sel = (0.54, 0.852)
    diamond(sel[0], sel[1], 0.085, 0.11, "select\nTop-K")

    final_train = box(0.61, 0.815, 0.24, 0.075, "Final training\n(best cfg, multi-seed)")
    ckpt = doc(0.87, 0.83, 0.10, 0.05, "checkpoints/")

    arrow(*_anchor(env_cfg, "bottom"), *_anchor(env_cfg_file, "top"))
    arrow(*_anchor(env_cfg, "right"), *_anchor(pbt, "left"), text="budget", text_dy=0.010)
    arrow(*_anchor(pbt, "bottom"), *_anchor(pbt_summary, "top"))
    arrow(*_anchor(pbt_summary, "right"), sel[0] - 0.045, sel[1], text="rank", text_dy=0.010)
    arrow(sel[0] + 0.045, sel[1], *_anchor(final_train, "left"), text="best cfg", text_dy=0.010)
    arrow(*_anchor(final_train, "right"), *_anchor(ckpt, "left"))

    # Evaluation / logging (left-to-right, aligned boxes)
    rollout = box(0.61, 0.515, 0.18, 0.075, "Rollout / Eval\n(N episodes)")
    logging = box(0.81, 0.515, 0.16, 0.075, "Logging\n(metrics, events)")
    metrics = doc(0.82, 0.435, 0.07, 0.05, "metrics.csv")
    traj = doc(0.90, 0.435, 0.07, 0.05, "traj.npz")

    arrow(*_anchor(final_train, "bottom"), *_anchor(rollout, "top"))
    arrow(*_anchor(rollout, "right"), *_anchor(logging, "left"))
    arrow(*_anchor(logging, "bottom"), *_anchor(metrics, "top"))
    arrow(*_anchor(logging, "bottom"), *_anchor(traj, "top"))

    # Analysis (vertical stack, straight)
    aggregate = box(0.81, 0.245, 0.16, 0.07, "Aggregate\n(mean ± CI)")
    stats = box(0.81, 0.155, 0.16, 0.07, "Stat tests\n/ ablations")
    plots = box(0.81, 0.065, 0.16, 0.07, "Plots & Tables\n(curves, snapshots)")

    arrow(*_anchor(metrics, "bottom"), *_anchor(aggregate, "top"))
    arrow(*_anchor(aggregate, "bottom"), *_anchor(stats, "top"))
    arrow(*_anchor(stats, "bottom"), *_anchor(plots, "top"))
    arrow(*_anchor(traj, "bottom"), *_anchor(plots, "top"), connectionstyle="arc3,rad=-0.18")

    # Footnote
    ax.text(
        0.5,
        0.02,
        "Color lanes indicate stage; artifacts shown as documents/folders. Parallelism shown via trial/seed mini-boxes.",
        ha="center",
        va="bottom",
        fontsize=style.font - 2,
        color=style.c_edge,
        alpha=0.9,
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=240, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _main(argv: Sequence[str] | None = None) -> int:
    out = Path("pipeline_overview.png")
    plot_pipeline_overview(out_path=out, show=False)
    print(f"Wrote: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
