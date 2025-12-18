from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge


Point = Tuple[float, float]


@dataclass(frozen=True)
class AgentObservationPlotSpec:
    world_xlim: Tuple[float, float] = (0.0, 20.0)
    world_ylim: Tuple[float, float] = (0.0, 20.0)
    sight_radius: float = 7.0
    sight_center_deg: float = 90.0
    sight_fov_deg: float = 360.0
    draw_hearing: bool = False

    # Color palette (from your reference image)
    color_prey: str = "#A5D1B0"
    color_predator: str = "#CE8A8D"
    color_sight: str = "#FFF7C1"
    color_observer: str = "#ADD3F4"
    color_background: str = "#7CA3B8"


def plot_agent_observation_space(
    *,
    observer: Point = (10.0, 10.0),
    predator: Optional[Point] = (12.2, 14.2),
    prey: Sequence[Point] = ((9.2, 15.8), (14.6, 14.6)),
    background_points: Optional[Iterable[Point]] = None,
    delta_e_text: Optional[str] = "ΔE: 4.5",
    spec: AgentObservationPlotSpec = AgentObservationPlotSpec(),
    out_path: Optional[str | Path] = None,
    show: bool = False,
    seed: int = 7,
):
    """
    Draw a 2D "Agent Observation Space" figure:
    - Hearing range: large circle (gray, transparent)
    - Sight range: wedge sector (yellow, transparent)
    - Observer (blue), Predator (red), Prey (green)
    - Background entities (gray points)
    - Arrows from observer to targets
    """
    out_path = Path(out_path) if out_path is not None else None

    rng = np.random.default_rng(seed)
    if background_points is None:
        n = 40
        xs = rng.uniform(spec.world_xlim[0], spec.world_xlim[1], size=n)
        ys = rng.uniform(spec.world_ylim[0], spec.world_ylim[1], size=n)
        background_points = list(zip(xs.tolist(), ys.tolist()))
    else:
        background_points = list(background_points)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Sight range (full 360° circle by default)
    if spec.sight_fov_deg >= 359.999:
        sight = Circle(
            observer,
            radius=spec.sight_radius,
            facecolor=spec.color_sight,
            edgecolor="none",
            alpha=0.55,
            label="Sight Range",
            zorder=1,
        )
    else:
        theta1 = spec.sight_center_deg - spec.sight_fov_deg / 2.0
        theta2 = spec.sight_center_deg + spec.sight_fov_deg / 2.0
        sight = Wedge(
            observer,
            r=spec.sight_radius,
            theta1=theta1,
            theta2=theta2,
            facecolor=spec.color_sight,
            edgecolor="none",
            alpha=0.55,
            label="Sight Range",
            zorder=1,
        )
    ax.add_patch(sight)

    # Background points (entities)
    if background_points:
        bx, by = zip(*background_points)
        ax.scatter(bx, by, s=60, c=spec.color_background, alpha=0.55, zorder=2)

    # Observer
    ax.scatter(
        [observer[0]],
        [observer[1]],
        s=200,
        c=spec.color_observer,
        edgecolors="black",
        linewidths=0.6,
        label="Observer",
        zorder=5,
    )

    # Targets + arrows
    def _arrow_to(target: Point, color: str):
        ax.annotate(
            "",
            xy=target,
            xytext=observer,
            arrowprops=dict(arrowstyle="-|>", color="black", lw=2, mutation_scale=18),
            zorder=4,
        )
        ax.scatter([target[0]], [target[1]], s=140, c=color, zorder=6)

    if predator is not None:
        _arrow_to(predator, spec.color_predator)
        ax.scatter([], [], s=140, c=spec.color_predator, label="Predator_<algo_name>)")

    if prey:
        for p in prey:
            _arrow_to(p, spec.color_prey)
        ax.scatter([], [], s=140, c=spec.color_prey, label="Prey_<algo_name>")

    if delta_e_text:
        ax.text(
            observer[0] + 0.8,
            observer[1] - 0.1,
            delta_e_text,
            color="blue",
            fontsize=14,
            zorder=7,
        )

    ax.set_title("Agent Observation Space", fontsize=18)
    ax.set_xlabel("X Coordinate", fontsize=14)
    ax.set_ylabel("Y Coordinate", fontsize=14)
    ax.set_xlim(*spec.world_xlim)
    ax.set_ylim(*spec.world_ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.7, linewidth=1)

    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)


def _main(argv: Sequence[str] | None = None) -> int:
    out = Path("agent_observation_space.png")
    plot_agent_observation_space(out_path=out, show=False)
    print(f"Wrote: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
