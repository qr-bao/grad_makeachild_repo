from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import to_rgb


Agent = Tuple[float, float, int]  # (x, y, group_idx)
Vec2 = Tuple[float, float]


@dataclass(frozen=True)
class EnvSnapshotSpec:
    predator_marker: str = "^"
    prey_marker: str = "o"
    food_marker: str = "s"
    marker_size: int = 90
    marker_edgecolor: str = "black"
    marker_edgewidth: float = 0.7
    show_axes: bool = False
    marker_margin: float = 1.4
    avoid_exact_overlaps: bool = True
    overlap_jitter_radius: float = 0.55
    resolve_overlaps: bool = True
    min_agent_separation: float = 1.35
    relax_max_iter: int = 180

    # Food/grass grid rendering (0/1 occupancy -> small squares)
    food_alpha: float = 0.55
    food_color: str = "#CDECCF"
    ground_color: str = "#FAFAFA"
    food_marker_size: int = 14
    max_food_points: int = 8000

    # Palette (from your reference image)
    palette: Tuple[str, ...] = (
        "#A5D1B0",
        "#CE8A8D",
        "#FFF7C1",
        "#E0F3FF",
        "#ADD3F4",
        "#F7C9CF",
        "#FEE4E8",
        "#7CA3B8",
        "#BFB8D6",
        "#FCCB8E",
    )
    predator_group_colors: Tuple[str, ...] = (
        "#CE8A8D",
        "#FCCB8E",
        "#7CA3B8",
        "#F7C9CF",
        "#FEE4E8",
    )
    prey_group_colors: Tuple[str, ...] = (
        "#A5D1B0",
        "#FFF7C1",
        "#E0F3FF",
        "#FEE4E8",
        "#BFB8D6",
    )

    legend_fontsize: int = 10
    title: str = "PredPreyGrass Environment Snapshot"

    # Motion arrows
    draw_motion: bool = True
    motion_fraction: float = 1.0
    motion_scale: float = 2.6
    motion_alpha: float = 0.7
    motion_color: str = "#2B2B2B"
    motion_width: float = 0.003
    motion_headwidth: float = 3.2
    motion_headlength: float = 4.0
    motion_headaxislength: float = 3.5

    # Trails (optional): fading dots looks cleaner than lines
    draw_trails: bool = True
    trail_steps: int = 6
    trail_dt: float = 0.35
    trail_alpha: float = 0.28
    trail_size: float = 28.0
    trail_size_decay: float = 0.86
    trail_alpha_decay: float = 0.78
    trail_lighten: float = 0.35  # blend toward white


def _apply_overlap_jitter(
    points: Sequence[Tuple[float, float]],
    *,
    jitter_radius: float,
    bounds: Tuple[float, float, float, float],
) -> List[Tuple[float, float]]:
    """
    Resolve exact coordinate overlaps by placing duplicates on a small ring.
    This keeps the visualization readable when multiple agents share a cell.
    """
    xmin, xmax, ymin, ymax = bounds
    counts: dict[Tuple[float, float], int] = {}
    out: List[Tuple[float, float]] = []
    for x, y in points:
        key = (round(float(x), 3), round(float(y), 3))
        k = counts.get(key, 0)
        counts[key] = k + 1
        if k == 0:
            out.append((float(x), float(y)))
            continue
        angle = (k * 2.399963229728653) % (2 * np.pi)  # golden angle
        dx = jitter_radius * float(np.cos(angle))
        dy = jitter_radius * float(np.sin(angle))
        xx = min(max(float(x) + dx, xmin), xmax)
        yy = min(max(float(y) + dy, ymin), ymax)
        out.append((xx, yy))
    return out


def _relax_min_distance(
    points: np.ndarray,
    *,
    min_dist: float,
    bounds: Tuple[float, float, float, float],
    max_iter: int,
) -> np.ndarray:
    """
    Simple repulsion-based relaxation to ensure points are not too close.
    Operates in-place on a copy and clips to bounds each iteration.
    """
    if len(points) <= 1:
        return points

    xmin, xmax, ymin, ymax = bounds
    pts = points.astype(float, copy=True)
    min_dist = float(min_dist)
    min_dist2 = min_dist * min_dist

    for _ in range(int(max_iter)):
        moved = 0.0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dx = pts[j, 0] - pts[i, 0]
                dy = pts[j, 1] - pts[i, 1]
                d2 = dx * dx + dy * dy
                if d2 >= min_dist2:
                    continue

                if d2 <= 1e-12:
                    angle = ((i + 1) * 2.399963229728653 + (j + 3) * 1.1) % (2 * np.pi)
                    ux = float(np.cos(angle))
                    uy = float(np.sin(angle))
                    dist = 0.0
                else:
                    dist = float(np.sqrt(d2))
                    ux = dx / dist
                    uy = dy / dist

                push = (min_dist - dist) * 0.5
                pts[i, 0] -= ux * push
                pts[i, 1] -= uy * push
                pts[j, 0] += ux * push
                pts[j, 1] += uy * push
                moved += push

        pts[:, 0] = np.clip(pts[:, 0], xmin, xmax)
        pts[:, 1] = np.clip(pts[:, 1], ymin, ymax)
        if moved < 1e-3:
            break

    return pts


def _blend_toward_white(hex_or_rgb: str | Tuple[float, float, float], amount: float) -> Tuple[float, float, float]:
    r, g, b = to_rgb(hex_or_rgb)
    amount = float(np.clip(amount, 0.0, 1.0))
    return (r + (1.0 - r) * amount, g + (1.0 - g) * amount, b + (1.0 - b) * amount)


def plot_environment_snapshot(
    *,
    grass_grid: np.ndarray,
    predators: Sequence[Agent],
    prey: Sequence[Agent],
    predator_vel: Optional[Sequence[Vec2]] = None,
    prey_vel: Optional[Sequence[Vec2]] = None,
    predator_group_names: Sequence[str],
    prey_group_names: Sequence[str],
    spec: EnvSnapshotSpec = EnvSnapshotSpec(),
    out_path: Optional[str | Path] = None,
    show: bool = False,
):
    """
    Plot a single environment snapshot:
    - Background: grass occupancy grid (0/1)
    - Agents: predator (triangle) & prey (circle)
    - Color: algorithm group (5 groups per type)
    """
    out_path = Path(out_path) if out_path is not None else None

    if grass_grid.ndim != 2:
        raise ValueError(f"grass_grid must be 2D, got shape={grass_grid.shape}")

    height, width = grass_grid.shape
    predators = list(predators)
    prey = list(prey)

    if len(predator_group_names) != 5 or len(prey_group_names) != 5:
        raise ValueError("predator_group_names and prey_group_names must be length 5 each")

    fig, ax = plt.subplots(figsize=(13.0, 8))
    # Leave room on the right for outside legends (prevents cropping/truncation)
    fig.subplots_adjust(right=0.58)

    # Ground
    ax.set_facecolor(spec.ground_color)

    # Food (grass) as small squares
    food_cells = np.argwhere(grass_grid > 0)
    if food_cells.size:
        if len(food_cells) > spec.max_food_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(food_cells), size=spec.max_food_points, replace=False)
            food_cells = food_cells[idx]
        ys = food_cells[:, 0].astype(float) + 0.5
        xs = food_cells[:, 1].astype(float) + 0.5
        ax.scatter(
            xs,
            ys,
            s=spec.food_marker_size,
            c=spec.food_color,
            marker=spec.food_marker,
            edgecolors="none",
            alpha=spec.food_alpha,
            zorder=0,
        )

    # Environment boundary
    ax.add_patch(
        Rectangle(
            (0, 0),
            width,
            height,
            fill=False,
            edgecolor="black",
            linewidth=1.2,
            zorder=3,
        )
    )

    bounds = (
        float(spec.marker_margin),
        float(width - spec.marker_margin),
        float(spec.marker_margin),
        float(height - spec.marker_margin),
    )

    # Preprocess points (optional jitter/relaxation to avoid overlaps)
    pred_pts = [(float(a[0]), float(a[1])) for a in predators]
    prey_pts = [(float(a[0]), float(a[1])) for a in prey]
    if spec.avoid_exact_overlaps:
        pred_pts = _apply_overlap_jitter(pred_pts, jitter_radius=spec.overlap_jitter_radius, bounds=bounds)
        prey_pts = _apply_overlap_jitter(prey_pts, jitter_radius=spec.overlap_jitter_radius, bounds=bounds)

    all_pts = np.array(pred_pts + prey_pts, dtype=float)
    if spec.resolve_overlaps and len(all_pts) > 1:
        all_pts = _relax_min_distance(
            all_pts,
            min_dist=spec.min_agent_separation,
            bounds=bounds,
            max_iter=spec.relax_max_iter,
        )
        pred_pts = [(float(x), float(y)) for x, y in all_pts[: len(pred_pts)]]
        prey_pts = [(float(x), float(y)) for x, y in all_pts[len(pred_pts) :]]

    def _scatter_agents(
        agents: Sequence[Agent],
        pts: Sequence[Tuple[float, float]],
        *,
        marker: str,
        colors: Sequence[str],
        z: int,
    ):
        if not agents:
            return None
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        gs = np.array([a[2] for a in agents], dtype=int)

        if np.any(gs < 0) or np.any(gs >= 5):
            bad = sorted(set(int(g) for g in gs if g < 0 or g >= 5))
            raise ValueError(f"group_idx must be in [0,4], got {bad}")

        cs = [colors[int(g)] for g in gs]
        ax.scatter(
            xs,
            ys,
            s=spec.marker_size,
            c=cs,
            marker=marker,
            edgecolors=spec.marker_edgecolor,
            linewidths=spec.marker_edgewidth,
            clip_on=True,
            zorder=z,
        )

        return xs, ys, gs

    pred_xyg = _scatter_agents(
        predators,
        pred_pts,
        marker=spec.predator_marker,
        colors=spec.predator_group_colors,
        z=5,
    )
    prey_xyg = _scatter_agents(
        prey,
        prey_pts,
        marker=spec.prey_marker,
        colors=spec.prey_group_colors,
        z=6,
    )

    # Motion arrows (optional; helps show "running" dynamics)
    def _draw_motion(xs: np.ndarray, ys: np.ndarray, vels: Sequence[Vec2] | None, z: int):
        if not spec.draw_motion or vels is None:
            return
        if len(vels) != len(xs):
            raise ValueError(f"velocity length mismatch: {len(vels)} vs {len(xs)}")
        vx = np.array([float(v[0]) for v in vels], dtype=float)
        vy = np.array([float(v[1]) for v in vels], dtype=float)
        n = len(xs)
        keep = max(0, min(n, int(np.ceil(n * float(spec.motion_fraction)))))
        if keep == 0:
            return
        idx = np.arange(n)
        # deterministic selection for stable figures
        idx = idx[:: max(1, int(np.floor(n / keep)))]
        idx = idx[:keep]
        ax.quiver(
            xs[idx],
            ys[idx],
            vx[idx],
            vy[idx],
            angles="xy",
            scale_units="xy",
            scale=float(spec.motion_scale),
            width=float(spec.motion_width),
            color=spec.motion_color,
            alpha=float(spec.motion_alpha),
            headwidth=float(spec.motion_headwidth),
            headlength=float(spec.motion_headlength),
            headaxislength=float(spec.motion_headaxislength),
            zorder=z,
            clip_on=True,
        )

    def _draw_trails(
        xs: np.ndarray,
        ys: np.ndarray,
        gs: np.ndarray,
        vels: Sequence[Vec2] | None,
        *,
        group_colors: Sequence[str],
        z: int,
    ):
        if not spec.draw_trails or vels is None:
            return
        if len(vels) != len(xs):
            raise ValueError(f"velocity length mismatch: {len(vels)} vs {len(xs)}")
        vx = np.array([float(v[0]) for v in vels], dtype=float)
        vy = np.array([float(v[1]) for v in vels], dtype=float)
        steps = int(spec.trail_steps)
        if steps <= 0:
            return
        dt = float(spec.trail_dt)
        base_s = float(spec.trail_size)
        base_a = float(spec.trail_alpha)
        for i in range(len(xs)):
            color = _blend_toward_white(group_colors[int(gs[i])], spec.trail_lighten)
            x0 = float(xs[i])
            y0 = float(ys[i])
            for t in range(1, steps + 1):
                xt = x0 - float(vx[i]) * t * dt
                yt = y0 - float(vy[i]) * t * dt
                xt = min(max(xt, bounds[0]), bounds[1])
                yt = min(max(yt, bounds[2]), bounds[3])
                s = base_s * (float(spec.trail_size_decay) ** (t - 1))
                a = base_a * (float(spec.trail_alpha_decay) ** (t - 1))
                ax.scatter(
                    [xt],
                    [yt],
                    s=s,
                    c=[color],
                    marker="o",
                    edgecolors="none",
                    alpha=a,
                    zorder=z,
                    clip_on=True,
                )

    if pred_xyg is not None:
        _draw_trails(
            pred_xyg[0],
            pred_xyg[1],
            pred_xyg[2],
            predator_vel,
            group_colors=spec.predator_group_colors,
            z=4,
        )
        _draw_motion(pred_xyg[0], pred_xyg[1], predator_vel, z=7)
    if prey_xyg is not None:
        _draw_trails(
            prey_xyg[0],
            prey_xyg[1],
            prey_xyg[2],
            prey_vel,
            group_colors=spec.prey_group_colors,
            z=4,
        )
        _draw_motion(prey_xyg[0], prey_xyg[1], prey_vel, z=8)

    ax.set_title(spec.title, fontsize=16)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")

    if spec.show_axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(False)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Legends: shapes/types + predator groups + prey groups
    type_handles = [
        Line2D(
            [0],
            [0],
            marker=spec.predator_marker,
            color="w",
            label="Predator",
            markerfacecolor="#DDDDDD",
            markeredgecolor="black",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker=spec.prey_marker,
            color="w",
            label="Prey",
            markerfacecolor="#DDDDDD",
            markeredgecolor="black",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker=spec.food_marker,
            color="w",
            label="Food",
            markerfacecolor=spec.food_color,
            markeredgecolor="none",
            markersize=9,
        ),
    ]

    pred_group_handles = [
        Patch(facecolor=spec.predator_group_colors[i], edgecolor="black", label=predator_group_names[i])
        for i in range(5)
    ]
    prey_group_handles = [
        Patch(facecolor=spec.prey_group_colors[i], edgecolor="black", label=prey_group_names[i]) for i in range(5)
    ]

    leg_pred = ax.legend(
        handles=pred_group_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=spec.legend_fontsize,
        frameon=True,
        title="Predator Groups (Color)",
        title_fontsize=spec.legend_fontsize,
    )
    ax.add_artist(leg_pred)

    leg_prey = ax.legend(
        handles=prey_group_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.68),
        borderaxespad=0.0,
        fontsize=spec.legend_fontsize,
        frameon=True,
        title="Prey Groups (Color)",
        title_fontsize=spec.legend_fontsize,
    )
    ax.add_artist(leg_prey)

    ax.legend(
        handles=type_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.40),
        borderaxespad=0.0,
        fontsize=spec.legend_fontsize,
        frameon=True,
        title="Legend",
        title_fontsize=spec.legend_fontsize,
    )

    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _example_data(
    *,
    width: int = 50,
    height: int = 50,
    n_predators: int = 18,
    n_prey: int = 28,
    grass_density: float = 0.15,
    seed: int = 3,
):
    rng = np.random.default_rng(seed)

    grass_grid = (rng.random((height, width)) < grass_density).astype(int)

    def _sample_centers(*, min_sep: float) -> List[Tuple[float, float]]:
        centers: List[Tuple[float, float]] = []
        tries = 0
        while len(centers) < 5 and tries < 2000:
            tries += 1
            cx = float(rng.uniform(10.0, width - 10.0))
            cy = float(rng.uniform(10.0, height - 10.0))
            if all((cx - x) ** 2 + (cy - y) ** 2 >= min_sep**2 for x, y in centers):
                centers.append((cx, cy))
        if len(centers) < 5:
            # fallback: fixed-ish positions
            centers = [
                (12.0, 12.0),
                (width - 12.0, 12.0),
                (12.0, height - 12.0),
                (width - 12.0, height - 12.0),
                (width / 2.0, height / 2.0),
            ]
        return centers

    def _clustered_agents(
        n: int,
        *,
        spread: float,
        min_dist_global: float,
        centers: Sequence[Tuple[float, float]],
        occupied: List[Tuple[float, float]],
    ) -> List[Agent]:
        group_counts = rng.multinomial(n, [1 / 5] * 5)
        agents: List[Agent] = []
        min2 = float(min_dist_global) ** 2

        def _ok(x: float, y: float) -> bool:
            return all((x - ox) ** 2 + (y - oy) ** 2 >= min2 for ox, oy in occupied)

        for group_idx in range(5):
            k = int(group_counts[group_idx])
            if k == 0:
                continue
            cx, cy = centers[group_idx]
            pts: List[Tuple[float, float]] = []
            attempts = 0
            while len(pts) < k and attempts < 12000:
                attempts += 1
                x = float(np.clip(rng.normal(cx, spread), 1.4, width - 1.4))
                y = float(np.clip(rng.normal(cy, spread), 1.4, height - 1.4))
                if _ok(x, y):
                    pts.append((x, y))
                    occupied.append((x, y))
            while len(pts) < k:
                # final fallback: place anywhere with weaker constraint
                x = float(rng.uniform(1.6, width - 1.6))
                y = float(rng.uniform(1.6, height - 1.6))
                pts.append((x, y))
                occupied.append((x, y))

            agents.extend([(pts[i][0], pts[i][1], int(group_idx)) for i in range(k)])

        rng.shuffle(agents)
        return agents

    predator_centers = _sample_centers(min_sep=18.0)
    prey_centers = _sample_centers(min_sep=18.0)

    occupied: List[Tuple[float, float]] = []
    predators = _clustered_agents(
        n_predators,
        spread=1.55,
        min_dist_global=1.35,
        centers=predator_centers,
        occupied=occupied,
    )
    prey = _clustered_agents(
        n_prey,
        spread=1.75,
        min_dist_global=1.25,
        centers=prey_centers,
        occupied=occupied,
    )

    predator_group_names = [f"predator_algo{i+1}" for i in range(5)]
    prey_group_names = [f"prey_algo{i+1}" for i in range(5)]

    # Example motion: short random velocities
    predator_vel = [(float(rng.normal(0, 0.9)), float(rng.normal(0, 0.9))) for _ in range(len(predators))]
    prey_vel = [(float(rng.normal(0, 0.9)), float(rng.normal(0, 0.9))) for _ in range(len(prey))]

    return grass_grid, predators, prey, predator_vel, prey_vel, predator_group_names, prey_group_names


def _main() -> int:
    out = Path("environment_snapshot.png")
    grass_grid, predators, prey, predator_vel, prey_vel, predator_group_names, prey_group_names = _example_data()
    plot_environment_snapshot(
        grass_grid=grass_grid,
        predators=predators,
        prey=prey,
        predator_vel=predator_vel,
        prey_vel=prey_vel,
        predator_group_names=predator_group_names,
        prey_group_names=prey_group_names,
        out_path=out,
        show=False,
    )
    print(f"Wrote: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
