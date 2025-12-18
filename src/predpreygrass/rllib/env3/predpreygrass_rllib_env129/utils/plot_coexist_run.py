"""
Quick plotting utility for PredPreyGrass multi-algo runs.

Usage:
    PYTHONPATH=../../../.. python utils/plot_coexist_run.py \
        --run-dir logs/orch_coexist_torch_final_20251206_XXXX \
        --out-dir plots

Outputs (saved under out-dir, created if missing):
    population_by_algo.png   # predator/prey population curves per algo_key
    reward_mean_by_algo.png  # predator/prey episode_reward_mean per algo_key

Notes:
    - Reads TensorBoard event files under --run-dir (the same path you passed to --log-dir when training).
    - Expects tags like population/<species>/<algo_key> and episode_reward_mean/<species>/<algo_key>.
    - If multiple event files exist, picks the latest by mtime.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


# Consistent, cleaner style
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = plt.get_cmap("tab10")


def load_scalars(event_file: Path, prefixes: List[str]) -> pd.DataFrame:
    ea = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={"scalars": 0},
    )
    ea.Reload()
    rows = []
    tags = ea.Tags().get("scalars", [])
    for tag in tags:
        if not any(tag.startswith(p) for p in prefixes):
            continue
        for ev in ea.Scalars(tag):
            rows.append({"tag": tag, "step": ev.step, "value": ev.value})
    return pd.DataFrame(rows)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    s = pd.Series(values)
    return s.rolling(window, min_periods=1, center=True).mean().to_numpy()


def _plot_lines(
    ax,
    df: pd.DataFrame,
    title: str,
    ylabel: str,
    smooth: int,
    tag_filter: Callable[[str], bool],
    max_legend_cols: int = 3,
):
    tags = [t for t in sorted(df.tag.unique()) if tag_filter(t)]
    for idx, tag in enumerate(tags):
        g = df[df["tag"] == tag].sort_values("step")
        y = _smooth(g["value"].to_numpy(), smooth)
        label = tag.split("/", 2)[-1]
        ax.plot(g["step"], y, label=label, color=PALETTE(idx % 10), linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if tags:
        ax.legend(fontsize=8, ncol=max_legend_cols)


def plot_population(df: pd.DataFrame, out_path: Path, smooth: int) -> None:
    if df.empty:
        print("[WARN] population dataframe is empty; skip plot.")
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for species, ax in zip(["predator", "prey"], axes):
        sdf = df[df["tag"].str.startswith(f"population/{species}/")]
        # 只画按算法聚合的曲线：
        # - 排除 total
        # - 排除具体种群编号（population/predator/predator_0、population/prey/prey_3 等）
        def _f(tag: str) -> bool:
            last = tag.split("/")[-1]
            if last in ("total",):
                return False
            # Exclude population ids like predator_0 / prey_4
            if last.startswith(f"{species}_"):
                suffix = last[len(species) + 1 :]
                if suffix.isdigit():
                    return False
            return True

        _plot_lines(ax, sdf, f"Population – {species}", "count", smooth, _f, max_legend_cols=3)
    axes[-1].set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] saved {out_path}")


def plot_reward_mean(df: pd.DataFrame, out_path: Path, smooth: int, include_all: bool) -> None:
    if df.empty:
        print("[WARN] reward dataframe is empty; skip plot.")
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for species, ax in zip(["predator", "prey"], axes):
        sdf = df[df["tag"].str.startswith(f"episode_reward_mean/{species}/")]
        def _f(tag: str) -> bool:
            last = tag.split("/")[-1]
            if not include_all and last == "ALL":
                return False
            return True

        _plot_lines(ax, sdf, f"Episode reward mean – {species}", "reward", smooth, _f, max_legend_cols=3)
    axes[-1].set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Training run directory (the same as --log-dir).")
    ap.add_argument("--out-dir", default="plots", help="Where to save PNGs.")
    ap.add_argument("--smoothing", type=int, default=7, help="Rolling window (episodes) for smoothing curves.")
    ap.add_argument("--include-all", action="store_true", help="Include ALL aggregate reward curves.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.exists():
        sys.exit(f"[ERR] run-dir not found: {run_dir}")

    # Pick the latest event file in run_dir
    event_files = sorted(run_dir.glob("events.*"), key=os.path.getmtime)
    if not event_files:
        sys.exit(f"[ERR] no TensorBoard event files under {run_dir}")
    event_file = event_files[-1]
    print(f"[INFO] using event file: {event_file}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    prefixes = [
        "population/predator/",
        "population/prey/",
        "episode_reward_mean/predator/",
        "episode_reward_mean/prey/",
    ]
    df = load_scalars(event_file, prefixes=prefixes)

    pop_df = df[df["tag"].str.startswith("population/")]
    rew_df = df[df["tag"].str.startswith("episode_reward_mean/")]

    plot_population(pop_df, out_dir / "population_by_algo.png", smooth=args.smoothing)
    plot_reward_mean(rew_df, out_dir / "reward_mean_by_algo.png", smooth=args.smoothing, include_all=args.include_all)


if __name__ == "__main__":
    main()
