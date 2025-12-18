"""
Plot population curves from manually exported CSVs (one file per tag).

Usage example:
    PYTHONPATH=../../../.. python utils/plot_population_csv_custom.py \
      --pred "a2c_predator=run_results/.../(42).csv,ppo_predator=...,random_predator=...,sac_predator=...,td3_predator=..." \
      --prey "a2c_prey=run_results/.../(47).csv,ppo_prey=...,random_prey=...,sac_prey=...,td3_prey=..." \
      --out-dir run_results/orch_coexist_torch_final_20251215_224814/plots_manual \
      --smoothing 9

Notes:
    - Each CSV should have columns including 'Step' and 'Value' (TensorBoard scalar export format).
    - Smoothing is a rolling mean over 'Step' order.
    - Outputs two PNGs: predator_population.png and prey_population.png.
"""

import argparse
import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


plt.style.use("seaborn-v0_8-whitegrid")


def parse_pairs(arg: str) -> Dict[str, Path]:
    """
    Parse a string like "label1=path1,label2=path2".
    """
    out = {}
    if not arg:
        return out
    for part in arg.split(","):
        if "=" not in part:
            raise ValueError(f"Expect label=path, got: {part}")
        label, path = part.split("=", 1)
        out[label.strip()] = Path(path.strip()).expanduser()
    return out


def load_series(csv_path: Path, smoothing: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Step" not in df or "Value" not in df:
        raise ValueError(f"CSV missing Step/Value columns: {csv_path}")
    df = df.sort_values("Step")
    if smoothing > 1:
        df["Smooth"] = df["Value"].rolling(window=smoothing, min_periods=1, center=True).mean()
    else:
        df["Smooth"] = df["Value"]
    return df[["Step", "Smooth"]]


def plot_group(series: Dict[str, pd.DataFrame], title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.get_cmap("tab10")
    for idx, (label, df) in enumerate(series.items()):
        ax.plot(df["Step"], df["Smooth"], label=label, color=colors(idx % 10), linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("episode")
    ax.set_ylabel("population")
    ax.legend(fontsize=9, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Comma-separated label=csv for predator.")
    ap.add_argument("--prey", required=True, help="Comma-separated label=csv for prey.")
    ap.add_argument("--out-dir", default="plots_manual", help="Output directory.")
    ap.add_argument("--smoothing", type=int, default=7, help="Rolling window size for smoothing.")
    args = ap.parse_args()

    pred_map = parse_pairs(args.pred)
    prey_map = parse_pairs(args.prey)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_series = {k: load_series(p, args.smoothing) for k, p in pred_map.items()}
    prey_series = {k: load_series(p, args.smoothing) for k, p in prey_map.items()}

    plot_group(pred_series, "Predator population (per algo)", out_dir / "predator_population.png")
    plot_group(prey_series, "Prey population (per algo)", out_dir / "prey_population.png")


if __name__ == "__main__":
    main()
