"""
Compute simple生态适应性指标（LVI/SI/RI/CI）从 TensorBoard 事件文件。

用法示例（在项目根目录）:
    PYTHONPATH=../../../.. python utils/compute_ppg_metrics.py \
        --run-dir run_results/orch_coexist_torch_final_20251215_224814 \
        --out metrics_ppg.csv

说明/假设：
    - 事件文件包含 population/predator/<algo_key>、population/prey/<algo_key>、
      以及 population/predator/total、population/prey/total（脚本会自动查找）。
    - 指标定义（简化版本，可按需调整）:
        * LVI: 该算法人口 >0 的 episode 比例。
        * SI: 1/(1+CV)，其中 CV=std/mean（仅在 mean>0 时计算，否则 0）。
        * RI: 从历史最小值回升到终值 80% 所需 episode 数；无法恢复则 NaN。
        * CI: 平均人口占本物种总人口的比例。
    - 输出 CSV，并在终端打印。
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def load_scalars(event_file: Path, prefix: str) -> Dict[str, List[Tuple[int, float]]]:
    ea = event_accumulator.EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    tags = [t for t in ea.Tags().get("scalars", []) if t.startswith(prefix)]
    for tag in tags:
        out[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
    return out


def to_series(values: List[Tuple[int, float]]) -> pd.Series:
    """Convert (step, value) list to Series indexed by step (episode)."""
    if not values:
        return pd.Series(dtype=float)
    steps, vals = zip(*values)
    s = pd.Series(vals, index=steps, dtype=float).sort_index()
    return s


def compute_metrics(series: pd.Series, total_series: pd.Series) -> Dict[str, float]:
    # align indices
    df = pd.DataFrame({"pop": series, "total": total_series})
    df = df.sort_index()
    df["pop"].fillna(0, inplace=True)
    df["total"].replace(0, np.nan, inplace=True)

    # LVI: fraction of episodes with pop > 0
    lvi = float((df["pop"] > 0).mean()) if not df.empty else float("nan")

    # SI: stability via CV
    if df["pop"].mean() > 0:
        cv = df["pop"].std() / df["pop"].mean()
        si = float(1.0 / (1.0 + cv))
    else:
        si = 0.0

    # RI: episodes to recover from historical min to 80% of final mean
    ri = float("nan")
    if not df.empty:
        min_idx = int(df["pop"].idxmin())
        target = 0.8 * df["pop"].iloc[-20:].mean()  # use tail mean as final
        recovered = df[df.index >= min_idx]
        hit = recovered[recovered["pop"] >= target]
        if target > 0 and not hit.empty:
            ri = float(hit.index[0] - min_idx)

    # CI: average share of total population
    with np.errstate(divide="ignore", invalid="ignore"):
        share = df["pop"] / df["total"]
    ci = float(share.mean(skipna=True))

    return {"LVI": lvi, "SI": si, "RI": ri, "CI": ci}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="包含 events.* 的目录（与训练时 --log-dir 相同）。")
    ap.add_argument("--out", default="metrics_ppg.csv", help="输出 CSV 路径。")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    events = sorted(run_dir.glob("events.*"), key=os.path.getmtime)
    if not events:
        raise SystemExit(f"No events.* under {run_dir}")
    event_file = events[-1]

    # population scalars
    pop_scalars = load_scalars(event_file, "population/")
    # total series per species
    pred_total = to_series(pop_scalars.get("population/predator/total", []))
    prey_total = to_series(pop_scalars.get("population/prey/total", []))

    rows = []
    for tag, vals in pop_scalars.items():
        if tag.endswith("/total"):
            continue
        s = to_series(vals)
        if tag.startswith("population/predator/"):
            total = pred_total
            algo = tag.split("/")[-1]
            species = "predator"
        elif tag.startswith("population/prey/"):
            total = prey_total
            algo = tag.split("/")[-1]
            species = "prey"
        else:
            continue
        metrics = compute_metrics(s, total)
        rows.append({"algo": algo, "species": species, **metrics})

    df = pd.DataFrame(rows).sort_values(["species", "algo"])
    out_path = Path(args.out)
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"[OK] saved {out_path}")


if __name__ == "__main__":
    main()
