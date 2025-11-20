#!/usr/bin/env python3
"""
Analyze cross-evaluation outputs and produce paper-ready score tables.

Inputs:
  - One or more evaluation result directories. Each directory should contain
    either `catch_rate_matrix.csv` directly, or a subfolder `cross_eval_*`
    that contains it (as produced by scripts/evaluation/cross_eval.py).

Outputs (per input directory):
  - algorithm_scores.csv        # Algo-level Predator/Prey/Overall (+adaptability if available)
  - summary_predator_scores.csv # Per-predator-label average catch rate
  - summary_prey_scores.csv     # Per-prey-label average escape/survival rate (1 - catch)
  - (optional) algorithm_scores.tex  # LaTeX tabular of algorithm_scores

Aggregated output (optional):
  - When multiple inputs are provided, `--merge-output` can write a combined
    table with an extra `run_label` column to compare runs (e.g., Stage 1.1 vs 1.2 vs 1.3).

Usage examples:
  # Single run (writes CSVs into the run directory)
  python scripts/analysis/analyze_results.py \
    --input outputs/evaluation_results/stage1_3_gen18 --with-latex

  # Compare multiple runs and emit a merged CSV
  python scripts/analysis/analyze_results.py \
    -i outputs/evaluation_results/stage1_1_baseline \
       outputs/evaluation_results/stage1_2_baseline \
       outputs/evaluation_results/stage1_3_gen18 \
    --labels Stage1.1 Stage1.2 Stage1.3 \
    --merge-output outputs/evaluation_results/summary_comparison.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def resolve_eval_leaf(eval_dir: Path) -> Path:
    """Return the directory that directly contains catch_rate_matrix.csv.

    Accept either eval_dir itself or the newest subdir matching cross_eval_*.
    """
    if (eval_dir / "catch_rate_matrix.csv").exists():
        return eval_dir
    candidates = sorted(
        [p for p in eval_dir.glob("cross_eval_*") if (p / "catch_rate_matrix.csv").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No catch_rate_matrix.csv found under {eval_dir}")


_ALGO_LABEL_RE = re.compile(r"([A-Za-z0-9]+)_(pred|prey)", flags=re.IGNORECASE)


def algo_from_label(label: str) -> str:
    """Extract algorithm name from matrix label like 'PPO_pred' or 'SAC_prey'."""
    s = str(label).strip()
    m = _ALGO_LABEL_RE.match(s)
    return (m.group(1) if m else s).upper()


def load_adaptability_df(leaf: Path) -> Optional[pd.DataFrame]:
    """Load adaptability scores if present; return DataFrame with columns:
    [algo, adaptability_score, adaptability_rank] or None if unavailable.
    Supports a few common JSON shapes.
    """
    candidates = [leaf / "adaptability_scores.json", leaf.parent / "adaptability_scores.json"]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        if "scores" in data and isinstance(data["scores"], dict):
            obj = data["scores"]
        elif "adaptability_scores" in data and isinstance(data["adaptability_scores"], dict):
            obj = data["adaptability_scores"]
        else:
            obj = data

        rows = []
        for algo, val in obj.items():
            if isinstance(val, dict):
                score = val.get("score", val.get("value", val.get("adaptability")))
                rank = val.get("rank")
            else:
                score, rank = val, None
            try:
                score_f = float(score)
            except Exception:
                continue
            rows.append({"algo": str(algo).upper(), "adaptability_score": score_f, "adaptability_rank": rank})

        if rows:
            return pd.DataFrame(rows)

    return None


def compute_scores(leaf: Path, sort_by: str = "overall", ascending: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute per-algorithm Predator/Prey/Overall and per-label summaries.

    Returns:
      algo_scores: columns [algo, predator_score, prey_score, overall, (opt) adaptability_*]
      pred_labels: columns [pred_label, predator_score]
      prey_labels: columns [prey_label, prey_score]
    """
    m = pd.read_csv(leaf / "catch_rate_matrix.csv", index_col=0)

    # Predator performance: row-wise mean catch rate
    pred_row_mean = m.mean(axis=1)
    # Ensure stable column names regardless of index name
    pred_df = pred_row_mean.to_frame(name="predator_score").reset_index()
    # Rename the first column to pred_label safely
    if len(pred_df.columns) >= 2:
        pred_df.columns = ["pred_label", "predator_score"] + list(pred_df.columns[2:])
    else:
        pred_df.rename(columns={pred_df.columns[0]: "pred_label"}, inplace=True)
    pred_df["algo"] = pred_df["pred_label"].apply(algo_from_label)

    # Prey performance: escape/survival = 1 - column-wise mean catch rate
    prey_col_escape = 1 - m.mean(axis=0)
    prey_df = prey_col_escape.to_frame(name="prey_score").reset_index()
    # Rename the first column to prey_label safely
    if len(prey_df.columns) >= 2:
        prey_df.columns = ["prey_label", "prey_score"] + list(prey_df.columns[2:])
    else:
        prey_df.rename(columns={prey_df.columns[0]: "prey_label"}, inplace=True)
    prey_df["algo"] = prey_df["prey_label"].apply(algo_from_label)

    # Aggregate at algorithm level
    pred_algo = pred_df.groupby("algo")["predator_score"].mean()
    prey_algo = prey_df.groupby("algo")["prey_score"].mean()
    summary = pd.concat([pred_algo, prey_algo], axis=1)
    summary["overall"] = summary[["predator_score", "prey_score"]].mean(axis=1)

    algo_scores = summary.reset_index().rename(columns={"index": "algo"})
    algo_scores["algo"] = algo_scores["algo"].str.upper()

    # Merge adaptability if available
    adapt_df = load_adaptability_df(leaf)
    if adapt_df is not None:
        algo_scores = algo_scores.merge(adapt_df, on="algo", how="left")

    # Sort
    if sort_by in algo_scores.columns:
        algo_scores = algo_scores.sort_values(sort_by, ascending=ascending, kind="mergesort")

    return algo_scores, pred_df[["pred_label", "predator_score"]], prey_df[["prey_label", "prey_score"]]


def write_latex_table(df: pd.DataFrame, out_path: Path, floatfmt: str = ".3f") -> None:
    cols = [c for c in ["algo", "predator_score", "prey_score", "overall", "adaptability_score", "adaptability_rank"] if c in df.columns]
    tex = df[cols].to_latex(index=False, float_format=(lambda x: (f"{x:{floatfmt}}" if isinstance(x, float) else str(x))))
    out_path.write_text(tex)


def analyze_single(input_dir: Path, with_latex: bool, sort_by: str, ascending: bool) -> Path:
    leaf = resolve_eval_leaf(input_dir)
    algo_scores, pred_labels, prey_labels = compute_scores(leaf, sort_by=sort_by, ascending=ascending)

    algo_path = leaf / "algorithm_scores.csv"
    pred_path = leaf / "summary_predator_scores.csv"
    prey_path = leaf / "summary_prey_scores.csv"

    algo_scores.to_csv(algo_path, index=False)
    pred_labels.to_csv(pred_path, index=False)
    prey_labels.to_csv(prey_path, index=False)

    if with_latex:
        write_latex_table(algo_scores, leaf / "algorithm_scores.tex")

    print(f"[OK] {leaf}:\n - {algo_path}\n - {pred_path}\n - {prey_path}" + ("\n - " + str(leaf / "algorithm_scores.tex") if with_latex else ""))
    return leaf


def merge_multiple(algo_tables: List[Tuple[str, pd.DataFrame]], out_csv: Path) -> None:
    frames = []
    for label, df in algo_tables:
        d = df.copy()
        d.insert(0, "run_label", label)
        frames.append(d)
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(out_csv, index=False)
    print(f"[OK] merged -> {out_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize cross-eval results into score tables.")
    p.add_argument("--input", "-i", nargs="+", required=True, help="One or more evaluation result directories")
    p.add_argument("--labels", nargs="*", help="Optional labels for inputs (same length as --input)")
    p.add_argument("--merge-output", help="If set and multiple inputs given, write merged CSV here")
    p.add_argument("--with-latex", action="store_true", help="Also write LaTeX table for each run")
    p.add_argument("--sort-by", default="overall", help="Column to sort algorithm_scores by (default: overall)")
    p.add_argument("--ascending", action="store_true", help="Sort ascending instead of descending")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [Path(x) for x in args.input]

    if args.labels and len(args.labels) != len(inputs):
        raise SystemExit("--labels length must equal number of --input paths")

    algo_tables: List[Tuple[str, pd.DataFrame]] = []
    for idx, inp in enumerate(inputs):
        leaf = analyze_single(inp, with_latex=args.with_latex, sort_by=args.sort_by, ascending=args.ascending)
        algo_df = pd.read_csv(leaf / "algorithm_scores.csv")
        label = args.labels[idx] if args.labels else leaf.parent.name if leaf.parent != inp else leaf.name
        algo_tables.append((label, algo_df))

    if len(algo_tables) > 1 and args.merge_output:
        merge_multiple(algo_tables, Path(args.merge_output))


if __name__ == "__main__":
    main()
