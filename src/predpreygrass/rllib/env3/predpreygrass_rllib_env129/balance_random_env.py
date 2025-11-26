"""Grid search helper to keep random predator/prey runs in dynamic balance.

It sweeps over a configurable parameter grid, rolls out short random games,
computes simple balance metrics, and dumps the best-scoring config.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from predpreygrass.rllib.env3.predpreygrass_rllib_env129.predpreygrass_rllib_env import (
    PredPreyGrass,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.config.config_env_base import (
    config_env_base,
)

DEFAULT_GRID: Dict[str, Iterable] = {
    "world_width": [350, 400],
    "world_height": [350, 400],
    "n_initial_active_predator": [4, 5, 6],
    "n_initial_active_prey": [18, 22, 26],
    "initial_num_grass": [60, 80, 100],
    "grass_respawn_delay": [120, 150, 200],
    "energy_loss_per_step_predator": [0.12, 0.15],
    "energy_loss_per_step_prey": [0.07, 0.08, 0.09],
    "reward_predator_catch_prey": [3.0, 4.0, 5.0],
    "reward_prey_step": [-0.01, -0.008, -0.006],
    "survival_bonus_predator": [0.1, 0.3],
}


def _iter_grid(grid: Dict[str, Iterable], limit: int | None, seed: int) -> List[Dict]:
    keys = list(grid.keys())
    all_values = [list(grid[k]) for k in keys]
    combos = list(itertools.product(*all_values))
    rng = np.random.default_rng(seed)
    if limit is not None and limit < len(combos):
        idx = rng.choice(len(combos), size=limit, replace=False)
        combos = [combos[i] for i in idx]
    return [dict(zip(keys, values)) for values in combos]


def run_random_episode(cfg: Dict, horizon: int, seed: int) -> Dict[str, float]:
    env = PredPreyGrass(cfg)
    obs, info = env.reset(seed=seed)
    predator_counts: List[int] = []
    prey_counts: List[int] = []
    grass_counts: List[int] = []

    for step in range(horizon):
        action_dict = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(action_dict)
        predator_counts.append(env.current_num_predators)
        prey_counts.append(env.current_num_prey)
        grass_counts.append(len(env.grass_positions))
        if terminations.get("__all__") or truncations.get("__all__"):
            break

    if hasattr(env, "close"):
        env.close()

    def summary(stats: List[int]) -> Dict[str, float]:
        return {
            "final": float(stats[-1]) if stats else 0.0,
            "min": float(min(stats)) if stats else 0.0,
            "avg": float(sum(stats) / len(stats)) if stats else 0.0,
        }

    return {
        "pred": summary(predator_counts),
        "prey": summary(prey_counts),
        "grass": summary(grass_counts),
        "steps": float(len(predator_counts)),
    }


def score_metrics(metrics: List[Dict[str, Dict[str, float]]], cfg: Dict) -> float:
    pred_target = cfg.get("n_initial_active_predator", 5)
    prey_target = cfg.get("n_initial_active_prey", 20)

    pred_final = statistics.mean(m["pred"]["final"] for m in metrics)
    prey_final = statistics.mean(m["prey"]["final"] for m in metrics)
    pred_min = min(m["pred"]["min"] for m in metrics)
    prey_min = min(m["prey"]["min"] for m in metrics)

    extinction_penalty = 0.0
    if pred_min <= 0:
        extinction_penalty += 5.0
    if prey_min <= 0:
        extinction_penalty += 5.0

    imbalance = abs(pred_final - prey_final) / max(1.0, prey_target)
    target_penalty = (
        abs(pred_final - pred_target) / max(1.0, pred_target)
        + abs(prey_final - prey_target) / max(1.0, prey_target)
    )

    return extinction_penalty + imbalance + target_penalty


def merge_config(base: Dict, overrides: Dict) -> Dict:
    cfg = deepcopy(base)
    cfg.update(overrides)
    if "world_width" in overrides and "world_height" not in overrides:
        cfg["world_height"] = overrides["world_width"]
    if "world_height" in overrides and "world_width" not in overrides:
        cfg["world_width"] = overrides["world_height"]
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser("Auto tune env parameters for random balance")
    parser.add_argument("--episodes-per-candidate", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--grid-file", type=str, default=None, help="JSON file that overrides the default grid")
    parser.add_argument("--limit", type=int, default=40, help="Max number of combinations to evaluate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="balanced_config.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    grid = deepcopy(DEFAULT_GRID)
    if args.grid_file:
        grid_path = Path(args.grid_file)
        with grid_path.open("r", encoding="utf-8") as fp:
            user_grid = json.load(fp)
        for key, values in user_grid.items():
            grid[key] = values

    combos = _iter_grid(grid, args.limit, args.seed)
    if not combos:
        raise ValueError("Grid is empty.")

    results = []
    for idx, overrides in enumerate(combos, 1):
        cfg = merge_config(config_env_base, overrides)
        metrics = []
        for episode in range(args.episodes_per_candidate):
            metrics.append(run_random_episode(cfg, args.horizon, seed=args.seed + episode * 17))
        score = score_metrics(metrics, cfg)
        results.append({
            "score": score,
            "overrides": overrides,
            "metrics": metrics,
        })
        print(f"[{idx}/{len(combos)}] overrides={overrides} score={score:.3f}")

    results.sort(key=lambda r: r["score"])
    top = results[: args.top_k]
    print("\nTop candidates:")
    for rank, item in enumerate(top, 1):
        print(f"#{rank}: score={item['score']:.3f} overrides={item['overrides']}")

    best_cfg = merge_config(config_env_base, top[0]["overrides"])
    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(best_cfg, fp, indent=2, ensure_ascii=False)
    print(f"\nBest config written to {output_path.resolve()}")


if __name__ == "__main__":
    main()
