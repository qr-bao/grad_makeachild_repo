"""Optuna-based environment parameter tuning for PredPreyGrass."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

from predpreygrass.rllib.env3.predpreygrass_rllib_env129.predpreygrass_rllib_env import (
    PredPreyGrass,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.config.config_env_base import (
    config_env_base,
)


DEFAULT_TARGET_GRASS = 120.0
EXTINCTION_WEIGHT = 5.0
GRASS_WEIGHT = 0.15


def run_random_episode(cfg: Dict[str, Any], horizon: int, seed: int) -> Dict[str, Any]:
    env = PredPreyGrass(cfg)
    obs, _ = env.reset(seed=seed)
    predator_counts: List[int] = []
    prey_counts: List[int] = []
    grass_counts: List[int] = []

    for step in range(horizon):
        actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        predator_counts.append(env.current_num_predators)
        prey_counts.append(env.current_num_prey)
        grass_counts.append(len(env.grass_positions))
        if terminations.get("__all__") or truncations.get("__all__"):
            break

    if hasattr(env, "close"):
        env.close()

    def summary(values: List[int]) -> Dict[str, float]:
        if not values:
            return {"final": 0.0, "min": 0.0, "avg": 0.0}
        return {
            "final": float(values[-1]),
            "min": float(min(values)),
            "avg": float(sum(values) / len(values)),
        }

    return {
        "pred": summary(predator_counts),
        "prey": summary(prey_counts),
        "grass": summary(grass_counts),
        "steps": len(predator_counts),
    }


def score_metrics(cfg: Dict[str, Any], metrics: List[Dict[str, Any]]) -> float:
    if not metrics:
        return float("inf")

    pred_target = max(1.0, float(cfg.get("n_initial_active_predator", 1)))
    prey_target = max(1.0, float(cfg.get("n_initial_active_prey", 1)))

    pred_final_avg = statistics.mean(m["pred"]["final"] for m in metrics)
    prey_final_avg = statistics.mean(m["prey"]["final"] for m in metrics)
    grass_avg = statistics.mean(m["grass"]["avg"] for m in metrics)

    extinction_penalty = 0.0
    for m in metrics:
        if m["pred"]["min"] <= 0:
            extinction_penalty += EXTINCTION_WEIGHT
        if m["prey"]["min"] <= 0:
            extinction_penalty += EXTINCTION_WEIGHT

    imbalance = abs(pred_final_avg - prey_final_avg) / prey_target
    target_penalty = (
        abs(pred_final_avg - pred_target) / pred_target
        + abs(prey_final_avg - prey_target) / prey_target
    )
    grass_target = cfg.get("initial_num_grass", DEFAULT_TARGET_GRASS)
    grass_penalty = GRASS_WEIGHT * abs(grass_avg - grass_target) / max(1.0, grass_target)

    return extinction_penalty + imbalance + target_penalty + grass_penalty


def build_config_from_trial(trial: optuna.trial.Trial) -> Dict[str, Any]:
    cfg = deepcopy(config_env_base)

    world_width = trial.suggest_int("world_width", 400, 520, step=20)
    world_height = trial.suggest_int("world_height", 400, 520, step=20)
    cfg["world_width"] = world_width
    cfg["world_height"] = world_height
    cfg["agent_radius"] = trial.suggest_float("agent_radius", 5.0, 8.0, step=0.5)

    cfg["n_initial_active_predator"] = trial.suggest_int("n_pred", 4, 10)
    cfg["n_initial_active_prey"] = trial.suggest_int("n_prey", 18, 36)

    cfg["initial_num_grass"] = trial.suggest_int("initial_grass", 80, 200)
    cfg["enable_grass_reproduction"] = True
    cfg["grass_respawn_delay"] = trial.suggest_int("grass_respawn", 80, 220, step=10)
    cfg["grass_offspring_energy"] = trial.suggest_float("grass_offspring_energy", 6.0, 15.0)

    cfg["energy_loss_per_step_predator"] = trial.suggest_float(
        "energy_loss_pred", 0.10, 0.20
    )
    cfg["energy_loss_per_step_prey"] = trial.suggest_float(
        "energy_loss_prey", 0.05, 0.11
    )
    cfg["reward_predator_catch_prey"] = trial.suggest_float(
        "reward_catch", 2.5, 5.5
    )
    cfg["reward_prey_step"] = trial.suggest_float("reward_prey_step", -0.014, -0.006)
    cfg["survival_bonus_predator"] = trial.suggest_float(
        "survival_bonus_pred", 0.0, 0.2
    )

    cfg["reproduction_energy_ratio_predator"] = trial.suggest_float(
        "repro_ratio_pred", 0.3, 0.6
    )
    cfg["reproduction_energy_ratio_prey"] = trial.suggest_float(
        "repro_ratio_prey", 0.4, 0.7
    )
    cfg["reproduction_cooldown_predator"] = trial.suggest_int(
        "repro_cooldown_pred", 10, 40
    )
    cfg["reproduction_cooldown_prey"] = trial.suggest_int(
        "repro_cooldown_prey", 60, 120
    )

    cfg["max_steps"] = max(cfg.get("max_steps", 3000), 3000)
    return cfg


def objective(trial: optuna.trial.Trial, args: argparse.Namespace) -> float:
    cfg = build_config_from_trial(trial)
    metrics = []
    try:
        for episode in range(args.episodes_per_trial):
            seed = args.seed + trial.number * 131 + episode * 17
            metrics.append(run_random_episode(cfg, args.horizon, seed))
    except Exception as exc:
        trial.set_user_attr("error", str(exc))
        return float("inf")

    score = score_metrics(cfg, metrics)
    trial.set_user_attr("config", cfg)
    trial.set_user_attr("metrics", metrics)
    trial.set_user_attr("score", score)
    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuning for PredPreyGrass env")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--episodes-per-trial", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--study", type=str, default="sqlite:///optuna_env_balance.db",
        help="Optuna storage URI (use 'none' for in-memory)",
    )
    parser.add_argument("--study-name", type=str, default="env_balance")
    parser.add_argument("--output", type=str, default="predpreygrass_rllib_env129/config/balanced_config_optuna.json")
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.study.lower() == "none":
        storage = None
    else:
        storage = args.study

    sampler = TPESampler(seed=args.seed)
    pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=2)

    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    best = study.best_trial
    best_cfg = best.user_attrs.get("config")
    if best_cfg is None:
        print("No best configuration recorded.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(best_cfg, fp, indent=2, ensure_ascii=False)

    print(f"Best score: {best.value:.4f}")
    print(f"Best config written to {output_path.resolve()}")


if __name__ == "__main__":
    main()
