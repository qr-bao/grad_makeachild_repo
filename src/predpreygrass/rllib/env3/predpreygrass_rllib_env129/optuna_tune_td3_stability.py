"""Optuna tuning for SimpleTD3 stability in PredPreyGrass orchestrator.

Goal: prefer non-extinction / stable ecology over raw reward.
This script tunes TD3 hyperparameters while holding the other side random.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import optuna
from optuna.samplers import TPESampler

# Allow running this file directly without installing the package by ensuring the
# repository `src/` root is on sys.path.
_SRC_ROOT = Path(__file__).resolve().parents[4]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from predpreygrass.rllib.env3.predpreygrass_rllib_env129.config.config_env_train import (
    config_env as config_env_train,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.orchestrator_multi_algo import (
    orchestrate,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.predpreygrass_rllib_env import (
    PredPreyGrass,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _derive_metrics(raw: Dict[str, Any]) -> Dict[str, float]:
    eps = 1e-8
    out: Dict[str, float] = {}
    T = float(raw.get("episode_steps", 0.0))
    if T <= 0:
        return out

    sum_n_prey = float(raw.get("prey_sum_N", 0.0))
    sum_n_pred = float(raw.get("pred_sum_N", 0.0))
    avg_n_prey = sum_n_prey / T
    avg_n_pred = sum_n_pred / T

    out["prey_eat_rate"] = float(raw.get("prey_eat_grass_count", 0.0)) / (T * avg_n_prey + eps)
    out["catch_rate"] = float(raw.get("pred_catch_prey_count", 0.0)) / (T * avg_n_pred + eps)

    out["energy_efficiency_prey"] = float(raw.get("prey_energy_gain_total", 0.0)) / (
        float(raw.get("prey_energy_cost_total", 0.0)) + eps
    )
    out["energy_efficiency_pred"] = float(raw.get("pred_energy_gain_total", 0.0)) / (
        float(raw.get("pred_energy_cost_total", 0.0)) + eps
    )

    mean_n_prey = avg_n_prey
    var_n_prey = float(raw.get("prey_sum_N2", 0.0)) / T - mean_n_prey**2
    var_n_prey = max(var_n_prey, 0.0)
    out["cv_prey"] = math.sqrt(var_n_prey) / (mean_n_prey + eps)

    mean_n_pred = avg_n_pred
    var_n_pred = float(raw.get("pred_sum_N2", 0.0)) / T - mean_n_pred**2
    var_n_pred = max(var_n_pred, 0.0)
    out["cv_pred"] = math.sqrt(var_n_pred) / (mean_n_pred + eps)

    out["grass_balance"] = (
        float(raw.get("grass_regrown_count", 0.0)) - float(raw.get("grass_consumed_count", 0.0))
    ) / (float(raw.get("grass_consumed_count", 0.0)) + eps)

    out["prey_ext_step"] = float(raw.get("prey_extinct_step") or math.nan)
    out["pred_ext_step"] = float(raw.get("pred_extinct_step") or math.nan)
    out["episode_steps"] = T
    return out


def _stability_score(
    derived: Dict[str, float],
    *,
    max_steps: int,
) -> float:
    # Higher is better.
    # Extinction: None -> NaN -> treat as max_steps (best).
    prey_ext = derived.get("prey_ext_step", math.nan)
    pred_ext = derived.get("pred_ext_step", math.nan)
    prey_ext = float(prey_ext) if not math.isnan(prey_ext) else float(max_steps)
    pred_ext = float(pred_ext) if not math.isnan(pred_ext) else float(max_steps)
    ext_norm = 0.5 * (prey_ext / max_steps + pred_ext / max_steps)

    cv = float(derived.get("cv_prey", 0.0) + derived.get("cv_pred", 0.0))
    grass_balance = abs(float(derived.get("grass_balance", 0.0)))
    eat = float(derived.get("prey_eat_rate", 0.0))
    catch = float(derived.get("catch_rate", 0.0))
    eff = float(derived.get("energy_efficiency_prey", 0.0) + derived.get("energy_efficiency_pred", 0.0))

    # Weights tuned for "no-extinction first" behaviour.
    return 10.0 * ext_norm + 0.5 * (eat + catch) + 0.25 * eff - 0.25 * cv - 0.1 * grass_balance


def _build_mapping(n_pops: int, role: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    role = role.lower()
    for i in range(n_pops):
        mapping[f"predator_{i}"] = "td3" if role in ("predator", "both") else "random"
        mapping[f"prey_{i}"] = "td3" if role in ("prey", "both") else "random"
    return mapping


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna tuning for TD3 stability (orchestrator)")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--train-episodes", type=int, default=15)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--role", type=str, default="predator", choices=("predator", "prey", "both"))
    p.add_argument("--env-config-file", type=str, default=None)
    p.add_argument("--log-dir", type=str, default="logs/optuna_td3")
    p.add_argument("--study", type=str, default="sqlite:///optuna_td3_stability.db")
    p.add_argument("--study-name", type=str, default="td3_stability")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_log_dir = Path(args.log_dir).expanduser().resolve()
    base_log_dir.mkdir(parents=True, exist_ok=True)

    env_config = deepcopy(config_env_train)
    if args.env_config_file:
        override_path = Path(args.env_config_file).expanduser().resolve()
        if override_path.is_file():
            env_config.update(_read_json(override_path))
    env_config["max_steps"] = int(args.max_steps)
    env_config["verbose_spawning"] = False
    env_config["verbose_engagement"] = False
    env_config["verbose_movement"] = False

    sample_env = PredPreyGrass(env_config)
    n_pops = int(getattr(sample_env, "n_populations", env_config.get("n_populations", 1)))
    if hasattr(sample_env, "close"):
        sample_env.close()

    mapping = _build_mapping(n_pops, args.role)

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=args.study,
        study_name=args.study_name,
        load_if_exists=True,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        td3_hparams = {
            "actor_lr": trial.suggest_float("actor_lr", 1e-5, 5e-4, log=True),
            "critic_lr": trial.suggest_float("critic_lr", 3e-5, 2e-3, log=True),
            "gamma": trial.suggest_float("gamma", 0.95, 0.999),
            "tau": trial.suggest_float("tau", 0.001, 0.02),
            "policy_noise": trial.suggest_float("policy_noise", 0.05, 0.3),
            "noise_clip": trial.suggest_float("noise_clip", 0.05, 0.5),
            "policy_delay": trial.suggest_int("policy_delay", 1, 4),
        }
        td3_train = {
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "warmup": trial.suggest_int("warmup", 32, 512, step=32),
            "exploration_noise_std": trial.suggest_float("exploration_noise_std", 0.0, 0.2),
            "buffer_size": trial.suggest_int("buffer_size", 10_000, 80_000, step=10_000),
            "updates_per_episode": trial.suggest_int("updates_per_episode", 1, 20),
        }

        trial_dir = base_log_dir / f"trial_{trial.number:04d}"
        train_json = trial_dir / "train.json"
        eval_json = trial_dir / "eval.json"
        ckpt_dir = trial_dir / "checkpoints"
        trial_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        try:
            orchestrate(
                mapping,
                env_config,
                args.train_episodes,
                args.max_steps,
                trial_dir / "tb_train",
                ckpt_dir,
                checkpoint_freq=0,
                mode="train",
                seed=args.seed + trial.number * 1000,
                export_json=str(train_json),
                td3_hparams=td3_hparams,
                td3_train=td3_train,
            )
            orchestrate(
                mapping,
                env_config,
                args.eval_episodes,
                args.max_steps,
                trial_dir / "tb_eval",
                ckpt_dir,
                checkpoint_freq=0,
                mode="eval",
                seed=args.seed + trial.number * 1000 + 777,
                export_json=str(eval_json),
                td3_hparams=td3_hparams,
                td3_train=td3_train,
            )
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            return -1e9

        data = _read_json(eval_json)
        episodes = data.get("episodes") or []
        scores: List[float] = []
        for ep in episodes:
            raw = ep.get("env_episode_metrics")
            if not raw:
                continue
            derived = _derive_metrics(raw)
            scores.append(_stability_score(derived, max_steps=args.max_steps))
        if not scores:
            return -1e9
        return float(statistics.mean(scores))

    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    print(f"Best value: {best.value}")
    print(f"Best params: {best.params}")
    out_path = base_log_dir / "best_td3_config.json"
    out_path.write_text(json.dumps(best.params, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
