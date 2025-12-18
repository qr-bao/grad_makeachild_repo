"""Optuna tuning for PredPreyGrass *environment balance* under random-vs-random.

Target (strict):
- 10 populations (predator_0..4 + prey_0..4) coexist without extinctions.
- Per-population counts spend most of the time in [min_n, max_n] (default [5, 50]).
- The system reaches a steady regime within `horizon` steps (default 5000), but
  evaluation can early-stop once stability is detected to save time.

Important: We tune only *dynamics* parameters (grass/energy/hunger/reproduction).
Reward weights are excluded because they do not affect random behaviour.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import statistics
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

# Allow running this file directly without installing the package by ensuring the
# repository `src/` root is on sys.path.
_SRC_ROOT = Path(__file__).resolve().parents[4]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from predpreygrass.rllib.env3.predpreygrass_rllib_env129.predpreygrass_rllib_env import (
    PredPreyGrass,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.config.config_env_base import (
    config_env_base,
)


DEFAULT_TARGET_GRASS = 150.0

# Extinction is irreversible with paired reproduction.
# Make extinction dominate the objective so Optuna prioritizes non-extinction
# regimes first; other penalties only matter once we have 0-extinction trials.
EXTINCTION_PENALTY = 10_000.0
EXTINCTION_EARLY_WEIGHT = 20_000.0

# Strict stability penalties (lower is better).
OUT_OF_RANGE_WEIGHT = 200.0
OUT_OF_RANGE_DISTANCE_WEIGHT = 30.0
CV_WEIGHT = 5.0
GRASS_WEIGHT = 0.5
STABILITY_TIME_WEIGHT = 20.0


@dataclass
class EpisodeRolloutSummary:
    steps: int
    pop_keys: Tuple[str, ...]
    counts_by_key: Dict[str, List[int]]
    grass_counts: List[int]
    stability_step: Optional[int]
    early_stop_reason: Optional[str]
    extinct_keys: Tuple[str, ...]


def _ensure_pop_keys(n_pops: int) -> Tuple[str, ...]:
    keys: List[str] = []
    for i in range(n_pops):
        keys.append(f"predator_{i}")
    for i in range(n_pops):
        keys.append(f"prey_{i}")
    return tuple(keys)


def _get_pop_counts(env, pop_keys: Tuple[str, ...]) -> Dict[str, int]:
    pop_counts = getattr(env, "population_counts", {}) or {}
    return {k: int(pop_counts.get(k, 0)) for k in pop_keys}


def _window_stats(values: List[int]) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    mean = float(sum(values) / len(values))
    var = float(sum((v - mean) ** 2 for v in values) / max(1, len(values)))
    std = float(math.sqrt(max(0.0, var)))
    cv = float(std / (mean + 1e-8))
    return mean, std, cv


def _is_stable(
    counts_by_key: Dict[str, List[int]],
    *,
    min_n: int,
    max_n: int,
    window: int,
    min_hit_rate: float,
    drift_tol: float,
    cv_tol: float,
) -> bool:
    if not counts_by_key:
        return False
    steps = len(next(iter(counts_by_key.values())))
    if steps < 2 * window:
        return False
    lo = steps - window
    mid = steps - 2 * window
    for _, values in counts_by_key.items():
        recent = values[lo:steps]
        prev = values[mid:lo]
        if not recent or not prev:
            return False
        hit = sum(1 for v in recent if min_n <= v <= max_n) / float(len(recent))
        if hit < min_hit_rate:
            return False
        mean_recent, _, cv_recent = _window_stats(recent)
        mean_prev, _, _ = _window_stats(prev)
        if not (min_n <= mean_recent <= max_n):
            return False
        if abs(mean_recent - mean_prev) > drift_tol:
            return False
        if cv_recent > cv_tol:
            return False
    return True


def run_random_episode(
    cfg: Dict[str, Any],
    *,
    horizon: int,
    seed: int,
    pop_keys: Tuple[str, ...],
    min_n: int,
    max_n: int,
    report_interval: int,
    stability_window: int,
    min_hit_rate: float,
    drift_tol: float,
    cv_tol: float,
    trial: Optional[optuna.trial.Trial] = None,
    prune_at_step: int = 600,
) -> EpisodeRolloutSummary:
    env = PredPreyGrass(cfg)
    obs, _ = env.reset(seed=seed)
    counts_by_key: Dict[str, List[int]] = {k: [] for k in pop_keys}
    grass_counts: List[int] = []
    stability_step: Optional[int] = None
    early_stop_reason: Optional[str] = None
    extinct_keys: Tuple[str, ...] = ()

    for step in range(horizon):
        actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        pop_counts = _get_pop_counts(env, pop_keys)
        for k in pop_keys:
            counts_by_key[k].append(pop_counts[k])
        grass_counts.append(int(len(getattr(env, "grass_positions", {}) or {})))

        # Extinction is irreversible -> hard fail + early stop.
        if any(pop_counts[k] <= 0 for k in pop_keys):
            extinct_keys = tuple(k for k in pop_keys if pop_counts[k] <= 0)
            early_stop_reason = "extinction"
            break

        if (step + 1) % report_interval == 0:
            if stability_step is None and _is_stable(
                counts_by_key,
                min_n=min_n,
                max_n=max_n,
                window=stability_window,
                min_hit_rate=min_hit_rate,
                drift_tol=drift_tol,
                cv_tol=cv_tol,
            ):
                stability_step = step + 1
                early_stop_reason = "stable"
                break

            if trial is not None and (step + 1) >= prune_at_step:
                steps_so_far = step + 1
                outside = []
                outside_dist = []
                for k in pop_keys:
                    series = counts_by_key[k]
                    out = sum(1 for v in series if v < min_n or v > max_n) / float(steps_so_far)
                    outside.append(out)
                    dist = sum(
                        (min_n - v) if v < min_n else (v - max_n) if v > max_n else 0
                        for v in series
                    ) / float(steps_so_far * (max_n - min_n + 1))
                    outside_dist.append(dist)
                grass_target = float(cfg.get("initial_num_grass", DEFAULT_TARGET_GRASS))
                grass_avg = float(sum(grass_counts) / len(grass_counts))
                grass_pen = abs(grass_avg - grass_target) / max(1.0, grass_target)
                interim = (
                    OUT_OF_RANGE_WEIGHT * float(statistics.mean(outside))
                    + OUT_OF_RANGE_DISTANCE_WEIGHT * float(statistics.mean(outside_dist))
                    + GRASS_WEIGHT * grass_pen
                )
                trial.report(interim, step + 1)
                if trial.should_prune():
                    early_stop_reason = "pruned"
                    raise optuna.TrialPruned()

        if terminations.get("__all__") or truncations.get("__all__"):
            break

    if hasattr(env, "close"):
        env.close()

    steps = len(grass_counts)
    return EpisodeRolloutSummary(
        steps=steps,
        pop_keys=pop_keys,
        counts_by_key=counts_by_key,
        grass_counts=grass_counts,
        stability_step=stability_step,
        early_stop_reason=early_stop_reason,
        extinct_keys=extinct_keys,
    )


def _episode_score(
    cfg: Dict[str, Any],
    rollout: EpisodeRolloutSummary,
    *,
    horizon: int,
    min_n: int,
    max_n: int,
) -> float:
    if rollout.steps <= 0:
        return float("inf")

    extinct = 0
    for _, series in rollout.counts_by_key.items():
        if series and min(series) <= 0:
            extinct += 1
    if extinct:
        # Break ties: earlier extinction is worse (close to 0 steps -> large add-on).
        # This makes Optuna search meaningful even when most trials die early.
        t = min(max(float(rollout.steps) / float(horizon), 0.0), 1.0)
        early = EXTINCTION_EARLY_WEIGHT * (1.0 - t)
        # Still include range pressure on the (partial) trajectory.
        outside_frac = []
        outside_dist = []
        for _, series in rollout.counts_by_key.items():
            steps = len(series)
            if steps <= 0:
                continue
            out = sum(1 for v in series if v < min_n or v > max_n) / float(steps)
            outside_frac.append(out)
            dist = sum(
                (min_n - v) if v < min_n else (v - max_n) if v > max_n else 0
                for v in series
            ) / float(steps * (max_n - min_n + 1))
            outside_dist.append(dist)
        partial_pen = (
            (OUT_OF_RANGE_WEIGHT * float(statistics.mean(outside_frac))) if outside_frac else 0.0
        ) + (
            (OUT_OF_RANGE_DISTANCE_WEIGHT * float(statistics.mean(outside_dist))) if outside_dist else 0.0
        )
        return EXTINCTION_PENALTY * extinct + early + partial_pen

    outside_frac = []
    outside_dist = []
    cvs = []
    for _, series in rollout.counts_by_key.items():
        steps = len(series)
        if steps <= 0:
            continue
        out = sum(1 for v in series if v < min_n or v > max_n) / float(steps)
        outside_frac.append(out)
        dist = sum(
            (min_n - v) if v < min_n else (v - max_n) if v > max_n else 0
            for v in series
        ) / float(steps * (max_n - min_n + 1))
        outside_dist.append(dist)
        _, _, cv = _window_stats(series)
        cvs.append(cv)

    grass_target = float(cfg.get("initial_num_grass", DEFAULT_TARGET_GRASS))
    grass_avg = float(sum(rollout.grass_counts) / len(rollout.grass_counts))
    grass_penalty = abs(grass_avg - grass_target) / max(1.0, grass_target)

    stability_penalty = STABILITY_TIME_WEIGHT
    if rollout.stability_step is not None:
        stability_penalty = STABILITY_TIME_WEIGHT * (rollout.stability_step / float(horizon))

    return (
        OUT_OF_RANGE_WEIGHT * float(statistics.mean(outside_frac)) if outside_frac else 0.0
    ) + (
        OUT_OF_RANGE_DISTANCE_WEIGHT * float(statistics.mean(outside_dist)) if outside_dist else 0.0
    ) + (
        CV_WEIGHT * float(statistics.mean(cvs)) if cvs else 0.0
    ) + GRASS_WEIGHT * grass_penalty + stability_penalty


def build_config_from_trial(trial: optuna.trial.Trial) -> Dict[str, Any]:
    cfg = deepcopy(config_env_base)

    # Keep totals in a "middle" range; we do NOT cap populations, we just avoid
    # trivial extremes that are unlikely to stabilize under random actions.
    cfg["n_populations"] = int(cfg.get("n_populations", 5))
    # With 5 populations, this corresponds roughly to:
    # - predators: 8..20 per pop
    # - prey: 30..50 per pop
    cfg["n_initial_active_predator"] = trial.suggest_int("n_pred_total", 25, 75, step=5)
    cfg["n_initial_active_prey"] = trial.suggest_int("n_prey_total", 175, 250, step=5)

    # Density controls encounter rates.
    cfg["world_width"] = trial.suggest_int("world_width", 550, 750, step=50)
    cfg["world_height"] = trial.suggest_int("world_height", 550, 750, step=50)
    cfg["agent_radius"] = trial.suggest_float("agent_radius", 3.0, 5.5, step=0.5)

    # Movement controls encounter rates under random actions.
    cfg["use_direct_velocity_control"] = True
    cfg["direct_velocity_speed_predator"] = trial.suggest_int("speed_pred", 150, 275, step=25)
    cfg["direct_velocity_speed_prey"] = trial.suggest_int("speed_prey", 200, 450, step=25)
    cfg["direct_velocity_accel_scale_predator"] = trial.suggest_int("accel_pred", 20, 80, step=10)
    cfg["direct_velocity_accel_scale_prey"] = trial.suggest_int("accel_prey", 20, 80, step=10)

    # Wall impacts can kill random agents quickly; tune these down for stability.
    cfg["enable_agent_collision"] = False
    cfg["wall_collision_damage"] = trial.suggest_float("wall_collision_damage", 0.0, 0.08)
    cfg["collision_damage"] = trial.suggest_float("collision_damage", 0.0, 0.08)

    # Grass dynamics (env129 uses fixed_grass_mode by default).
    cfg["fixed_grass_mode"] = True
    cfg["initial_num_grass"] = trial.suggest_int("initial_grass", 250, 550, step=25)
    cfg["initial_energy_grass"] = trial.suggest_float("initial_energy_grass", 0.8, 4.0)
    cfg["energy_gain_per_step_grass"] = trial.suggest_float("grass_growth", 0.002, 0.03, log=True)
    cfg["grass_respawn_delay"] = trial.suggest_int("grass_respawn_delay", 25, 150, step=25)

    # Energy balance.
    cfg["initial_energy_predator"] = trial.suggest_float("init_e_pred", 2.0, 8.0)
    cfg["initial_energy_prey"] = trial.suggest_float("init_e_prey", 2.0, 8.0)
    cfg["energy_loss_per_step_predator"] = trial.suggest_float("loss_pred", 0.0, 0.01)
    cfg["energy_loss_per_step_prey"] = trial.suggest_float("loss_prey", 0.0, 0.01)
    cfg["energy_transfer_efficiency_predator"] = trial.suggest_float("eff_pred", 0.35, 0.7)
    cfg["energy_transfer_efficiency_prey"] = trial.suggest_float("eff_prey", 0.8, 0.95)

    # A physiological cap (NOT a population cap) helps avoid runaway energy ->
    # runaway reproduction dynamics under random play.
    cfg["enable_max_energy"] = True
    cfg["max_energy_predator"] = trial.suggest_float("max_e_pred", 25.0, 90.0, step=5.0)
    cfg["max_energy_prey"] = trial.suggest_float("max_e_prey", 12.0, 30.0, step=2.0)

    # Hunger (adds pressure when food access is poor).
    cfg["enable_hunger"] = True
    cfg["max_steps_without_food_predator"] = trial.suggest_int("hungry_steps_pred", 400, 1200, step=50)
    cfg["max_steps_without_food_prey"] = trial.suggest_int("hungry_steps_prey", 600, 1500, step=50)
    cfg["hunger_damage"] = trial.suggest_float("hunger_damage", 0.0, 0.00005)

    # Reproduction dynamics.
    cfg["enable_paired_reproduction"] = True
    # Critical: env129 defaults to `reproduction_mode="fixed_ratio"` with an
    # `offspring_min_energy` floor that can *create energy from nothing* and make
    # the random ecology explode/behave unrealistically. For "natural" balance
    # tuning we always use energy-conserving ratio mode.
    cfg["reproduction_mode_predator"] = "ratio"
    cfg["reproduction_mode_prey"] = "ratio"
    # Make sure the fixed_ratio knobs don't accidentally affect anything if the
    # env reads them elsewhere.
    cfg["reproduction_fixed_cost_predator"] = 0.0
    cfg["reproduction_fixed_cost_prey"] = 0.0
    cfg["reproduction_transfer_ratio_predator"] = 0.0
    cfg["reproduction_transfer_ratio_prey"] = 0.0
    cfg["offspring_min_energy_predator"] = 0.0
    cfg["offspring_min_energy_prey"] = 0.0
    # Under random movement, too-small `mating_distance` makes reproduction fail
    # stochastically per-population; we bias it larger to reduce "one-pop dies"
    # outcomes.
    cfg["mating_distance"] = trial.suggest_float("mating_distance", 120.0, 600.0)
    cfg["predator_creation_energy_threshold"] = trial.suggest_float("repro_threshold_pred", 10.0, 28.0)
    cfg["prey_creation_energy_threshold"] = trial.suggest_float("repro_threshold_prey", 6.0, 18.0)
    cfg["min_reproduction_age_predator"] = trial.suggest_int("repro_min_age_pred", 80, 260, step=10)
    cfg["min_reproduction_age_prey"] = trial.suggest_int("repro_min_age_prey", 10, 120, step=5)
    cfg["reproduction_energy_ratio_predator"] = trial.suggest_float("repro_ratio_pred", 0.15, 0.4)
    cfg["reproduction_energy_ratio_prey"] = trial.suggest_float("repro_ratio_prey", 0.15, 0.5)
    cfg["reproduction_cooldown_predator"] = trial.suggest_int("repro_cooldown_pred", 120, 450, step=10)
    cfg["reproduction_cooldown_prey"] = trial.suggest_int("repro_cooldown_prey", 20, 250, step=10)

    cfg["verbose_spawning"] = False
    cfg["verbose_engagement"] = False
    cfg["verbose_movement"] = False
    return cfg


def objective(trial: optuna.trial.Trial, args: argparse.Namespace) -> float:
    cfg = build_config_from_trial(trial)
    cfg["max_steps"] = max(int(cfg.get("max_steps", args.horizon)), int(args.horizon))
    n_pops = int(cfg.get("n_populations", 5))
    pop_keys = _ensure_pop_keys(n_pops)
    rollouts: List[EpisodeRolloutSummary] = []
    try:
        for episode in range(args.episodes_per_trial):
            seed = args.seed + trial.number * 131 + episode * 17
            rollouts.append(
                run_random_episode(
                    cfg,
                    horizon=args.horizon,
                    seed=seed,
                    pop_keys=pop_keys,
                    min_n=args.min_n,
                    max_n=args.max_n,
                    report_interval=args.report_interval,
                    stability_window=args.stability_window,
                    min_hit_rate=args.min_hit_rate,
                    drift_tol=args.drift_tol,
                    cv_tol=args.cv_tol,
                    trial=trial,
                    prune_at_step=args.prune_at_step,
                )
            )
    except optuna.TrialPruned:
        # Let Optuna mark this trial as pruned (do not convert to `inf`).
        raise
    except Exception as exc:
        trial.set_user_attr("error", repr(exc))
        return float("inf")

    scores = [
        _episode_score(cfg, r, horizon=args.horizon, min_n=args.min_n, max_n=args.max_n)
        for r in rollouts
    ]
    score = float(statistics.mean(scores)) if scores else float("inf")
    trial.set_user_attr("config", cfg)
    trial.set_user_attr(
        "rollout_summary",
        {
            "episodes": len(rollouts),
            "steps": [r.steps for r in rollouts],
            "stability_step": [r.stability_step for r in rollouts],
            "early_stop_reason": [r.early_stop_reason for r in rollouts],
            "extinct_keys": [list(r.extinct_keys) for r in rollouts],
        },
    )
    trial.set_user_attr("score", score)
    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuning for PredPreyGrass env")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--episodes-per-trial", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-n", type=int, default=5)
    parser.add_argument("--max-n", type=int, default=50)
    parser.add_argument("--report-interval", type=int, default=200)
    parser.add_argument("--prune-at-step", type=int, default=600)
    parser.add_argument("--stability-window", type=int, default=500)
    parser.add_argument("--min-hit-rate", type=float, default=0.95)
    parser.add_argument("--drift-tol", type=float, default=2.0)
    parser.add_argument("--cv-tol", type=float, default=0.35)
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
        # Ensure parent directory exists for sqlite storage paths.
        if isinstance(storage, str) and storage.startswith("sqlite:///"):
            raw_path = storage[len("sqlite:///") :]
            # sqlite:////abs/path.db keeps a leading slash in raw_path.
            if raw_path:
                db_path = Path(raw_path).expanduser()
                if db_path.parent and str(db_path.parent) not in ("", "."):
                    db_path.parent.mkdir(parents=True, exist_ok=True)

    sampler = TPESampler(seed=args.seed)
    pruner = SuccessiveHalvingPruner(min_resource=args.prune_at_step, reduction_factor=2)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_best_so_far(study: optuna.Study, _: optuna.trial.FrozenTrial) -> None:
        """Persist best config after each completed trial so evaluation can run mid-sweep."""
        try:
            best_cfg = study.best_trial.user_attrs.get("config")
        except Exception:
            return
        if not isinstance(best_cfg, dict):
            return
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fp:
            json.dump(best_cfg, fp, indent=2, ensure_ascii=False)
        tmp_path.replace(output_path)

    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    try:
        study.optimize(
            lambda trial: objective(trial, args),
            n_trials=args.trials,
            n_jobs=args.n_jobs,
            show_progress_bar=True,
            callbacks=[_write_best_so_far],
        )
    except KeyboardInterrupt:
        # Still write the current best config so users can evaluate without
        # rerunning the whole sweep.
        print("\nInterrupted. Saving best configuration so far...")

    try:
        best = study.best_trial
    except ValueError:
        print("No completed trials to export.")
        return

    best_cfg = best.user_attrs.get("config")
    if best_cfg is None:
        print("No best configuration recorded.")
        return

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(best_cfg, fp, indent=2, ensure_ascii=False)

    print(f"Best score: {best.value:.4f}")
    print(f"Best config written to {output_path.resolve()}")


if __name__ == "__main__":
    main()
