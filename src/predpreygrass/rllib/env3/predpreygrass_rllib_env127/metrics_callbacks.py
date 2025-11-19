"""
Callbacks to pull environment-provided episode metrics into Ray Tune logs.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from ray.rllib.algorithms.callbacks import DefaultCallbacks


EPS = 1e-8


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) < EPS:
        return float("nan")
    return numerator / denominator


def _compute_episode_metrics(raw: Dict[str, float]) -> Dict[str, float]:
    """Compute derived indicators from the raw env payload."""
    metrics = {}

    T = float(raw.get("episode_steps", 0.0))
    if T <= 0:
        return metrics

    sum_n_prey = float(raw.get("prey_sum_N", 0.0))
    sum_n_pred = float(raw.get("pred_sum_N", 0.0))
    avg_n_prey = sum_n_prey / T
    avg_n_pred = sum_n_pred / T

    metrics["prey_eat_rate"] = _safe_div(
        float(raw.get("prey_eat_grass_count", 0.0)), T * avg_n_prey + EPS
    )
    metrics["catch_rate"] = _safe_div(
        float(raw.get("pred_catch_prey_count", 0.0)), T * avg_n_pred + EPS
    )

    metrics["energy_efficiency_prey"] = _safe_div(
        float(raw.get("prey_energy_gain_total", 0.0)),
        float(raw.get("prey_energy_cost_total", 0.0)) + EPS,
    )
    metrics["energy_efficiency_pred"] = _safe_div(
        float(raw.get("pred_energy_gain_total", 0.0)),
        float(raw.get("pred_energy_cost_total", 0.0)) + EPS,
    )

    metrics["lpvi_prey"] = _safe_div(
        sum_n_prey, T * (float(raw.get("prey_initial_count", 0.0)) + EPS)
    )
    metrics["lpvi_pred"] = _safe_div(
        sum_n_pred, T * (float(raw.get("pred_initial_count", 0.0)) + EPS)
    )

    metrics["avg_lifespan_prey"] = _safe_div(
        float(raw.get("prey_lifespan_sum", 0.0)),
        float(raw.get("prey_lifespan_count", 0.0)) + EPS,
    )
    metrics["avg_lifespan_pred"] = _safe_div(
        float(raw.get("pred_lifespan_sum", 0.0)),
        float(raw.get("pred_lifespan_count", 0.0)) + EPS,
    )

    metrics["avg_per_capita_energy_prey"] = _safe_div(
        float(raw.get("prey_sum_E_div_N", 0.0)), T
    )
    metrics["avg_per_capita_energy_pred"] = _safe_div(
        float(raw.get("pred_sum_E_div_N", 0.0)), T
    )

    mean_n_prey = avg_n_prey
    var_n_prey = _safe_div(float(raw.get("prey_sum_N2", 0.0)), T) - mean_n_prey**2
    var_n_prey = max(var_n_prey, 0.0)
    metrics["cv_prey"] = _safe_div(math.sqrt(var_n_prey), mean_n_prey + EPS)

    mean_n_pred = avg_n_pred
    var_n_pred = _safe_div(float(raw.get("pred_sum_N2", 0.0)), T) - mean_n_pred**2
    var_n_pred = max(var_n_pred, 0.0)
    metrics["cv_pred"] = _safe_div(math.sqrt(var_n_pred), mean_n_pred + EPS)

    metrics["grass_balance"] = _safe_div(
        float(raw.get("grass_regrown_count", 0.0))
        - float(raw.get("grass_consumed_count", 0.0)),
        float(raw.get("grass_consumed_count", 0.0)) + EPS,
    )

    # Extinction steps (keep raw value)
    metrics["prey_extinction_step"] = float(
        raw.get("prey_extinct_step")
        if raw.get("prey_extinct_step") is not None
        else math.nan
    )
    metrics["pred_extinction_step"] = float(
        raw.get("pred_extinct_step")
        if raw.get("pred_extinct_step") is not None
        else math.nan
    )
    metrics["episode_steps"] = T
    return metrics


class EpisodeMetricsCallbacks(DefaultCallbacks):
    """Callback that pushes env episode metrics into custom_metrics."""

    def on_episode_end(self, *, episode, **kwargs) -> None:

        metrics_payload: Optional[Dict[str, float]] = None
        try:
            agent_ids = episode.get_agents()
        except AttributeError:
            agent_ids = episode._agent_to_last_info.keys()

        for agent_id in agent_ids:
            info = episode.last_info_for(agent_id)
            if info and "episode_metrics" in info:
                metrics_payload = info["episode_metrics"]
                break

        if metrics_payload is None and hasattr(episode, "last_infos"):
            for info in episode.last_infos.values():
                if info and "episode_metrics" in info:
                    metrics_payload = info["episode_metrics"]
                    break

        if not metrics_payload:
            return

        derived = _compute_episode_metrics(metrics_payload)
        for name, value in derived.items():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            episode.custom_metrics[f"eval_{name}"] = float(value)
