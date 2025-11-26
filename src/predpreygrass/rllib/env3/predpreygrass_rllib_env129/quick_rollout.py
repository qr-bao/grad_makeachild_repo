"""Lightweight rollout driver for PredPreyGrass env129.

Runs a few episodes with a random policy (or action noise) to sanity check
reward scales + termination logic without booting up Ray.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np

from predpreygrass.rllib.env3.predpreygrass_rllib_env129.predpreygrass_rllib_env import (
    PredPreyGrass,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.config.config_env_train import (
    config_env as default_env_config,
)


def _extract_episode_metrics(infos: Dict[str, Dict[str, dict]]) -> Dict[str, float] | None:
    """Pull episode_metrics payload if environment attached it to infos."""
    common = infos.get("__common__")
    if common and "episode_metrics" in common:
        return common["episode_metrics"]
    for payload in infos.values():
        if "episode_metrics" in payload:
            return payload["episode_metrics"]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick PredPreyGrass rollouts")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--max-env-steps",
        type=int,
        default=None,
        help="Override env max_steps config for debugging",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for metrics",
    )
    parser.add_argument(
        "--policy",
        choices=["random", "zero"],
        default="random",
        help="Which simple policy to use (random actions or zero thrust).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step reward/event diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    env_conf = default_env_config.copy()
    if args.max_env_steps is not None:
        env_conf["max_steps"] = args.max_env_steps

    env = PredPreyGrass(env_conf)
    summary = {
        "episodes": args.episodes,
        "per_episode_returns": [],
        "per_episode_stats": [],
        "termination_reasons": defaultdict(int),
        "event_counts": defaultdict(int),
    }

    for ep in range(args.episodes):
        observations, _ = env.reset()
        done = False
        episode_reward = defaultdict(float)
        steps = 0
        prev_metrics_snapshot = {
            "prey_eat_grass_count": getattr(env.metrics, "prey_eat_grass_count", 0),
            "pred_catch_prey_count": getattr(env.metrics, "pred_catch_prey_count", 0),
        }
        per_step_info = []
        while not done:
            actions: Dict[str, np.ndarray] = {}
            for agent_id in env.agents:
                act_space = env.action_spaces[agent_id]
                if args.policy == "random":
                    actions[agent_id] = act_space.sample()
                else:
                    actions[agent_id] = np.zeros_like(act_space.sample())
            observations, rewards, terminations, truncations, infos = env.step(actions)
            for agent_id, reward in rewards.items():
                episode_reward[agent_id] += reward
            # Use metrics tracker to compute delta events for this step.
            current_metrics_snapshot = {
                "prey_eat_grass_count": getattr(env.metrics, "prey_eat_grass_count", 0),
                "pred_catch_prey_count": getattr(env.metrics, "pred_catch_prey_count", 0),
            }
            delta_grass = (
                current_metrics_snapshot["prey_eat_grass_count"]
                - prev_metrics_snapshot["prey_eat_grass_count"]
            )
            delta_catch = (
                current_metrics_snapshot["pred_catch_prey_count"]
                - prev_metrics_snapshot["pred_catch_prey_count"]
            )
            prev_metrics_snapshot = current_metrics_snapshot
            per_step_info.append(
                {
                    "step": steps,
                    "reward_sum": sum(rewards.values()),
                    "ate_grass": max(0, delta_grass),
                    "caught_prey": max(0, delta_catch),
                }
            )
            summary["event_counts"]["ate_grass"] += per_step_info[-1]["ate_grass"]
            summary["event_counts"]["caught_prey"] += per_step_info[-1]["caught_prey"]
            steps += 1
            done = terminations.get("__all__", False) or truncations.get("__all__", False)
        reason = "termination" if terminations.get("__all__", False) else "truncation"
        summary["termination_reasons"][reason] += 1
        total_reward = sum(episode_reward.values())
        summary["per_episode_returns"].append(
            {
                "episode": ep,
                "steps": steps,
                "total_reward": total_reward,
            }
        )
        metrics_payload = _extract_episode_metrics(infos)
        summary["per_episode_stats"].append(
            {
                "episode": ep,
                "mean_step_reward": np.mean([p["reward_sum"] for p in per_step_info]),
                "caught_events": sum(p["caught_prey"] for p in per_step_info),
                "grass_events": sum(p["ate_grass"] for p in per_step_info),
                "metrics": metrics_payload,
            }
        )
        if metrics_payload:
            summary["event_counts"]["ate_grass"] += metrics_payload.get(
                "prey_eat_grass_count", 0
            )
            summary["event_counts"]["caught_prey"] += metrics_payload.get(
                "pred_catch_prey_count", 0
            )
        print(
            f"Episode {ep}: steps={steps}, total_reward={total_reward:.2f}, "
            f"reason={reason}"
        )
        if args.verbose:
            for entry in per_step_info[:5]:
                print(
                    f"  step {entry['step']:4d} | "
                    f"reward={entry['reward_sum']:+.3f} "
                    f"grass={entry['ate_grass']} caught={entry['caught_prey']}"
                )

    mean_return = np.mean(
        [entry["total_reward"] for entry in summary["per_episode_returns"]]
    )
    mean_steps = np.mean([entry["steps"] for entry in summary["per_episode_returns"]])
    print("\nSummary:")
    print(f"  Episodes      : {summary['episodes']}")
    print(f"  Avg steps     : {mean_steps:.1f}")
    print(f"  Avg reward    : {mean_return:.2f}")
    print(f"  Terminations  : {dict(summary['termination_reasons'])}")
    print(f"  Event counts  : {dict(summary['event_counts'])}")

    if args.output:
        summary_to_dump = {
            **summary,
            "termination_reasons": dict(summary["termination_reasons"]),
            "event_counts": dict(summary["event_counts"]),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary_to_dump, indent=2))
        print(f"Metrics written to {args.output}")


if __name__ == "__main__":
    main()
