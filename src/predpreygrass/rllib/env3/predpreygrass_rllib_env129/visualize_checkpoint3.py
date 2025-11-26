#!/usr/bin/env python3
"""Visualize trained policy (legacy Policy API)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from predpreygrass.rllib.env3.predpreygrass_rllib_env129.predpreygrass_rllib_env import (
    PredPreyGrass,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.prey_test_config import (
    prey_test_config,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.visualizer import (
    PredPreyVisualizer,
)


def infer_policy_id(agent_id: str) -> str:
    return "predator_policy" if "predator" in agent_id else "prey_policy"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualise a trained PredPreyGrass policy checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="RLlib checkpoint directory")
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--explore", action="store_true")
    parser.add_argument(
        "--env-config-file",
        type=Path,
        default=None,
        help="Optional JSON file overriding env config (usually logs/run_*/env_config.json).",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    def _env_creator(env_config):
        return PredPreyGrass(dict(env_config))

    for env_name in ("PredPreyGrass-continuous", "PredPreyGrass-discrete"):
        try:
            register_env(env_name, _env_creator)
        except Exception:
            pass

    algo = Algorithm.from_checkpoint(str(checkpoint_path))

    policies = {}
    for pid in ("predator_policy", "prey_policy"):
        policy = algo.get_policy(pid)
        if policy is None:
            raise RuntimeError(f"Policy {pid} not found in checkpoint.")
        policies[pid] = policy

    env_config = dict(prey_test_config)
    algo_env_cfg = algo.config.get("env_config", {})
    if algo_env_cfg:
        env_config.update(algo_env_cfg)
    if args.env_config_file:
        cfg_path = args.env_config_file.expanduser().resolve()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Env config file not found: {cfg_path}")
        env_config.update(json.loads(cfg_path.read_text(encoding="utf-8")))
    env_config["debug_logging"] = False
    env = PredPreyGrass(env_config)

    latest_obs: Dict[str, np.ndarray] = {}
    original_reset = env.reset

    def reset_and_capture(*reset_args, **reset_kwargs):
        observations, info = original_reset(*reset_args, **reset_kwargs)
        latest_obs.clear()
        latest_obs.update(observations)
        return observations, info

    env.reset = reset_and_capture  # type: ignore
    observations, _ = env.reset(seed=args.seed)
    latest_obs.update(observations)

    visualizer = PredPreyVisualizer(env, fps=args.fps)
    step_count = 0
    frame_interval = 1.0 / max(args.fps, 1)
    last_frame_time = time.time()

    try:
        while visualizer.render() and step_count < args.max_steps:
            if visualizer.paused:
                time.sleep(0.01)
                continue

            now = time.time()
            elapsed = now - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.time()

            actions: Dict[str, np.ndarray] = {}

            for agent_id in list(env.agents):
                obs_vector = latest_obs.get(agent_id)
                if obs_vector is None:
                    continue
                policy_id = infer_policy_id(agent_id)
                policy = policies[policy_id]
                action, _, _ = policy.compute_single_action(
                    obs_vector,
                    explore=args.explore,
                    unsquash_action=False,
                    clip_action=False,
                )
                actions[agent_id] = np.asarray(action, dtype=np.float32)

            new_obs, rewards, terminations, truncations, infos = env.step(actions)
            latest_obs.update(new_obs)
            for agent_id in list(latest_obs.keys()):
                if agent_id not in env.agents:
                    latest_obs.pop(agent_id, None)

            step_count += 1
            if truncations.get("__all__", False) or terminations.get("__all__", False):
                break

        print(f"[INFO] Simulation finished after {step_count} steps.")

    finally:
        visualizer.close()
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
