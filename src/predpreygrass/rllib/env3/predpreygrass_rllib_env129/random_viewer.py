#!/usr/bin/env python3
"""可视化随机策略在环境中的表现。"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np

# Allow running this file directly without installing the package by ensuring the
# repository `src/` root is on sys.path.
import sys

_SRC_ROOT = Path(__file__).resolve().parents[4]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from predpreygrass.rllib.env3.predpreygrass_rllib_env129.predpreygrass_rllib_env import (
    PredPreyGrass,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.config.config_env_base import (
    config_env_base,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.prey_test_config import (
    prey_test_config,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.visualizer import (
    PredPreyVisualizer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Random agent visualiser for PredPreyGrass.")
    parser.add_argument(
        "--env-config-file",
        type=Path,
        default=None,
        help="Optional JSON file overriding environment config.",
    )
    parser.add_argument(
        "--base-config",
        choices=("env_base", "prey_test"),
        default="env_base",
        help="Which built-in config to start from before applying --env-config-file overrides.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50000,
        help="Maximum simulation steps before exiting.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frame rate for the visualiser.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Environment reset seed.",
    )
    return parser


def load_env_config(path: Path | None, *, base: str) -> dict:
    config = dict(config_env_base if base == "env_base" else prey_test_config)
    if path is not None:
        cfg_path = path.expanduser().resolve()
        if cfg_path.is_dir():
            cfg_path = cfg_path / "best_cfg.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Env config file not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as fp:
            extra = json.load(fp)
        config.update(extra)
    config["debug_logging"] = False
    return config


def main() -> None:
    args = build_parser().parse_args()
    env_config = load_env_config(args.env_config_file, base=args.base_config)
    # 确保环境本身的 max_steps 也同步到命令行设置，避免 500 步自动截断
    env_config["max_steps"] = args.max_steps

    env = PredPreyGrass(env_config)
    observations, _ = env.reset(seed=args.seed)

    visualizer = PredPreyVisualizer(env, fps=args.fps)
    latest_obs: Dict[str, np.ndarray] = dict(observations)

    frame_interval = 1.0 / max(args.fps, 1)
    last_frame_time = time.time()
    step_count = 0

    try:
        while visualizer.render() and step_count < args.max_steps:
            if visualizer.paused:
                time.sleep(0.05)
                continue

            now = time.time()
            elapsed = now - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.time()

            actions = {
                agent_id: env.action_spaces[agent_id].sample()
                for agent_id in env.agents
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            latest_obs.update(observations)
            for agent_id in list(latest_obs.keys()):
                if agent_id not in env.agents:
                    latest_obs.pop(agent_id, None)

            step_count += 1
            if truncations.get("__all__", False) or terminations.get("__all__", False):
                break
    finally:
        visualizer.close()


if __name__ == "__main__":
    main()
