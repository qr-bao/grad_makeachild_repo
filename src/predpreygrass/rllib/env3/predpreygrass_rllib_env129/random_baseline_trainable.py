"""Ray Tune Trainable that logs pure random rollouts for PredPreyGrass."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict

import numpy as np
from ray import tune

from predpreygrass.rllib.env3.predpreygrass_rllib_env129.predpreygrass_rllib_env import (
    PredPreyGrass,
)


class RandomBaselineTrainable(tune.Trainable):
    """Runs random actions to collect baseline metrics and reports them via Tune."""

    def setup(self, config: Dict[str, Any]) -> None:
        env_config = config["env_config"]
        self.episodes_per_iteration = config.get("episodes_per_iteration", 1)
        self.seed = config.get("seed")
        self.env = PredPreyGrass(env_config)
        self.rng = np.random.default_rng(self.seed) if self.seed is not None else np.random.default_rng()

    def step(self) -> Dict[str, float]:
        episode_stats = []
        total_env_steps = 0

        for _ in range(self.episodes_per_iteration):
            reset_kwargs = {"seed": int(self.rng.integers(0, 1_000_000_000))}
            obs, _ = self.env.reset(**reset_kwargs)
            terminations = {"__all__": False}
            truncations = {"__all__": False}
            rewards_acc = defaultdict(float)
            steps_acc = defaultdict(int)  # 按策略统计累计步数（对齐 RLlib policy_reward_mean 定义）
            steps = 0

            while not (terminations.get("__all__", False) or truncations.get("__all__", False)):
                actions = {aid: self.env.action_spaces[aid].sample() for aid in obs.keys()}
                obs, rewards, terminations, truncations, _ = self.env.step(actions)
                for agent_id, reward in rewards.items():
                    team = "predator" if agent_id.startswith("predator") else "prey"
                    rewards_acc[team] += reward
                    steps_acc[team] += 1
                steps += 1

            rewards_acc["steps"] = steps
            rewards_acc["pred_steps"] = steps_acc.get("predator", 0)
            rewards_acc["prey_steps"] = steps_acc.get("prey", 0)
            total_env_steps += steps
            episode_stats.append(rewards_acc)

        # 对齐 RLlib：policy_reward_mean = 总 reward / 总 agent_steps（同一策略）
        total_pred_reward = sum(r.get("predator", 0.0) for r in episode_stats)
        total_prey_reward = sum(r.get("prey", 0.0) for r in episode_stats)
        total_pred_steps = sum(r.get("pred_steps", 0) for r in episode_stats)
        total_prey_steps = sum(r.get("prey_steps", 0) for r in episode_stats)

        mean_pred = total_pred_reward / max(total_pred_steps, 1)
        mean_prey = total_prey_reward / max(total_prey_steps, 1)

        mean_steps = sum(r.get("steps", 0.0) for r in episode_stats) / len(episode_stats)

        # 合成一个对称的 episode_reward_mean 供对比（简单平均两策略的 policy 平均回报）
        combined_reward = 0.5 * (mean_pred + mean_prey)
        return {
            "episode_reward_mean": combined_reward,
            "episode_len_mean": mean_steps,
            "policy_reward_mean/predator_policy": mean_pred,
            "policy_reward_mean/prey_policy": mean_prey,
            "episodes_this_iter": len(episode_stats),
            "timesteps_this_iter": total_env_steps,
        }

    def cleanup(self) -> None:
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
            self.env = None

    def save_checkpoint(self, checkpoint_dir: str) -> str:
        state = {"rng_state": self.rng.bit_generator.state}
        path = os.path.join(checkpoint_dir, "state.json")
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(state, fp)
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_path: str) -> None:
        path = (
            checkpoint_path
            if checkpoint_path.endswith(".json")
            else os.path.join(checkpoint_path, "state.json")
        )
        with open(path, "r", encoding="utf-8") as fp:
            state = json.load(fp)
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state
