#!/usr/bin/env python3
"""Multi-agent Soft Actor-Critic launcher (predator=SAC, prey=SAC)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ray
from ray import air, tune
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env

from predpreygrass.rllib.env3.predpreygrass_rllib_env129.predpreygrass_rllib_env import (
    PredPreyGrass,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.metrics_callbacks import (
    EpisodeMetricsCallbacks,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--num-iterations", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--num-envs-per-worker", type=int, default=2)
    p.add_argument("--checkpoint-freq", type=int, default=20)
    p.add_argument("--keep-all-checkpoints", action="store_true")
    p.add_argument(
        "--env-config-file",
        type=Path,
        default=Path("predpreygrass_rllib_env129/config/config_env_base.json"),
    )
    p.add_argument("--max-env-steps", type=int, default=2000)
    p.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    p.add_argument("--train-batch-size", type=int, default=1024)
    p.add_argument("--rollout-fragment-length", type=int, default=64)
    p.add_argument("--learning-starts", type=int, default=20_000)
    p.add_argument("--target-network-update-freq", type=int, default=0)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--entropy-coeff-auto", action="store_true")
    p.add_argument("--entropy-coeff", type=float, default=0.2)
    p.add_argument("--replay-buffer-capacity", type=int, default=1_000_000)
    p.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs_sac"),
        help="Base directory for logs/checkpoints.",
    )
    return p


def load_env_config(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Env config file not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    scope: dict = {}
    exec(path.read_text(), scope)
    return scope.get("config_env_base") or scope.get("config") or {}


def make_env_creator(env_config_base: dict):
    def _env_creator(env_config):
        cfg = dict(env_config_base)
        cfg.update(env_config or {})
        return PredPreyGrass(cfg)

    return _env_creator


def main() -> int:
    args = build_parser().parse_args()
    base_cfg = load_env_config(args.env_config_file)
    base_cfg["max_steps"] = args.max_env_steps

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    env_name = "PredPreyGrass-continuous"
    register_env(env_name, make_env_creator(base_cfg))

    sample_env = PredPreyGrass(dict(base_cfg))
    obs_space_pred = sample_env.observation_spaces["predator_0"]
    act_space_pred = sample_env.action_spaces["predator_0"]
    obs_space_prey = sample_env.observation_spaces["prey_0"]
    act_space_prey = sample_env.action_spaces["prey_0"]
    sample_env.close()

    policies = {
        "predator_policy": (None, obs_space_pred, act_space_pred, {}),
        "prey_policy": (None, obs_space_prey, act_space_prey, {}),
    }

    def policy_mapping_fn(agent_id: str, episode=None, **kwargs) -> str:
        return "predator_policy" if "predator" in agent_id else "prey_policy"

    config = (
        SACConfig()
        .environment(env=env_name, env_config={})
        .framework("torch")
        .env_runners(
            num_env_runners=args.num_workers,
            num_envs_per_env_runner=args.num_envs_per_worker,
            rollout_fragment_length=args.rollout_fragment_length,
            batch_mode="truncate_episodes",
            num_cpus_per_env_runner=1,
        )
        .training(
            gamma=args.gamma,
            tau=args.tau,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            train_batch_size=args.train_batch_size,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["predator_policy", "prey_policy"],
        )
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .debugging(log_level=args.log_level)
        .callbacks(EpisodeMetricsCallbacks)
    )

    config.learning_starts = args.learning_starts
    config.target_network_update_freq = args.target_network_update_freq
    config.min_sample_timesteps_per_iteration = (
        args.rollout_fragment_length * args.num_envs_per_worker * args.num_workers
    )
    config.min_time_s_per_iteration = 0
    config.min_train_timesteps_per_iteration = 0
    if args.entropy_coeff_auto:
        config.target_entropy = "auto"
    else:
        config.target_entropy = None
        config.initial_alpha = args.entropy_coeff
    config.replay_buffer_config.update(
        {
            "_enable_replay_buffer_api": True,
            "capacity": args.replay_buffer_capacity,
            "type": "MultiAgentReplayBuffer",
            "storage_unit": "timesteps",
            "replay_sequence_length": 1,
            "prioritized_replay": False,
        }
    )

    stop = {"training_iteration": args.num_iterations}
    checkpoint_cfg = air.CheckpointConfig(
        checkpoint_frequency=args.checkpoint_freq,
        checkpoint_at_end=True,
        num_to_keep=None if args.keep_all_checkpoints else 5,
    )
    storage_path = str(args.log_dir.expanduser().resolve())
    storage_uri = storage_path if storage_path.startswith("file://") else f"file://{storage_path}"
    run_cfg = air.RunConfig(
        name="SAC_PredPreyGrass",
        stop=stop,
        storage_path=storage_uri,
        checkpoint_config=checkpoint_cfg,
    )
    tuner = tune.Tuner(
        "SAC",
        run_config=run_cfg,
        param_space=config.to_dict(),
    )
    results = tuner.fit()
    ray.shutdown()
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
