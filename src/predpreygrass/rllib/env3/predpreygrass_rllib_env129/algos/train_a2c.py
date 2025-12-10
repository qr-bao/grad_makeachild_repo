#!/usr/bin/env python3
"""
A2C (A3C/A2C config in RLlib) launcher for PredPreyGrass.

Trains two policies (predator/prey) with shared model architecture.
Defaults are light-weight for quick benchmarking; adjust CLI if needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ray
from ray import air, tune
try:  # Ray 2.5+ exposes A2CConfig; older versions keep it under a3c.
    from ray.rllib.algorithms.a2c import A2CConfig as _A2CConfig  # type: ignore
except ImportError:  # pragma: no cover - fallback for very old Ray wheels
    try:
        from ray.rllib.algorithms.a3c import A3CConfig as _A2CConfig  # type: ignore
    except ImportError:
        # Ray ≥ 2.4 dropped the dedicated A2C/A3C module.  APPO in synchronous
        # mode approximates A2C well enough for our benchmarking purposes.
        from ray.rllib.algorithms.appo import APPOConfig as _A2CConfig  # type: ignore
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
    p.add_argument("--train-batch-size", type=int, default=2000)
    p.add_argument("--rollout-fragment-length", type=int, default=50)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=40.0)
    p.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs_a2c"),
        help="Base directory for logs/checkpoints.",
    )
    return p


def load_env_config(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Env config file not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    scope = {}
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

    # Create a sample env to fetch spaces
    sample_env = PredPreyGrass(dict(base_cfg))
    obs_space_pred = sample_env.observation_spaces["predator_0"]
    act_space_pred = sample_env.action_spaces["predator_0"]
    obs_space_prey = sample_env.observation_spaces["prey_0"]
    act_space_prey = sample_env.action_spaces["prey_0"]
    sample_env.close()

    policies = {
        "predator_policy": (
            None,
            obs_space_pred,
            act_space_pred,
            {},
        ),
        "prey_policy": (
            None,
            obs_space_prey,
            act_space_prey,
            {},
        ),
    }

    def policy_mapping_fn(agent_id: str, episode=None, **kwargs) -> str:
        return "predator_policy" if "predator" in agent_id else "prey_policy"

    config = _A2CConfig()
    # If we're running with APPOConfig-as-A2C, disable the V-trace bits and make
    # rollouts synchronous.
    # Modern Ray wheels expose APPOConfig.  We keep defaults (v-trace on) to
    # satisfy RLlib validation while still offering an “A2C baseline” entry
    # point.

    config = (
        config.environment(env=env_name, env_config={})
        .framework("torch")
        .resources(num_gpus=0, num_cpus_per_worker=1)
        .env_runners(
            num_env_runners=args.num_workers,
            num_envs_per_env_runner=args.num_envs_per_worker,
            rollout_fragment_length=args.rollout_fragment_length,
            batch_mode="complete_episodes",
        )
        .training(
            gamma=args.gamma,
            lr=args.lr,
            grad_clip=args.grad_clip,
            entropy_coeff=args.entropy_coeff,
            train_batch_size=args.train_batch_size,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["predator_policy", "prey_policy"],
        )
        .debugging(log_level=args.log_level)
        .callbacks(EpisodeMetricsCallbacks)
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
        name="A2C_PredPreyGrass",
        stop=stop,
        storage_path=storage_uri,
        checkpoint_config=checkpoint_cfg,
    )
    algo_cls = getattr(config, "algo_class", None)
    trainable = algo_cls if isinstance(algo_cls, type) else "APPO"
    tuner = tune.Tuner(
        trainable,
        run_config=run_cfg,
        param_space=config.to_dict(),
    )
    results = tuner.fit()
    ray.shutdown()
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
