"""
Shared training utilities for PredPreyGrass strategy variants.

This module mirrors the behaviour of ``train_simple.py`` but allows the predator
and prey populations to mix PPO and Random strategies independently.  Concrete
entrypoints import :func:`launch_strategy_training` with different defaults.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.tune import CheckpointConfig, RunConfig, Tuner
from ray.tune.registry import register_env

from predpreygrass.rllib.env3.predpreygrass_rllib_env124.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.env3.predpreygrass_rllib_env124 import train_simple as base_train
from predpreygrass.rllib.env3.predpreygrass_rllib_env124.random_baseline_trainable import (
    RandomBaselineTrainable,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env124.torch_random_rl_module import TorchRandomRLModule


StrategyLiteral = Literal["ppo", "random"]
STRATEGY_CHOICES: Sequence[StrategyLiteral] = ("ppo", "random")

create_env_config = base_train.create_env_config
create_model_config = base_train.create_model_config
setup_logger = base_train.setup_logger
policy_mapping_fn = base_train.policy_mapping_fn
LOG_DIR_DEFAULT = base_train.LOG_DIR_DEFAULT


def parse_args(default_predator_strategy: StrategyLiteral, default_prey_strategy: StrategyLiteral) -> argparse.Namespace:
    """Create an argument parser aligned with ``train_simple`` plus strategy flags."""
    parser = argparse.ArgumentParser(description="Train PredPreyGrass with configurable predator/prey strategies.")
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-envs-per-worker", type=int, default=3)
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(LOG_DIR_DEFAULT),
        help="Base directory for training and environment logs.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for the training script.",
    )
    parser.add_argument(
        "--env-debug-level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for the environment debug logs.",
    )
    parser.add_argument(
        "--disable-env-debug",
        action="store_true",
        help="Disable environment-level debug logging.",
    )
    parser.add_argument(
        "--predator-strategy",
        type=str,
        default=default_predator_strategy,
        choices=STRATEGY_CHOICES,
        help="Strategy applied to all predator agents.",
    )
    parser.add_argument(
        "--prey-strategy",
        type=str,
        default=default_prey_strategy,
        choices=STRATEGY_CHOICES,
        help="Strategy applied to all prey agents.",
    )
    return parser.parse_args()


def _module_spec_for_strategy(
    strategy: StrategyLiteral,
    obs_space,
    act_space,
    model_config: dict,
) -> RLModuleSpec:
    if strategy == "ppo":
        return RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            observation_space=obs_space,
            action_space=act_space,
            inference_only=False,
            model_config=model_config,
            catalog_class=None,
        )
    if strategy == "random":
        return RLModuleSpec(
            module_class=TorchRandomRLModule,
            observation_space=obs_space,
            action_space=act_space,
            inference_only=True,
            model_config=None,
            catalog_class=None,
        )
    raise ValueError(f"Unsupported strategy: {strategy}")


def _policies_to_train(predator_strategy: StrategyLiteral, prey_strategy: StrategyLiteral) -> list[str]:
    policies = []
    if predator_strategy == "ppo":
        policies.append("predator_policy")
    if prey_strategy == "ppo":
        policies.append("prey_policy")
    return policies


def run_training(
    args: argparse.Namespace,
    predator_strategy: StrategyLiteral,
    prey_strategy: StrategyLiteral,
    experiment_name: str | None = None,
) -> None:
    """Core training routine shared by the thin wrapper scripts."""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = Path(args.log_dir)
    run_log_dir = base_log_dir / f"run_{run_timestamp}"
    env_log_dir = run_log_dir / "envs"
    env_log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_log_dir, args.log_level)
    logger.info(
        "Run initialised | predator_strategy=%s | prey_strategy=%s | iterations=%s | workers=%s | envs_per_worker=%s",
        predator_strategy,
        prey_strategy,
        args.num_iterations,
        args.num_workers,
        args.num_envs_per_worker,
    )
    logger.info("Logs directory: %s", run_log_dir.resolve())
    print(f"Logs will be written to: {run_log_dir.resolve()}")
    if not args.disable_env_debug:
        print(f"Environment logs directory: {env_log_dir.resolve()}")

    base_env_config = create_env_config()
    base_env_config["debug_logging"] = not args.disable_env_debug
    base_env_config["debug_log_level"] = args.env_debug_level
    base_env_config["debug_log_dir"] = str(env_log_dir)

    logger.info(
        "Environment debug logging enabled=%s level=%s directory=%s",
        not args.disable_env_debug,
        args.env_debug_level,
        env_log_dir.resolve(),
    )

    def env_creator(config):
        merged_config = base_env_config.copy()
        if config:
            merged_config.update(config)
        return PredPreyGrass(merged_config)

    env_name = "PredPreyGrass-continuous"
    register_env(env_name, env_creator)
    logger.info("Environment registered: %s", env_name)

    ray.shutdown()
    logger.debug("Ray shutdown called before initialisation.")
    ray.init(log_to_driver=True, ignore_reinit_error=True)
    logger.info("Ray initialised.")
    try:
        cluster_resources = ray.cluster_resources()
        logger.debug("Cluster resources: %s", cluster_resources)
    except Exception:
        logger.debug("Unable to fetch cluster resources for logging.", exc_info=True)

    sample_env = env_creator({})
    logger.info("Sample environment created for space inspection.")
    obs_space_pred = sample_env.observation_spaces["predator_0"]
    act_space_pred = sample_env.action_spaces["predator_0"]
    obs_space_prey = sample_env.observation_spaces["prey_0"]
    act_space_prey = sample_env.action_spaces["prey_0"]

    print(f"\n{'='*80}")
    print("Training Mode: CONTINUOUS")
    print(f"{'='*80}")
    print(f"Observation Space (Predator): {obs_space_pred}")
    print(f"Action Space (Predator): {act_space_pred}")
    print(f"Observation Space (Prey): {obs_space_prey}")
    print(f"Action Space (Prey): {act_space_prey}")
    print(f"Predator Strategy: {predator_strategy.upper()} | Prey Strategy: {prey_strategy.upper()}")
    print(f"{'='*80}\n")

    logger.info("Observation Space (Predator): %s", obs_space_pred)
    logger.info("Action Space (Predator): %s", act_space_pred)
    logger.info("Observation Space (Prey): %s", obs_space_prey)
    logger.info("Action Space (Prey): %s", act_space_prey)

    sample_env_log_path = getattr(sample_env, "debug_log_path", None)
    if sample_env_log_path and not args.disable_env_debug:
        logger.info("Sample environment log file: %s", sample_env_log_path)
        print(f"Sample environment debug log file: {sample_env_log_path}")
    if hasattr(sample_env, "close"):
        sample_env.close()

    model_config = create_model_config()
    multi_module_spec = MultiRLModuleSpec(
        rl_module_specs={
            "predator_policy": _module_spec_for_strategy(
                predator_strategy,
                obs_space_pred,
                act_space_pred,
                model_config,
            ),
            "prey_policy": _module_spec_for_strategy(
                prey_strategy,
                obs_space_prey,
                act_space_prey,
                model_config,
            ),
        }
    )

    policies_to_train = _policies_to_train(predator_strategy, prey_strategy)
    multi_agent_kwargs = {
        "policies": {
            "predator_policy": (None, obs_space_pred, act_space_pred, {}),
            "prey_policy": (None, obs_space_prey, act_space_prey, {}),
        },
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": policies_to_train,
    }

    logger.info("Environment logs directory: %s", env_log_dir.resolve())

    random_only_mode = not policies_to_train
    trainable_for_resume = None

    if random_only_mode:
        logger.info(
            "All policies set to Random; running rollouts through Ray (TensorBoard logging, no parameter updates)."
        )
        print(f"\n{'='*80}")
        print("Random-only mode: rollouts will be logged via Ray/TensorBoard")
        print("Policies remain inference-only; no gradients will be computed.")
        print(f"{'='*80}\n")

        exp_name = experiment_name or f"RandomBaseline_{predator_strategy}_{prey_strategy}"
        episodes_per_iter = max(1, args.num_workers * args.num_envs_per_worker)
        logger.info(
            "Starting random baseline for %s iterations (episodes_per_iter=%s, checkpoint_freq=%s)",
            args.num_iterations,
            episodes_per_iter,
            args.checkpoint_freq,
        )
        print(
            f"\nStarting random baseline for {args.num_iterations} iterations "
            f"(episodes_per_iter={episodes_per_iter})..."
        )
        param_space = {
            "env_config": base_env_config,
            "episodes_per_iteration": episodes_per_iter,
            "seed": None,
        }
        run_config = RunConfig(
            name=exp_name,
            stop={"training_iteration": args.num_iterations},
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_frequency=args.checkpoint_freq,
                checkpoint_at_end=False,
            ),
        )
        tuner = Tuner(RandomBaselineTrainable, param_space=param_space, run_config=run_config)
        trainable_for_resume = RandomBaselineTrainable
    else:
        ppo_config = (
            PPOConfig()
            .environment(env=env_name, env_config=base_env_config)
            .framework("torch")
            .multi_agent(**multi_agent_kwargs)
            .learners(num_gpus_per_learner=1, num_learners=1)
            .training(
                train_batch_size_per_learner=2048,
                minibatch_size=256,
                num_epochs=30,
                gamma=0.99,
                lr=0.0003,
                entropy_coeff=0.01,
                vf_loss_coeff=1.0,
                clip_param=0.3,
                kl_coeff=0.2,
                kl_target=0.01,
            )
            .rl_module(rl_module_spec=multi_module_spec)
            .env_runners(
                num_env_runners=args.num_workers,
                num_envs_per_env_runner=args.num_envs_per_worker,
                num_cpus_per_env_runner=3,
                rollout_fragment_length=128,
                sample_timeout_s=600,
            )
            .resources(num_cpus_for_main_process=4)
        )

        exp_name = experiment_name or f"PPO_PredPreyGrass_{predator_strategy}_{prey_strategy}"
        logger.info(
            "Starting training for %s iterations (checkpoint_freq=%s, resume=%s) | exp=%s",
            args.num_iterations,
            args.checkpoint_freq,
            args.resume,
            exp_name,
        )
        logger.info("Environment logs directory: %s", env_log_dir.resolve())
        print(f"\nStarting training for {args.num_iterations} iterations...")

        run_config = RunConfig(
            name=exp_name,
            stop={"training_iteration": args.num_iterations},
            checkpoint_config=CheckpointConfig(
                num_to_keep=5,
                checkpoint_frequency=args.checkpoint_freq,
                checkpoint_at_end=True,
            ),
        )
        tuner = Tuner(ppo_config.algo_class, param_space=ppo_config, run_config=run_config)
        trainable_for_resume = ppo_config.algo_class

    try:
        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
            logger.info("Resuming from checkpoint: %s", args.resume)
            tuner = Tuner.restore(args.resume, trainable_for_resume)

        results = tuner.fit()
        logger.info("Tuner finished execution.")

        best_result = results.get_best_result()
        print(f"\n{'='*80}")
        print("Training Completed!")
        print(f"Best checkpoint: {best_result.checkpoint}")
        print(f"{'='*80}\n")
        logger.info("Training completed. Best checkpoint: %s", best_result.checkpoint)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        logger.warning("Training interrupted by user.")
    except Exception as exc:
        logger.exception("Training failed with exception: %s", exc)
        raise
    finally:
        logger.info("Shutting down Ray.")
        ray.shutdown()


def launch_strategy_training(
    default_predator_strategy: StrategyLiteral,
    default_prey_strategy: StrategyLiteral,
    experiment_name: str | None = None,
) -> None:
    """Parse CLI arguments and execute the shared training routine."""
    args = parse_args(default_predator_strategy, default_prey_strategy)
    run_training(
        args=args,
        predator_strategy=args.predator_strategy,
        prey_strategy=args.prey_strategy,
        experiment_name=experiment_name,
    )
