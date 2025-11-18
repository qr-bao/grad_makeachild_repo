"""Optuna + ASHA hyper-parameter sweep for env127."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import CheckpointConfig, RunConfig, Tuner
from ray.tune.registry import register_env
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.stopper import MaximumIterationStopper, TrialPlateauStopper

from predpreygrass.rllib.env3.predpreygrass_rllib_env127.config.config_env_train import (
    config_env as default_env_config,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env127.config.config_ppo_cpu import (
    config_ppo as default_ppo_cpu,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env127.config.config_ppo_cpu_smoke import (
    config_ppo as config_ppo_smoke,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env127.config.config_ppo_gpu import (
    config_ppo as config_ppo_gpu,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env127.predpreygrass_rllib_env import (
    PredPreyGrass,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env127.train_simple import (
    create_model_config,
    policy_mapping_fn,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env127.utils import (
    build_multi_module_spec,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env127.utils.tuning_callbacks import (
    DropStopper,
    FinalMetricsLogger,
    PredatorScore,
    ReasonedStopper,
)


def get_config_ppo():
    if os.getenv("PPG_SMOKE"):
        return dict(config_ppo_smoke)
    if os.getenv("PPG_USE_GPU"):
        return dict(config_ppo_gpu)
    return dict(default_ppo_cpu)


def create_env_config():
    return dict(default_env_config)


def env_creator_factory(base_env_config):
    def env_creator(config):
        merged = base_env_config.copy()
        if config:
            merged.update(config)
        return PredPreyGrass(merged)

    return env_creator


def build_training_setup(env_config):
    env_creator = env_creator_factory(env_config)
    env_name = "PredPreyGrass-env127"
    register_env(env_name, env_creator)
    sample_env = env_creator({})
    obs_spaces = {}
    act_spaces = {}
    for agent_id, obs_space in sample_env.observation_spaces.items():
        pid = policy_mapping_fn(agent_id)
        if pid not in obs_spaces:
            obs_spaces[pid] = obs_space
            act_spaces[pid] = sample_env.action_spaces[agent_id]
    if hasattr(sample_env, "close"):
        sample_env.close()
    multi_module_spec = build_multi_module_spec(obs_spaces, act_spaces, create_model_config())
    policies = {pid: (None, obs_spaces[pid], act_spaces[pid], {}) for pid in obs_spaces}
    return env_name, env_creator, policies, multi_module_spec


def main():
    config_env = create_env_config()
    config_ppo = get_config_ppo()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ray_results_dir = (Path("logs") / "tune_optuna").resolve()
    experiment_name = f"PPO_ASHA_{timestamp}"
    experiment_path = ray_results_dir / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    with (experiment_path / "run_config.json").open("w", encoding="utf-8") as fp:
        json.dump({"config_env": config_env, "config_ppo": config_ppo}, fp, indent=2)

    env_name, env_creator, policies, multi_module_spec = build_training_setup(config_env)

    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    ppo_config = (
        PPOConfig()
        .environment(env=env_name, env_config=config_env)
        .framework("torch")
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .training(
            train_batch_size_per_learner=config_ppo["train_batch_size_per_learner"],
            minibatch_size=config_ppo["minibatch_size"],
            num_epochs=config_ppo["num_epochs"],
            gamma=config_ppo["gamma"],
            lr=config_ppo["lr"],
            lambda_=config_ppo["lambda_"],
            entropy_coeff=config_ppo["entropy_coeff"],
            vf_loss_coeff=config_ppo["vf_loss_coeff"],
            clip_param=config_ppo["clip_param"],
            kl_coeff=config_ppo["kl_coeff"],
            kl_target=config_ppo["kl_target"],
        )
        .rl_module(rl_module_spec=multi_module_spec)
        .learners(
            num_learners=config_ppo["num_learners"],
            num_gpus_per_learner=config_ppo["num_gpus_per_learner"],
            num_cpus_per_learner=config_ppo["num_cpus_per_learner"],
        )
        .env_runners(
            num_env_runners=config_ppo["num_env_runners"],
            num_envs_per_env_runner=config_ppo["num_envs_per_env_runner"],
            rollout_fragment_length=config_ppo["rollout_fragment_length"],
            sample_timeout_s=config_ppo["sample_timeout_s"],
            num_cpus_per_env_runner=config_ppo["num_cpus_per_env_runner"],
        )
        .resources(
            num_cpus_for_main_process=config_ppo["num_cpus_for_main_process"],
            placement_strategy="PACK",
        )
        .callbacks(PredatorScore)
    )

    max_iters = config_ppo["max_iters"]
    checkpoint_every = 10

    search_space = ppo_config.copy(copy_frozen=False)
    search_space.training(
        num_epochs=tune.qrandint(10, 40, q=2),
        lr=tune.loguniform(1e-4, 6e-4),
        entropy_coeff=tune.uniform(0.0, 0.02),
    )

    plateau = TrialPlateauStopper(
        metric="score_pred",
        mode="max",
        std=0.02,
        num_results=12,
        grace_period=40,
    )
    plateau.name = "plateau"
    drop = DropStopper(window=5, drop_threshold=0.2, name="drop")
    cap = MaximumIterationStopper(max_iters)
    cap.name = "max_iters"
    reasoned = ReasonedStopper(plateau, drop, cap)

    grace_period = 40
    reduction_factor = 3
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=max_iters,
        grace_period=grace_period,
        reduction_factor=reduction_factor,
    )
    searcher = OptunaSearch()

    cpus_total = os.cpu_count() or 1
    per_trial_cpus = config_ppo["num_cpus_for_main_process"] + config_ppo["num_cpus_per_env_runner"] * config_ppo["num_env_runners"]
    max_concurrent_trials = max(1, cpus_total // max(1, per_trial_cpus))

    tuner = Tuner(
        ppo_config.algo_class,
        param_space=search_space,
        run_config=RunConfig(
            name=experiment_name,
            storage_path=str(ray_results_dir),
            stop=reasoned,
            checkpoint_config=CheckpointConfig(
                num_to_keep=100,
                checkpoint_frequency=checkpoint_every,
                checkpoint_at_end=True,
            ),
            callbacks=[FinalMetricsLogger(reasoned, max_iters, grace_period, reduction_factor)],
        ),
        tune_config=tune.TuneConfig(
            metric="score_pred",
            mode="max",
            scheduler=scheduler,
            search_alg=searcher,
            num_samples=20,
            max_concurrent_trials=max_concurrent_trials,
        ),
    )

    result = tuner.fit()
    print("Best trial:", result.get_best_result(metric="score_pred", mode="max").metrics.get("score_pred"))
    ray.shutdown()


if __name__ == "__main__":
    main()
