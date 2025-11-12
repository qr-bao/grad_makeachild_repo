"""
简化版训练脚本：PredPreyGrass v2
专门针对 Ray RLlib 2.50.0 优化
移除了复杂的回调函数，提高稳定性

使用方法：
    python train_simple.py
"""

from predpreygrass.rllib.env3.predpreygrass_rllib_env124.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.env3.predpreygrass_rllib_env124.config_env import config_env
from predpreygrass.rllib.env3.predpreygrass_rllib_env124.prey_test_config import prey_test_config
import argparse
import logging
from datetime import datetime
from pathlib import Path
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.tune.registry import register_env
from ray.tune import Tuner, RunConfig, CheckpointConfig


def create_env_config():
    """创建连续空间环境配置（直接使用 prey_test_config）"""
    return prey_test_config.copy()


def create_model_config():
    """连续空间模型配置"""
    return {
        "fcnet_hiddens": [256, 256, 128],
        "fcnet_activation": "relu",
        "vf_share_layers": False,
    }


LOG_DIR_DEFAULT = Path("logs")


def setup_logger(log_dir: Path, log_level: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_simple")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_dir / "train_simple.log")
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def policy_mapping_fn(agent_id, *args, **kwargs):
    """策略映射函数"""
    if "predator" in agent_id:
        return "predator_policy"
    elif "prey" in agent_id:
        return "prey_policy"
    return None


def main():
    parser = argparse.ArgumentParser(description="Train PredPreyGrass Environment (Continuous-only)")
    parser.add_argument("--num-iterations", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-envs-per-worker", type=int, default=3)
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=str(LOG_DIR_DEFAULT), help="Base directory for training and environment logs.")
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
        "--predator-survival-bonus",
        type=float,
        default=None,
        help="Bonus reward granted to each predator that survives until the episode ends.",
    )
    parser.add_argument(
        "--prey-survival-bonus",
        type=float,
        default=None,
        help="Bonus reward granted to each prey that survives until the episode ends.",
    )
    
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = Path(args.log_dir)
    run_log_dir = base_log_dir / f"run_{run_timestamp}"
    env_log_dir = run_log_dir / "envs"
    env_log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_log_dir, args.log_level)
    logger.info(
        "Run initialised | mode=continuous | iterations=%s | workers=%s | envs_per_worker=%s",
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
    if args.predator_survival_bonus is not None:
        base_env_config["survival_bonus_predator"] = args.predator_survival_bonus
    if args.prey_survival_bonus is not None:
        base_env_config["survival_bonus_prey"] = args.prey_survival_bonus

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
    
    # 注册环境
    env_name = "PredPreyGrass-continuous"
    register_env(env_name, env_creator)
    logger.info("Environment registered: %s", env_name)
    
    # 初始化Ray
    ray.shutdown()
    logger.debug("Ray shutdown called before initialisation.")
    ray.init(log_to_driver=True, ignore_reinit_error=True)
    logger.info("Ray initialised.")
    try:
        cluster_resources = ray.cluster_resources()
        logger.debug("Cluster resources: %s", cluster_resources)
    except Exception:
        logger.debug("Unable to fetch cluster resources for logging.", exc_info=True)
    
    # 创建样本环境
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
    
    # 创建模型配置
    model_config = create_model_config()
    
    # 构建MultiRLModuleSpec
    multi_module_spec = MultiRLModuleSpec(
        rl_module_specs={
            "predator_policy": RLModuleSpec(
                module_class=DefaultPPOTorchRLModule,
                observation_space=obs_space_pred,
                action_space=act_space_pred,
                inference_only=False,
                model_config=model_config,
                catalog_class=None,
            ),
            "prey_policy": RLModuleSpec(
                module_class=DefaultPPOTorchRLModule,
                observation_space=obs_space_prey,
                action_space=act_space_prey,
                inference_only=False,
                model_config=model_config,
                catalog_class=None,
            ),
        }
    )
    
    # 配置PPO
    ppo_config = (
        PPOConfig()
        .environment(env=env_name, env_config=base_env_config)
        .framework("torch")
        .multi_agent(
            policies={
                "predator_policy": (None, obs_space_pred, act_space_pred, {}),
                "prey_policy": (None, obs_space_prey, act_space_prey, {}),
            },
            policy_mapping_fn=policy_mapping_fn,
        )
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
            rollout_fragment_length="auto",
            sample_timeout_s=600,
        )
        .resources(num_cpus_for_main_process=4)
    )
    
    # 配置Tuner
    logger.info(
        "Starting training for %s iterations (checkpoint_freq=%s, resume=%s)",
        args.num_iterations,
        args.checkpoint_freq,
        args.resume,
    )
    logger.info("Environment logs directory: %s", env_log_dir.resolve())
    print(f"\nStarting training for {args.num_iterations} iterations...")
    
    tuner = Tuner(
        ppo_config.algo_class,
        param_space=ppo_config,
        run_config=RunConfig(
            name="PPO_PredPreyGrass_continuous_simple",
            stop={"training_iteration": args.num_iterations},
            checkpoint_config=CheckpointConfig(
                num_to_keep=5,
                checkpoint_frequency=args.checkpoint_freq,
                checkpoint_at_end=True,
            ),
        ),
    )

    # 运行训练
    try:
        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
            logger.info("Resuming from checkpoint: %s", args.resume)
            tuner = Tuner.restore(args.resume, ppo_config.algo_class)
        
        results = tuner.fit()
        logger.info("Tuner finished execution.")
        
        best_result = results.get_best_result()
        print(f"\n{'='*80}")
        print("Training Completed!")
        print(f"Best checkpoint: {best_result.checkpoint}")
        print(f"{'='*80}\n")
        logger.info("Training completed. Best checkpoint: %s", best_result.checkpoint)
        # if best_result.log_dir:
        #     logger.info("Best result log directory: %s", best_result.log_dir)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        logger.warning("Training interrupted by user.")
    except Exception as exc:
        logger.exception("Training failed with exception: %s", exc)
        raise
    finally:
        logger.info("Shutting down Ray.")
        ray.shutdown()


if __name__ == "__main__":
    main()
