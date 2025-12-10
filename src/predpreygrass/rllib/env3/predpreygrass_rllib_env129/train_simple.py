"""
简化版训练脚本：PredPreyGrass v2
专门针对 Ray RLlib 2.50.0 优化
移除了复杂的回调函数，提高稳定性

使用方法：
    python train_simple.py
"""

import argparse
import json
import logging
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("RLLIB_ENABLE_NEW_API_STACK", "0")

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.examples._old_api_stack.policy.random_policy import RandomPolicy
from ray.tune import CheckpointConfig, RunConfig, Tuner
from ray.tune.registry import register_env
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper, Stopper
from ray.rllib.policy.sample_batch import SampleBatch

from predpreygrass.rllib.env3.predpreygrass_rllib_env129.predpreygrass_rllib_env import (
    PredPreyGrass,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.config.config_env_train import (
    config_env as config_env_train,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.metrics_callbacks import (
    EpisodeMetricsCallbacks,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.random_baseline_trainable import (
    RandomBaselineTrainable,
)


def _patch_sample_batch():
    original_fn = SampleBatch.is_single_trajectory

    def _is_single_trajectory(self):
        terminateds = list(self[SampleBatch.TERMINATEDS])
        truncations = (
            list(self[SampleBatch.TRUNCATEDS])
            if SampleBatch.TRUNCATEDS in self
            else [False] * len(terminateds)
        )
        if not terminateds:
            return True
        if truncations and truncations[-1] and not terminateds[-1]:
            terminateds = terminateds[:-1]
            truncations = truncations[:-1]
        if not terminateds:
            return True
        return not any(terminateds[:-1]) and not any(truncations[:-1])

    if getattr(SampleBatch.is_single_trajectory, "__patched", False) is False:
        _is_single_trajectory.__patched = True
        SampleBatch.is_single_trajectory = _is_single_trajectory


_patch_sample_batch()

class RewardPlateauStopper(Stopper):
    """Stop training if monitored metric fails to improve."""

    def __init__(self, metric: str, patience: int):
        self.metric = metric
        self.patience = patience
        self._best = None
        self._bad_iters = 0

    def __call__(self, trial_id, result):
        if self.patience <= 0:
            return False
        value = result.get(self.metric)
        if value is None:
            return False
        if self._best is None or value > self._best:
            self._best = value
            self._bad_iters = 0
        else:
            self._bad_iters += 1
        return self._bad_iters >= self.patience

    def stop_all(self):
        return False


def create_env_config():
    """创建连续空间环境配置（基于 config_env_train）。"""
    return config_env_train.copy()


def create_model_config():
    """连续空间模型配置（可在 main 中追加 LSTM 设置）"""
    return {
        "fcnet_hiddens": [256, 256, 128],
        "fcnet_activation": "relu",
        "vf_share_layers": False,
    }


LOG_DIR_DEFAULT = Path("logs")
_POLICY_MAPPING_STATE = {
    "pred_default": "predator_policy",
    "prey_default": "prey_policy",
    "pred_n_pops": 1,
    "prey_n_pops": 1,
    "population_policy_map": {},
    "use_population_suffix": False,
}


def _serialise(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    return obj


def _write_run_metadata(run_log_dir: Path, metadata: dict) -> None:
    run_log_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_log_dir / "run_metadata.json"
    txt_path = run_log_dir / "run_metadata.txt"
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)

    summary_lines = [
        f"Run ID: {metadata.get('run_id')}",
        f"Status: {metadata.get('status')}",
        f"Command: {metadata.get('command')}",
        f"Run Directory: {metadata.get('directories', {}).get('run_log_dir')}",
        f"Environment Logs: {metadata.get('directories', {}).get('env_log_dir')}",
        f"Ray/TensorBoard Logs: {metadata.get('directories', {}).get('ray_results_dir')}",
        f"TensorBoard Command: {metadata.get('tensorboard_command')}",
        f"Best Checkpoint: {metadata.get('best_checkpoint') or 'N/A'}",
    ]
    env_cfg = metadata.get("env_config") or {}
    if env_cfg:
        summary_lines.append("")
        summary_lines.append("Environment Config:")
        for key in sorted(env_cfg):
            summary_lines.append(f"  {key} = {env_cfg[key]}")
    txt_path.write_text("\n".join(summary_lines), encoding="utf-8")


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


def configure_population_policy_mapping(
    n_predator_pops: int = 1,
    n_prey_pops: int = 1,
    *,
    use_population_suffix: bool = False,
    population_policy_map: dict | None = None,
    predator_default: str = "predator_policy",
    prey_default: str = "prey_policy",
) -> None:
    """Configure how agent populations are mapped to policies.

    Args:
        n_predator_pops: Number of predator populations in the env.
        n_prey_pops: Number of prey populations in the env.
        use_population_suffix: If True, auto-generate policy ids as
            ``{species}_pop{pop_id}_policy`` when no explicit mapping exists.
        population_policy_map: Optional explicit map {(species, pop_id): policy_id}.
        predator_default: Fallback policy id for predators.
        prey_default: Fallback policy id for prey.
    """
    _POLICY_MAPPING_STATE["pred_n_pops"] = max(1, int(n_predator_pops))
    _POLICY_MAPPING_STATE["prey_n_pops"] = max(1, int(n_prey_pops))
    _POLICY_MAPPING_STATE["use_population_suffix"] = bool(use_population_suffix)
    _POLICY_MAPPING_STATE["pred_default"] = predator_default
    _POLICY_MAPPING_STATE["prey_default"] = prey_default
    if population_policy_map is None:
        _POLICY_MAPPING_STATE["population_policy_map"] = {}
    else:
        _POLICY_MAPPING_STATE["population_policy_map"] = dict(population_policy_map)


def _get_last_info_for_agent(episode, agent_id: str) -> dict:
    """Best-effort fetch of the latest info payload for an agent."""
    if episode is None:
        return {}
    for getter in ("last_info_for",):
        try:
            fn = getattr(episode, getter, None)
            if fn:
                info = fn(agent_id)
                if info:
                    return info
        except Exception:
            pass
    try:
        info = episode._agent_to_last_info.get(agent_id)  # type: ignore[attr-defined]
        if info:
            return info
    except Exception:
        pass
    return {}


def policy_mapping_fn(agent_id, episode=None, *args, **kwargs):
    """Population-aware policy mapping (falls back to species defaults)."""
    info = _get_last_info_for_agent(episode, agent_id)
    species = info.get("species")
    if species is None:
        if "predator" in agent_id:
            species = "predator"
        elif "prey" in agent_id:
            species = "prey"
    if species is None:
        return None

    pop_id = info.get("population_id")
    try:
        pop_id = int(pop_id) if pop_id is not None else None
    except Exception:
        pop_id = None

    default_policy = (
        _POLICY_MAPPING_STATE["pred_default"] if species == "predator" else _POLICY_MAPPING_STATE["prey_default"]
    )

    if pop_id is not None:
        policy_map = _POLICY_MAPPING_STATE.get("population_policy_map", {})
        key = (species, pop_id)
        if key in policy_map:
            return policy_map[key]

        if _POLICY_MAPPING_STATE.get("use_population_suffix"):
            if species == "predator":
                pop_id = min(pop_id, _POLICY_MAPPING_STATE["pred_n_pops"] - 1)
            else:
                pop_id = min(pop_id, _POLICY_MAPPING_STATE["prey_n_pops"] - 1)
            return f"{species}_pop{pop_id}_policy"

    return default_policy


def build_population_policies(
    *,
    n_predator_pops: int,
    n_prey_pops: int,
    predator_strategies: list[str],
    prey_strategies: list[str],
    obs_space_pred,
    act_space_pred,
    obs_space_prey,
    act_space_prey,
    model_config: dict,
) -> tuple[dict, list[str]]:
    """Create per-population policy specs and configure mapping."""
    use_population_suffix = (n_predator_pops > 1) or (n_prey_pops > 1)

    def _policy_id(species: str, pop_id: int) -> str:
        if use_population_suffix:
            return f"{species}_pop{pop_id}_policy"
        return f"{species}_policy"

    configure_population_policy_mapping(
        n_predator_pops=n_predator_pops,
        n_prey_pops=n_prey_pops,
        use_population_suffix=use_population_suffix,
        predator_default=_policy_id("predator", 0),
        prey_default=_policy_id("prey", 0),
    )

    policies: dict = {}
    policies_to_train: list[str] = []

    for pop_id in range(n_predator_pops):
        strategy = predator_strategies[pop_id] if pop_id < len(predator_strategies) else predator_strategies[-1]
        pid = _policy_id("predator", pop_id)
        if strategy in ("ppo", "sac"):
            policies[pid] = (None, obs_space_pred, act_space_pred, {"model": model_config.copy()})
            if pid not in policies_to_train:
                policies_to_train.append(pid)
        else:
            policies[pid] = (RandomPolicy, obs_space_pred, act_space_pred, {})

    for pop_id in range(n_prey_pops):
        strategy = prey_strategies[pop_id] if pop_id < len(prey_strategies) else prey_strategies[-1]
        pid = _policy_id("prey", pop_id)
        if strategy in ("ppo", "sac"):
            policies[pid] = (None, obs_space_prey, act_space_prey, {"model": model_config.copy()})
            if pid not in policies_to_train:
                policies_to_train.append(pid)
        else:
            policies[pid] = (RandomPolicy, obs_space_prey, act_space_prey, {})

    return policies, policies_to_train


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
        "--max-env-steps",
        type=int,
        default=500,
        help="Cap environment max_steps to avoid runaway episodes (<=0 disables the cap).",
    )
    parser.add_argument(
        "--keep-all-checkpoints",
        action="store_true",
        help="Disable checkpoint pruning so every checkpoint is retained.",
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
    parser.add_argument(
        "--env-config-file",
        type=str,
        default="env_config_enable_repro.json",
        help="Optional JSON file with env config overrides (merged onto config_env_train).",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5,
        help="Run evaluation every N training iterations (set <=0 to disable).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of episodes per evaluation run.",
    )
    parser.add_argument(
        "--eval-num-workers",
        type=int,
        default=1,
        help="Number of parallel workers used for evaluation.",
    )
    parser.add_argument(
        "--rollout-fragment-length",
        type=str,
        default="auto",
        help="rollout_fragment_length passed to PPOConfig.env_runners(). "
        "Supports positive integers or 'auto'. Default=auto (train_batch_size divided by total envs).",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=396,
        help="train_batch_size_per_learner override (default: 2048).",
    )
    parser.add_argument(
        "--ppo-minibatch-size",
        type=int,
        default=256,
        help="Minibatch size used by PPO SGD (smaller values improve stability).",
    )
    parser.add_argument(
        "--ppo-num-epochs",
        type=int,
        default=10,
        help="Number of SGD passes over each train batch (lower to curb gradient blowups).",
    )
    parser.add_argument(
        "--use-lstm",
        action="store_true",
        help="Enable LSTM wrapper for policies (shared config for predator/prey).",
    )
    parser.add_argument(
        "--lstm-cell-size",
        type=int,
        default=256,
        help="LSTM hidden size when --use-lstm is enabled.",
    )
    parser.add_argument(
        "--lstm-seq-len",
        type=int,
        default=32,
        help="max_seq_len for LSTM when --use-lstm is enabled.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Clip gradients by global norm to this value (<=0 disables clipping).",
    )
    parser.add_argument(
        "--debug-logging",
        action="store_true",
        help="Shortcut to force script/env logging to DEBUG.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=50,
        help="Stop if reward fails to improve for N iterations (0 disables early stop).",
    )
    parser.add_argument(
        "--early-stop-metric",
        type=str,
        default="env_runners/episode_reward_mean",
        help="Metric key to monitor for early stopping.",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="[[0, 0.0003], [500, 0.0001]]",
        help="Optional JSON list for lr_schedule, e.g. '[[0,3e-4],[500,1e-4]]'.",
    )
    parser.add_argument(
        "--predator-strategy",
        type=str,
        choices=["ppo", "random", "sac"],
        default="ppo",
        help="Learning strategy for predator policy.",
    )
    parser.add_argument(
        "--predator-strategies",
        type=str,
        default=None,
        help="Optional comma-separated per-population strategies for predators (overrides --predator-strategy).",
    )
    parser.add_argument(
        "--prey-strategy",
        type=str,
        choices=["ppo", "random", "sac"],
        default="ppo",
        help="Learning strategy for prey policy.",
    )
    parser.add_argument(
        "--prey-strategies",
        type=str,
        default=None,
        help="Optional comma-separated per-population strategies for prey (overrides --prey-strategy).",
    )
    
    args = parser.parse_args()
    if args.debug_logging:
        args.log_level = "DEBUG"
        args.env_debug_level = "DEBUG"
    lr_schedule = None
    if args.lr_schedule:
        try:
            lr_schedule = json.loads(args.lr_schedule)
        except json.JSONDecodeError as exc:
            raise ValueError("--lr-schedule must be valid JSON list, e.g. '[[0, 3e-4],[500,1e-4]]'") from exc
        if not isinstance(lr_schedule, list):
            raise ValueError("--lr-schedule must decode to a list of [t, lr] pairs.")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = Path(args.log_dir)
    run_log_dir = base_log_dir / f"run_{run_timestamp}"
    env_log_dir = run_log_dir / "envs"
    ray_results_dir = run_log_dir / "ray_results"
    env_log_dir.mkdir(parents=True, exist_ok=True)
    ray_results_dir.mkdir(parents=True, exist_ok=True)

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
    if args.env_config_file:
        file_path = Path(args.env_config_file).expanduser().resolve()
        if file_path.is_file():
            with file_path.open("r", encoding="utf-8") as fp:
                override_cfg = json.load(fp)
            base_env_config.update(override_cfg)
            logger.info("Loaded env config overrides from %s", file_path)
        else:
            logger.warning("env_config_file %s not found; ignoring override.", file_path)

    logger.info(
        "Environment debug logging enabled=%s level=%s directory=%s",
        not args.disable_env_debug,
        args.env_debug_level,
        env_log_dir.resolve(),
    )

    if args.max_env_steps and args.max_env_steps > 0:
        current_max_steps = base_env_config.get("max_steps")
        capped_max_steps = (
            min(current_max_steps, args.max_env_steps)
            if current_max_steps is not None
            else args.max_env_steps
        )
        if current_max_steps != capped_max_steps:
            logger.info(
                "Capping env max_steps from %s to %s to keep episodes bounded.",
                current_max_steps,
                capped_max_steps,
            )
        base_env_config["max_steps"] = capped_max_steps

    run_metadata = {
        "run_id": run_timestamp,
        "script": str(Path(__file__).resolve()),
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "started_at": run_timestamp,
        "args": _serialise(vars(args)),
        "env_config": _serialise(base_env_config),
        "env_config_file": str(Path(args.env_config_file).resolve())
        if args.env_config_file
        else None,
        "directories": {
            "run_log_dir": str(run_log_dir.resolve()),
            "env_log_dir": str(env_log_dir.resolve()),
            "ray_results_dir": str(ray_results_dir.resolve()),
        },
        "tensorboard_command": f"tensorboard --logdir {ray_results_dir.resolve()}",
        "status": "initialised",
        "best_checkpoint": None,
    }
    run_metadata["keep_all_checkpoints"] = args.keep_all_checkpoints
    _write_run_metadata(run_log_dir, run_metadata)
    print(f"TensorBoard command: {run_metadata['tensorboard_command']}")

    def env_creator(config):
        merged_config = base_env_config.copy()
        if config:
            merged_config.update(config)
        return PredPreyGrass(merged_config)
    
    # 注册环境
    env_name = "PredPreyGrass-continuous"
    register_env(env_name, env_creator)
    logger.info("Environment registered: %s", env_name)
    if args.keep_all_checkpoints:
        logger.info("Checkpoint retention configured to keep all checkpoints (num_to_keep=None).")
    else:
        logger.info(
            "Checkpoint retention will keep a limited window (use --keep-all-checkpoints to disable pruning)."
        )
    
    # 初始化Ray
    ray.shutdown()
    logger.debug("Ray shutdown called before initialisation.")
    runtime_env_vars = {
        "RLLIB_ENABLE_NEW_API_STACK": os.environ.get(
            "RLLIB_ENABLE_NEW_API_STACK", "0"
        )
    }
    debug_dump_dir = os.environ.get("PREDPREY_BAD_BATCH_DIR")
    if debug_dump_dir:
        runtime_env_vars["PREDPREY_BAD_BATCH_DIR"] = debug_dump_dir
    ray.init(
        log_to_driver=True,
        ignore_reinit_error=True,
        runtime_env={"env_vars": runtime_env_vars},
    )
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
    if args.use_lstm:
        model_config.update(
            {
                "use_lstm": True,
                "max_seq_len": args.lstm_seq_len,
                "lstm_cell_size": args.lstm_cell_size,
                "lstm_use_prev_action": False,
                "lstm_use_prev_reward": False,
            }
        )
        logger.info(
            "LSTM enabled for policies | cell_size=%s | max_seq_len=%s",
            args.lstm_cell_size,
            args.lstm_seq_len,
        )
    
    # 配置PPO
    total_envs = max(1, args.num_workers * args.num_envs_per_worker)
    auto_fragment = False
    rollout_fragment_length = args.rollout_fragment_length
    if isinstance(rollout_fragment_length, str):
        rollout_fragment_length = rollout_fragment_length.strip()
        if rollout_fragment_length.lower() == "auto":
            rollout_fragment_length = max(1, args.train_batch_size // total_envs)
            auto_fragment = True
        else:
            try:
                rollout_fragment_length = int(rollout_fragment_length)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid --rollout-fragment-length value: {args.rollout_fragment_length}"
                ) from exc
    if isinstance(rollout_fragment_length, int) and rollout_fragment_length <= 0:
        raise ValueError("--rollout-fragment-length must be a positive integer.")
    if auto_fragment:
        logger.info(
            "Auto rollout_fragment_length resolved to %s based on train_batch_size=%s, workers=%s, envs_per_worker=%s",
            rollout_fragment_length,
            args.train_batch_size,
            args.num_workers,
            args.num_envs_per_worker,
        )
    if isinstance(rollout_fragment_length, int):
        min_train_batch = total_envs * rollout_fragment_length
        if args.train_batch_size < min_train_batch:
            logger.warning(
                "train_batch_size=%s is too small for rollout_fragment_length=%s with total_envs=%s; "
                "increasing train_batch_size to %s to satisfy RLlib sampling requirements.",
                args.train_batch_size,
                rollout_fragment_length,
                total_envs,
                min_train_batch,
            )
            args.train_batch_size = min_train_batch
    if args.train_batch_size <= 0:
        raise ValueError("--train-batch-size must be positive.")
    if args.ppo_minibatch_size <= 0:
        raise ValueError("--ppo-minibatch-size must be positive.")
    if args.ppo_num_epochs <= 0:
        raise ValueError("--ppo-num-epochs must be positive.")

    training_kwargs = dict(
        train_batch_size=args.train_batch_size,
        train_batch_size_per_learner=args.train_batch_size,
        minibatch_size=args.ppo_minibatch_size,
        num_epochs=args.ppo_num_epochs,
        gamma=0.99,
        lr=0.0003,
        use_gae=True,
        lambda_=0.95,
        use_critic=True,
        entropy_coeff=0.01,
        vf_loss_coeff=1.0,
        clip_param=0.3,
        kl_coeff=0.2,
        kl_target=0.01,
        model=model_config,
    )
    if args.grad_clip and args.grad_clip > 0:
        training_kwargs["grad_clip"] = args.grad_clip
    else:
        logger.warning("Gradient clipping disabled (grad_clip=%s).", args.grad_clip)

    if lr_schedule:
        training_kwargs["lr_schedule"] = lr_schedule

    n_populations = getattr(sample_env, "n_populations", base_env_config.get("n_populations", 1))

    def _parse_strategy_list(raw: str | None, fallback: str, n: int) -> list[str]:
        allowed = ("ppo", "random", "sac")
        if not raw:
            return [fallback] * n
        values = [v.strip().lower() for v in raw.split(",") if v.strip()]
        if not values:
            return [fallback] * n
        if any(v not in allowed for v in values):
            invalid = [v for v in values if v not in allowed]
            raise ValueError(f"Invalid strategy in list: {invalid}; allowed: {allowed}")
        if len(values) < n:
            values.extend([values[-1]] * (n - len(values)))
        return values[:n]

    predator_strategies = _parse_strategy_list(args.predator_strategies, args.predator_strategy, n_populations)
    prey_strategies = _parse_strategy_list(args.prey_strategies, args.prey_strategy, n_populations)

    strategies_all = predator_strategies + prey_strategies
    has_ppo = any(s == "ppo" for s in strategies_all)
    has_sac = any(s == "sac" for s in strategies_all)
    if has_ppo and has_sac:
        raise ValueError("Mixing PPO and SAC in a single trainer is not supported; choose one family plus optional random.")
    algo_kind = "sac" if has_sac else "ppo"

    policies, policies_to_train = build_population_policies(
        n_predator_pops=n_populations,
        n_prey_pops=n_populations,
        predator_strategies=predator_strategies,
        prey_strategies=prey_strategies,
        obs_space_pred=obs_space_pred,
        act_space_pred=act_space_pred,
        obs_space_prey=obs_space_prey,
        act_space_prey=act_space_prey,
        model_config=model_config,
    )

    random_only_mode = len(policies_to_train) == 0
    tuner = None
    trainable_cls = None

    run_metadata["training_mode"] = "random_baseline" if random_only_mode else "ppo"

    if random_only_mode:
        logger.info(
            "All policies configured as Random. Switching to RandomBaselineTrainable (no PPO updates)."
        )
        print(f"\n{'='*80}")
        print("Random-only mode detected.")
        print("Policies will act randomly; metrics still logged via Ray/TensorBoard.")
        print(f"{'='*80}\n")

        episodes_per_iter = max(1, args.num_workers * args.num_envs_per_worker)
        stopper = MaximumIterationStopper(args.num_iterations)
        checkpoint_num_to_keep = None if args.keep_all_checkpoints else 1
        tuner = Tuner(
            RandomBaselineTrainable,
            param_space={
                "env_config": base_env_config,
                "episodes_per_iteration": episodes_per_iter,
                "seed": base_env_config.get("seed"),
            },
            run_config=RunConfig(
                storage_path=str(ray_results_dir.resolve()),
                name="RandomBaseline_PredPreyGrass_continuous",
                stop=stopper,
                checkpoint_config=CheckpointConfig(
                    num_to_keep=checkpoint_num_to_keep,
                    checkpoint_frequency=args.checkpoint_freq,
                    checkpoint_at_end=False,
                ),
            ),
        )
        trainable_cls = RandomBaselineTrainable
    else:
        if algo_kind == "ppo":
            algo_config = (
                PPOConfig()
                .api_stack(
                    enable_rl_module_and_learner=False,
                    enable_env_runner_and_connector_v2=False,
                )
                .environment(env=env_name, env_config=base_env_config)
                .framework("torch")
                .multi_agent(
                    policies=policies,
                    policy_mapping_fn=policy_mapping_fn,
                    policies_to_train=policies_to_train,
                )
                .training(**training_kwargs)
                .learners(num_learners=1)
                .env_runners(
                    num_env_runners=args.num_workers,
                    num_envs_per_env_runner=args.num_envs_per_worker,
                    num_cpus_per_env_runner=3,
                    rollout_fragment_length=rollout_fragment_length,
                    batch_mode="complete_episodes",
                    sample_timeout_s=600,
                )
                .resources(
                    num_cpus_for_main_process=4,
                    num_gpus=0,
                )
                .callbacks(EpisodeMetricsCallbacks)
            )
        else:
            algo_config = (
                SACConfig()
                .api_stack(
                    enable_rl_module_and_learner=False,
                    enable_env_runner_and_connector_v2=False,
                )
                .environment(env=env_name, env_config=base_env_config)
                .framework("torch")
                .multi_agent(
                    policies=policies,
                    policy_mapping_fn=policy_mapping_fn,
                    policies_to_train=policies_to_train,
                )
                .training(
                    train_batch_size=args.train_batch_size,
                    gamma=0.99,
                    lr=0.0003,
                    replay_buffer_config={
                        "type": "MultiAgentPrioritizedReplayBuffer",
                        "capacity": int(1e5),
                        "prioritized_replay_alpha": 0.6,
                        "prioritized_replay_beta": 0.4,
                        "prioritized_replay_eps": 1e-6,
                    },
                    num_steps_sampled_before_learning_starts=1000,
                )
                .learners(num_learners=1)
                .env_runners(
                    num_env_runners=args.num_workers,
                    num_envs_per_env_runner=args.num_envs_per_worker,
                    num_cpus_per_env_runner=3,
                    rollout_fragment_length=rollout_fragment_length,
                    batch_mode="complete_episodes",
                    sample_timeout_s=600,
                )
                .resources(
                    num_cpus_for_main_process=4,
                    num_gpus=0,
                )
                .callbacks(EpisodeMetricsCallbacks)
            )

        if args.eval_interval > 0 and args.eval_episodes > 0:
            eval_env_config = base_env_config.copy()
            eval_env_config["max_steps"] = min(
                int(base_env_config.get("max_steps", 2000)), 1000
            )
            eval_env_config["allow_empty_predator_population"] = True
            eval_config = {
                "explore": False,
                "env_config": eval_env_config,
                "enable_env_runner_and_connector_v2": False,
                "enable_rl_module_and_learner": False,
                "batch_mode": "complete_episodes",
            }
            algo_config = algo_config.evaluation(
                evaluation_interval=args.eval_interval,
                evaluation_duration=args.eval_episodes,
                evaluation_duration_unit="episodes",
                evaluation_num_workers=max(1, args.eval_num_workers),
                evaluation_parallel_to_training=False,
                evaluation_config=eval_config,
            )
            logger.info(
                "Evaluation enabled: interval=%s iterations, episodes=%s, workers=%s",
                args.eval_interval,
                args.eval_episodes,
                max(1, args.eval_num_workers),
            )
        else:
            logger.info(
                "Evaluation disabled (eval_interval=%s, eval_episodes=%s).",
                args.eval_interval,
                args.eval_episodes,
            )

        logger.info(
            "Starting training for %s iterations (checkpoint_freq=%s, resume=%s)",
            args.num_iterations,
            args.checkpoint_freq,
            args.resume,
        )
        logger.info("Environment logs directory: %s", env_log_dir.resolve())
        print(f"\nStarting training for {args.num_iterations} iterations...")

        stopper = MaximumIterationStopper(args.num_iterations)
        if args.early_stop_patience > 0:
            stopper = CombinedStopper(
                stopper,
                RewardPlateauStopper(args.early_stop_metric, args.early_stop_patience),
            )

        checkpoint_num_to_keep = None if args.keep_all_checkpoints else 5
        tuner = Tuner(
            algo_config.algo_class,
            param_space=algo_config,
            run_config=RunConfig(
                storage_path=str(ray_results_dir.resolve()),
                name="PPO_PredPreyGrass_continuous_simple",
                stop=stopper,
                checkpoint_config=CheckpointConfig(
                    num_to_keep=checkpoint_num_to_keep,
                    checkpoint_frequency=args.checkpoint_freq,
                    checkpoint_at_end=True,
                ),
            ),
        )
        trainable_cls = algo_config.algo_class

    # 运行训练
    try:
        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
            logger.info("Resuming from checkpoint: %s", args.resume)
            tuner = Tuner.restore(args.resume, trainable_cls)
        
        results = tuner.fit()
        logger.info("Tuner finished execution.")
        
        best_result = results.get_best_result()
        print(f"\n{'='*80}")
        print("Training Completed!")
        print(f"Best checkpoint: {best_result.checkpoint}")
        print(f"{'='*80}\n")
        logger.info("Training completed. Best checkpoint: %s", best_result.checkpoint)
        run_metadata["status"] = "completed"
        run_metadata["best_checkpoint"] = (
            str(best_result.checkpoint) if best_result and best_result.checkpoint else None
        )
        run_metadata["best_result_log_dir"] = str(
            getattr(best_result, "log_dir", "") or ""
        )
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        logger.warning("Training interrupted by user.")
        run_metadata["status"] = "interrupted"
    except Exception as exc:
        logger.exception("Training failed with exception: %s", exc)
        run_metadata["status"] = "failed"
        run_metadata["error"] = str(exc)
        raise
    finally:
        _write_run_metadata(run_log_dir, run_metadata)
        logger.info("Shutting down Ray.")
        ray.shutdown()


if __name__ == "__main__":
    main()
