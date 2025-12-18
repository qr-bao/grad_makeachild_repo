"""
Population-based hyperparameter optimisation inside a single coexisting multi-agent run (SAC).

Goal:
  - 5 predator populations + 5 prey populations (n_populations in env config)
  - All populations use SAC, but each population has its own independent SAC instance
    (weights + optimiser + hyperparameters).
  - All populations act and learn simultaneously in the same environment.
  - Periodically apply PBT: copy weights+hyperparams from elites and mutate.

Notes:
  - We maintain a lightweight per-population replay buffer in Python (list of transitions).
  - RLlib SAC policies are updated via policy.postprocess_trajectory + learn_on_batch.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import ray
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except Exception:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

from predpreygrass.rllib.env3.predpreygrass_rllib_env129.predpreygrass_rllib_env import (
    PredPreyGrass,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.train_simple import (
    create_model_config,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.utils.progress import ProgressBar


ALGO_PREFIX = "sac"


@dataclass(frozen=True)
class SACHyperparams:
    actor_lr: float
    critic_lr: float
    alpha_lr: float
    tau: float
    initial_alpha: float
    target_entropy: float
    train_batch_size: int
    warmup_steps: int
    target_update_freq: int

    def clamp(self) -> "SACHyperparams":
        def _clip(v, lo, hi):
            return float(np.clip(v, lo, hi))

        return SACHyperparams(
            actor_lr=_clip(self.actor_lr, 1e-6, 5e-3),
            critic_lr=_clip(self.critic_lr, 1e-6, 5e-3),
            alpha_lr=_clip(self.alpha_lr, 1e-6, 5e-3),
            tau=_clip(self.tau, 1e-4, 0.02),
            initial_alpha=_clip(self.initial_alpha, 1e-4, 2.0),
            target_entropy=_clip(self.target_entropy, -10.0, 0.0),
            train_batch_size=int(np.clip(self.train_batch_size, 32, 2048)),
            warmup_steps=int(np.clip(self.warmup_steps, 0, 200000)),
            target_update_freq=int(np.clip(self.target_update_freq, 1, 1000)),
        )


def _algo_key(species: str, pop_id: int) -> str:
    return f"{ALGO_PREFIX}_{species}_{pop_id}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("PBT orchestrator for 10 independent SAC populations")
    parser.add_argument("--env-config-file", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/pbt_orchestrator_sac"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--n-populations", type=int, default=None)

    parser.add_argument("--pbt-interval", type=int, default=20)
    parser.add_argument("--elite-fraction", type=float, default=0.4)
    parser.add_argument("--buffer-capacity", type=int, default=50000)
    parser.add_argument("--updates-per-episode", type=int, default=20)

    # Init distributions (log-normal for lrs, normal for others).
    parser.add_argument("--init-lr-log-mean", type=float, default=math.log(3e-4))
    parser.add_argument("--init-lr-log-std", type=float, default=0.7)
    parser.add_argument("--init-tau-mean", type=float, default=0.005)
    parser.add_argument("--init-tau-std", type=float, default=0.002)
    parser.add_argument("--init-alpha-mean", type=float, default=0.2)
    parser.add_argument("--init-alpha-std", type=float, default=0.2)
    parser.add_argument("--init-target-entropy", type=float, default=-2.0)
    parser.add_argument("--init-train-batch", type=int, default=256)
    parser.add_argument("--init-warmup", type=int, default=2000)
    parser.add_argument("--init-target-update-freq", type=int, default=1)

    # Mutation magnitudes.
    parser.add_argument("--mutate-lr-sigma", type=float, default=0.25)
    parser.add_argument("--mutate-tau-sigma", type=float, default=0.001)
    parser.add_argument("--mutate-alpha-sigma", type=float, default=0.05)
    parser.add_argument("--mutate-target-entropy-sigma", type=float, default=0.5)
    parser.add_argument("--mutate-batch-scale", type=float, default=0.25)

    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--progress-every-steps", type=int, default=50)
    parser.add_argument("--save-best", action="store_true")
    return parser


def load_env_config(path: Path | None) -> dict:
    from predpreygrass.rllib.env3.predpreygrass_rllib_env129.config.config_env_train import (
        config_env as default_env_config,
    )

    cfg = dict(default_env_config)
    if path is not None:
        p = path.expanduser().resolve()
        with p.open("r", encoding="utf-8") as fp:
            cfg.update(json.load(fp))
    return cfg


def _lognormal(rng: np.random.Generator, log_mean: float, log_std: float) -> float:
    return float(math.exp(rng.normal(log_mean, log_std)))


def sample_initial_hparams(rng: np.random.Generator, args: argparse.Namespace) -> SACHyperparams:
    lr = _lognormal(rng, args.init_lr_log_mean, args.init_lr_log_std)
    tau = float(max(1e-4, rng.normal(args.init_tau_mean, args.init_tau_std)))
    init_alpha = float(max(1e-4, rng.normal(args.init_alpha_mean, args.init_alpha_std)))
    return SACHyperparams(
        actor_lr=lr,
        critic_lr=lr,
        alpha_lr=lr,
        tau=tau,
        initial_alpha=init_alpha,
        target_entropy=float(args.init_target_entropy),
        train_batch_size=int(args.init_train_batch),
        warmup_steps=int(args.init_warmup),
        target_update_freq=int(args.init_target_update_freq),
    ).clamp()


def mutate_hparams(rng: np.random.Generator, base: SACHyperparams, args: argparse.Namespace) -> SACHyperparams:
    def _mut_lr(x: float) -> float:
        return float(x * math.exp(rng.normal(0.0, args.mutate_lr_sigma)))

    actor_lr = _mut_lr(base.actor_lr)
    critic_lr = _mut_lr(base.critic_lr)
    alpha_lr = _mut_lr(base.alpha_lr)
    tau = float(base.tau + rng.normal(0.0, args.mutate_tau_sigma))
    init_alpha = float(base.initial_alpha + rng.normal(0.0, args.mutate_alpha_sigma))
    target_entropy = float(base.target_entropy + rng.normal(0.0, args.mutate_target_entropy_sigma))
    # batch size multiplicative (rounded to 32)
    batch = int(round(base.train_batch_size * (1.0 + rng.normal(0.0, args.mutate_batch_scale)) / 32.0) * 32)
    return SACHyperparams(
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        alpha_lr=alpha_lr,
        tau=tau,
        initial_alpha=init_alpha,
        target_entropy=target_entropy,
        train_batch_size=batch,
        warmup_steps=base.warmup_steps,
        target_update_freq=base.target_update_freq,
    ).clamp()


def build_sac_algo(env_config: dict, obs_space, act_space, hparams: SACHyperparams):
    cfg = SACConfig()
    if hasattr(cfg, "api_stack"):
        cfg = cfg.api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    if hasattr(cfg, "_enable_rl_module_api"):
        cfg._enable_rl_module_api = False
    if hasattr(cfg, "_enable_learner_api"):
        cfg._enable_learner_api = False
    cfg = cfg.framework("torch").environment(
        env=PredPreyGrass,
        env_config=env_config,
        disable_env_checking=True,
        observation_space=obs_space,
        action_space=act_space,
    )
    if hasattr(cfg, "auto_wrap_old_gym_envs"):
        cfg.auto_wrap_old_gym_envs = False
    if hasattr(cfg, "env_runners"):
        cfg = cfg.env_runners(num_env_runners=0)
    elif hasattr(cfg, "rollouts"):
        cfg = cfg.rollouts(num_rollout_workers=0, num_envs_per_worker=1)
    cfg = cfg.training(
        actor_lr=hparams.actor_lr,
        critic_lr=hparams.critic_lr,
        alpha_lr=hparams.alpha_lr,
        tau=hparams.tau,
        initial_alpha=hparams.initial_alpha,
        target_entropy=hparams.target_entropy,
        target_network_update_freq=hparams.target_update_freq,
        num_steps_sampled_before_learning_starts=hparams.warmup_steps,
        train_batch_size=hparams.train_batch_size,
        replay_buffer_config={
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": int(1e5),
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta": 0.4,
            "prioritized_replay_eps": 1e-6,
        },
        model=create_model_config(),
    )
    return cfg.build()


def apply_sac_hparams_to_policy(policy, hparams: SACHyperparams) -> None:
    policy.config["actor_lr"] = float(hparams.actor_lr)
    policy.config["critic_lr"] = float(hparams.critic_lr)
    policy.config["alpha_lr"] = float(hparams.alpha_lr)
    policy.config["tau"] = float(hparams.tau)
    policy.config["initial_alpha"] = float(hparams.initial_alpha)
    policy.config["target_entropy"] = float(hparams.target_entropy)
    policy.config["target_network_update_freq"] = int(hparams.target_update_freq)
    policy.config["num_steps_sampled_before_learning_starts"] = int(hparams.warmup_steps)
    policy.config["train_batch_size"] = int(hparams.train_batch_size)
    # Best-effort update learning rates on optimizers (TorchPolicy).
    try:
        optimizers = getattr(policy, "_optimizers", None) or getattr(policy, "optimizers", lambda: [])()
        for opt in optimizers or []:
            for group in opt.param_groups:
                # If the optimiser is shared across modules, just apply actor_lr.
                group["lr"] = float(hparams.actor_lr)
    except Exception:
        pass


def _score_window(values: List[float], window: int) -> float:
    if not values:
        return float("-inf")
    if window <= 1:
        return float(values[-1])
    recent = values[-window:]
    return float(np.mean(recent))


def _select_and_mutate(
    rng: np.random.Generator,
    keys: List[str],
    scores: Dict[str, float],
    hparams: Dict[str, SACHyperparams],
    policies: Dict[str, any],
    buffers: Dict[str, List[dict]],
    args: argparse.Namespace,
    *,
    writer: SummaryWriter | None,
    episode: int,
) -> None:
    if not keys:
        return
    ranked = sorted(keys, key=lambda k: scores.get(k, float("-inf")), reverse=True)
    n_elite = max(1, int(math.ceil(len(ranked) * float(args.elite_fraction))))
    elites = ranked[:n_elite]
    losers = ranked[n_elite:]
    if not losers:
        return
    for loser in losers:
        donor = rng.choice(elites)
        donor_hp = hparams[donor]
        new_hp = mutate_hparams(rng, donor_hp, args)
        try:
            policies[loser].set_weights(policies[donor].get_weights())
        except Exception:
            try:
                policies[loser].set_state(policies[donor].get_state())
            except Exception:
                pass
        hparams[loser] = new_hp
        apply_sac_hparams_to_policy(policies[loser], new_hp)
        # Clear replay buffer to avoid mixing distributions after weight copy.
        buffers[loser].clear()

        if writer is not None:
            writer.add_scalar(f"pbt/{loser}/lr", new_hp.actor_lr, episode)
            writer.add_scalar(f"pbt/{loser}/tau", new_hp.tau, episode)
            writer.add_scalar(f"pbt/{loser}/alpha", new_hp.initial_alpha, episode)
            writer.add_scalar(f"pbt/{loser}/target_entropy", new_hp.target_entropy, episode)


def main() -> None:
    args = build_parser().parse_args()
    log_dir = args.log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    env_config = load_env_config(args.env_config_file)
    env_config["max_steps"] = int(args.max_steps)
    env_config["verbose_spawning"] = False
    env_config["verbose_engagement"] = False
    env_config["verbose_movement"] = False

    sample_env = PredPreyGrass(env_config)
    obs_space = next(iter(sample_env.observation_spaces.values()))
    act_space = next(iter(sample_env.action_spaces.values()))
    n_pops = int(getattr(sample_env, "n_populations", env_config.get("n_populations", 5)))
    if hasattr(sample_env, "close"):
        sample_env.close()
    if args.n_populations is not None:
        n_pops = int(args.n_populations)
        env_config["n_populations"] = n_pops

    predator_keys = [_algo_key("predator", i) for i in range(n_pops)]
    prey_keys = [_algo_key("prey", i) for i in range(n_pops)]
    keys = predator_keys + prey_keys

    hparam_map: Dict[str, SACHyperparams] = {k: sample_initial_hparams(rng, args) for k in keys}
    algos = {}
    policies = {}
    replay: Dict[str, List[dict]] = {k: [] for k in keys}
    returns_hist: Dict[str, List[float]] = {k: [] for k in keys}

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    writer = None
    if not args.no_tensorboard and TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=str(log_dir))

    try:
        for key in keys:
            algo = build_sac_algo(env_config, obs_space, act_space, hparam_map[key])
            pol = algo.get_policy()
            apply_sac_hparams_to_policy(pol, hparam_map[key])
            algos[key] = algo
            policies[key] = pol

        env = PredPreyGrass(env_config)
        best_seen = {k: float("-inf") for k in keys}

        progress = None
        if not args.no_progress:
            progress = ProgressBar(int(args.num_episodes), label="[PBT-SAC] ", stream=sys.stderr)
            progress.update(0)

        for ep in range(int(args.num_episodes)):
            obs, infos = env.reset(seed=int(args.seed) + ep)
            ep_ret_by_key: Dict[str, List[float]] = {k: [] for k in keys}
            ep_ret_by_agent: Dict[str, float] = {}
            meta_by_agent: Dict[str, dict] = {}

            for t in range(int(args.max_steps)):
                groups: Dict[str, List[str]] = {}
                for aid in obs.keys():
                    info = infos.get(aid, {}) or {}
                    species = info.get("species") or ("predator" if "predator" in aid else "prey")
                    pop_id = info.get("population_id")
                    if pop_id is None:
                        pop_id = getattr(env, "agent_population_id", {}).get(aid, 0)
                    key = _algo_key(species, int(pop_id))
                    groups.setdefault(key, []).append(aid)
                    meta_by_agent[aid] = {"key": key}

                actions: Dict[str, np.ndarray] = {}
                for key, aids in groups.items():
                    pol = policies[key]
                    obs_batch = np.stack([obs[aid] for aid in aids]).astype(np.float32)
                    act_b, _, _extra = pol.compute_actions_from_input_dict(
                        {SampleBatch.OBS: obs_batch}, explore=True
                    )
                    for i, aid in enumerate(aids):
                        actions[aid] = act_b[i]

                next_obs, rewards, dones, truncs, infos = env.step(actions)

                for aid, rew in rewards.items():
                    if aid not in obs:
                        continue
                    key = meta_by_agent.get(aid, {}).get("key")
                    if not key:
                        continue
                    ep_ret_by_agent[aid] = ep_ret_by_agent.get(aid, 0.0) + float(rew)

                    transition = {
                        "obs": obs[aid],
                        "actions": actions[aid],
                        "rewards": float(rew),
                        "dones": bool(dones.get(aid, False) or truncs.get(aid, False)),
                        "next_obs": next_obs.get(aid, obs[aid]),
                        "t": t,
                    }
                    replay[key].append(transition)
                    if len(replay[key]) > int(args.buffer_capacity):
                        replay[key] = replay[key][-int(args.buffer_capacity) :]

                obs = next_obs
                if dones.get("__all__") or truncs.get("__all__"):
                    break
                if progress is not None and int(args.progress_every_steps) > 0:
                    every = int(args.progress_every_steps)
                    if (t + 1) % every == 0:
                        progress.update(
                            float(ep) + float(t + 1) / float(max(1, int(args.max_steps))),
                            detail=f"ep={ep+1}/{int(args.num_episodes)} step={t+1}/{int(args.max_steps)} alive={len(obs)}",
                        )

            # Per-key return mean for scoring.
            for aid, ret in ep_ret_by_agent.items():
                key = meta_by_agent.get(aid, {}).get("key")
                if key:
                    ep_ret_by_key[key].append(float(ret))
            for key, vals in ep_ret_by_key.items():
                if not vals:
                    continue
                mean_ret = float(np.mean(vals))
                returns_hist[key].append(mean_ret)
                best_seen[key] = max(best_seen[key], mean_ret)
                if writer is not None:
                    writer.add_scalar(f"episode_return_mean/{key}", mean_ret, ep)

            # SAC updates.
            for key in keys:
                pol = policies[key]
                buf = replay[key]
                hp = hparam_map[key]
                if len(buf) < max(int(hp.warmup_steps), int(hp.train_batch_size)):
                    continue
                for _ in range(int(args.updates_per_episode)):
                    batch_size = int(hp.train_batch_size)
                    idx = rng.choice(len(buf), size=batch_size, replace=len(buf) < batch_size)
                    sample = [buf[i] for i in idx]
                    batch = SampleBatch(
                        {
                            SampleBatch.OBS: np.stack([s["obs"] for s in sample]).astype(np.float32),
                            SampleBatch.ACTIONS: np.stack([s["actions"] for s in sample]).astype(np.float32),
                            SampleBatch.REWARDS: np.array([s["rewards"] for s in sample], dtype=np.float32),
                            SampleBatch.DONES: np.array([s["dones"] for s in sample], dtype=np.bool_),
                            SampleBatch.TERMINATEDS: np.array([s["dones"] for s in sample], dtype=np.bool_),
                            SampleBatch.TRUNCATEDS: np.zeros(len(sample), dtype=np.bool_),
                            SampleBatch.NEXT_OBS: np.stack([s["next_obs"] for s in sample]).astype(np.float32),
                            PRIO_WEIGHTS: np.ones(len(sample), dtype=np.float32),
                            SampleBatch.EPS_ID: np.zeros(len(sample), dtype=np.int32),
                            SampleBatch.AGENT_INDEX: np.zeros(len(sample), dtype=np.int32),
                            SampleBatch.T: np.array([s["t"] for s in sample], dtype=np.int32),
                            SampleBatch.UNROLL_ID: np.zeros(len(sample), dtype=np.int32),
                        }
                    )
                    post = pol.postprocess_trajectory(batch, other_agent_batches=None, episode=None)
                    pol.learn_on_batch(post)
                    if hasattr(pol, "update_target"):
                        pol.update_target()

            # PBT step.
            if args.pbt_interval > 0 and (ep + 1) % int(args.pbt_interval) == 0:
                window = int(args.pbt_interval)
                scores = {k: _score_window(returns_hist[k], window) for k in keys}
                _select_and_mutate(
                    rng,
                    predator_keys,
                    scores,
                    hparam_map,
                    policies,
                    replay,
                    args,
                    writer=writer,
                    episode=ep,
                )
                _select_and_mutate(
                    rng,
                    prey_keys,
                    scores,
                    hparam_map,
                    policies,
                    replay,
                    args,
                    writer=writer,
                    episode=ep,
                )
                if writer is not None:
                    for k in keys:
                        hp = hparam_map[k]
                        writer.add_scalar(f"hparams/{k}/actor_lr", hp.actor_lr, ep)
                        writer.add_scalar(f"hparams/{k}/tau", hp.tau, ep)
                        writer.add_scalar(f"hparams/{k}/alpha", hp.initial_alpha, ep)
                        writer.add_scalar(f"hparams/{k}/target_entropy", hp.target_entropy, ep)
                        writer.add_scalar(f"hparams/{k}/train_batch_size", hp.train_batch_size, ep)

            if writer is not None:
                writer.add_scalar("episode_len", t + 1, ep)
            if (ep + 1) % max(1, int(args.pbt_interval)) == 0:
                print(f"[PBT-SAC] episode={ep} done (len={t+1})")
            if progress is not None:
                progress.update(
                    float(ep + 1),
                    detail=f"ep={ep+1}/{int(args.num_episodes)} last_len={t+1}/{int(args.max_steps)}",
                )

        # Final: best by last-window mean (per species).
        window = max(1, int(args.pbt_interval))
        final_scores = {k: _score_window(returns_hist[k], window) for k in keys}
        best_pred = max(predator_keys, key=lambda k: final_scores.get(k, float("-inf")))
        best_prey = max(prey_keys, key=lambda k: final_scores.get(k, float("-inf")))

        summary = {
            "best_predator_key": best_pred,
            "best_prey_key": best_prey,
            "best_predator_hparams": asdict(hparam_map[best_pred]),
            "best_prey_hparams": asdict(hparam_map[best_prey]),
            "final_scores": final_scores,
            "best_score_seen": best_seen,
        }
        (log_dir / "pbt_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("[PBT-SAC] best predator:", best_pred, hparam_map[best_pred])
        print("[PBT-SAC] best prey:", best_prey, hparam_map[best_prey])
        print("[PBT-SAC] summary written to:", (log_dir / "pbt_summary.json"))

        if args.save_best:
            ckpt_dir = log_dir / "checkpoints"
            try:
                algos[best_pred].save(str(ckpt_dir / best_pred))
            except Exception as exc:
                print("[WARN] failed to save best_pred:", exc)
            try:
                algos[best_prey].save(str(ckpt_dir / best_prey))
            except Exception as exc:
                print("[WARN] failed to save best_prey:", exc)

        if writer is not None:
            writer.close()
        env.close()
        if progress is not None:
            progress.close()
    finally:
        for algo in list(algos.values()):
            try:
                algo.stop()
            except Exception:
                pass
        ray.shutdown()


if __name__ == "__main__":
    main()
