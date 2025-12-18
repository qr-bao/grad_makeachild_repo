"""
Population-based hyperparameter optimisation inside a single coexisting multi-agent run.

Goal:
  - 5 predator populations + 5 prey populations (n_populations in env config)
  - All populations use PPO, but each population has its own independent PPO instance
    (weights + optimiser + hyperparameters).
  - All populations act and learn simultaneously in the same environment.
  - Periodically apply PBT: copy weights+hyperparams from elites and mutate.

This is intentionally lightweight and uses RLlib PPO policies only for action selection
and per-episode learn_on_batch updates (no RLlib rollout workers).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.sample_batch import SampleBatch

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


ALGO_PREFIX = "ppo"


@dataclass(frozen=True)
class PPOHyperparams:
    lr: float
    clip_param: float
    entropy_coeff: float
    gamma: float = 0.99
    lambda_: float = 0.95

    def clamp(self) -> "PPOHyperparams":
        return PPOHyperparams(
            lr=float(np.clip(self.lr, 1e-6, 1e-2)),
            clip_param=float(np.clip(self.clip_param, 0.02, 0.4)),
            entropy_coeff=float(np.clip(self.entropy_coeff, 0.0, 0.05)),
            gamma=float(np.clip(self.gamma, 0.8, 0.999)),
            lambda_=float(np.clip(self.lambda_, 0.8, 0.999)),
        )


def _algo_key(species: str, pop_id: int) -> str:
    return f"{ALGO_PREFIX}_{species}_{pop_id}"


def _species_from_key(key: str) -> str:
    parts = key.split("_", 2)
    if len(parts) < 3:
        return "unknown"
    return parts[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("PBT orchestrator for 10 independent PPO populations")
    parser.add_argument("--env-config-file", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/pbt_orchestrator"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--n-populations",
        type=int,
        default=None,
        help="Override n_populations (useful for quick smoke tests).",
    )
    parser.add_argument("--pbt-interval", type=int, default=20, help="Episodes between PBT exploit/explore.")
    parser.add_argument("--elite-fraction", type=float, default=0.4, help="Elite fraction per species.")
    parser.add_argument("--mutate-lr-sigma", type=float, default=0.25, help="Log-space sigma for lr mutation.")
    parser.add_argument("--mutate-clip-sigma", type=float, default=0.03)
    parser.add_argument("--mutate-entropy-sigma", type=float, default=0.003)
    parser.add_argument("--init-lr-log-mean", type=float, default=math.log(3e-4))
    parser.add_argument("--init-lr-log-std", type=float, default=0.6)
    parser.add_argument("--init-clip-mean", type=float, default=0.2)
    parser.add_argument("--init-clip-std", type=float, default=0.06)
    parser.add_argument("--init-entropy-mean", type=float, default=0.005)
    parser.add_argument("--init-entropy-std", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.95)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--progress-every-steps", type=int, default=50)
    parser.add_argument("--save-best", action="store_true", help="Save best predator/prey policy checkpoints.")
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


def sample_initial_hparams(rng: np.random.Generator, args: argparse.Namespace) -> PPOHyperparams:
    lr = float(math.exp(rng.normal(args.init_lr_log_mean, args.init_lr_log_std)))
    clip_param = float(rng.normal(args.init_clip_mean, args.init_clip_std))
    entropy = float(rng.normal(args.init_entropy_mean, args.init_entropy_std))
    return PPOHyperparams(
        lr=lr,
        clip_param=clip_param,
        entropy_coeff=entropy,
        gamma=float(args.gamma),
        lambda_=float(args.lambda_),
    ).clamp()


def mutate_hparams(
    rng: np.random.Generator,
    base: PPOHyperparams,
    args: argparse.Namespace,
) -> PPOHyperparams:
    lr = float(base.lr * math.exp(rng.normal(0.0, args.mutate_lr_sigma)))
    clip_param = float(base.clip_param + rng.normal(0.0, args.mutate_clip_sigma))
    entropy = float(base.entropy_coeff + rng.normal(0.0, args.mutate_entropy_sigma))
    return PPOHyperparams(
        lr=lr,
        clip_param=clip_param,
        entropy_coeff=entropy,
        gamma=base.gamma,
        lambda_=base.lambda_,
    ).clamp()


def build_ppo_algo(env_config: dict, obs_space, act_space, hparams: PPOHyperparams):
    cfg = PPOConfig()
    if hasattr(cfg, "api_stack"):
        cfg = cfg.api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    if hasattr(cfg, "_enable_rl_module_api"):
        cfg._enable_rl_module_api = False
    if hasattr(cfg, "_enable_learner_api"):
        cfg._enable_learner_api = False
    cfg = (
        cfg.framework("torch")
        .environment(
            env=PredPreyGrass,
            env_config=env_config,
            disable_env_checking=True,
            observation_space=obs_space,
            action_space=act_space,
        )
    )
    if hasattr(cfg, "auto_wrap_old_gym_envs"):
        cfg.auto_wrap_old_gym_envs = False
    if hasattr(cfg, "env_runners"):
        cfg = cfg.env_runners(num_env_runners=0)
    elif hasattr(cfg, "rollouts"):
        cfg = cfg.rollouts(num_rollout_workers=0, num_envs_per_worker=1)
    cfg = cfg.training(
        gamma=hparams.gamma,
        lambda_=hparams.lambda_,
        lr=hparams.lr,
        clip_param=hparams.clip_param,
        entropy_coeff=hparams.entropy_coeff,
        model=create_model_config(),
        train_batch_size=4000,
        minibatch_size=512,
        num_epochs=4,
    )
    return cfg.build()


def apply_ppo_hparams_to_policy(policy, hparams: PPOHyperparams) -> None:
    # Update config values used by the loss.
    policy.config["lr"] = float(hparams.lr)
    policy.config["clip_param"] = float(hparams.clip_param)
    policy.config["entropy_coeff"] = float(hparams.entropy_coeff)
    policy.config["gamma"] = float(hparams.gamma)
    policy.config["lambda_"] = float(hparams.lambda_)
    # Update optimiser learning rate (TorchPolicy).
    try:
        optimizers = getattr(policy, "_optimizers", None) or getattr(policy, "optimizers", lambda: [])()
        if optimizers:
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = float(hparams.lr)
    except Exception:
        # Best effort.
        pass


def _build_agent_batches(traj_by_agent: Dict[str, List[dict]]) -> List[SampleBatch]:
    batches = []
    for _aid, steps in traj_by_agent.items():
        if not steps:
            continue
        batches.append(
            SampleBatch(
                {
                    SampleBatch.OBS: np.stack([s["obs"] for s in steps]).astype(np.float32),
                    SampleBatch.ACTIONS: np.stack([s["actions"] for s in steps]).astype(np.float32),
                    SampleBatch.REWARDS: np.asarray([s["rewards"] for s in steps], dtype=np.float32),
                    SampleBatch.NEXT_OBS: np.stack([s["next_obs"] for s in steps]).astype(np.float32),
                    SampleBatch.TERMINATEDS: np.asarray([s["terminated"] for s in steps], dtype=np.bool_),
                    SampleBatch.TRUNCATEDS: np.asarray([s["truncated"] for s in steps], dtype=np.bool_),
                    SampleBatch.ACTION_LOGP: np.asarray([s["logp"] for s in steps], dtype=np.float32),
                    SampleBatch.VF_PREDS: np.asarray([s["vf"] for s in steps], dtype=np.float32),
                }
            )
        )
    return batches


def select_and_mutate(
    rng: np.random.Generator,
    keys: List[str],
    scores: Dict[str, float],
    hparams: Dict[str, PPOHyperparams],
    policies: Dict[str, any],
    elite_fraction: float,
    args: argparse.Namespace,
    *,
    writer: SummaryWriter | None,
    episode: int,
) -> None:
    if not keys:
        return
    ranked = sorted(keys, key=lambda k: scores.get(k, float("-inf")), reverse=True)
    n_elite = max(1, int(math.ceil(len(ranked) * float(elite_fraction))))
    elites = ranked[:n_elite]
    losers = ranked[n_elite:]
    if not losers:
        return

    for loser in losers:
        donor = rng.choice(elites)
        donor_hp = hparams[donor]
        new_hp = mutate_hparams(rng, donor_hp, args)
        # Copy weights.
        try:
            policies[loser].set_weights(policies[donor].get_weights())
        except Exception:
            # Fallback to algorithm-level methods if present.
            try:
                policies[loser].set_state(policies[donor].get_state())
            except Exception:
                pass
        # Apply mutated hyperparams.
        hparams[loser] = new_hp
        apply_ppo_hparams_to_policy(policies[loser], new_hp)

        if writer is not None:
            prefix = f"pbt/{loser}"
            writer.add_scalar(f"{prefix}/copied_from", float(int(donor.split("_")[-1])), episode)
            writer.add_scalar(f"{prefix}/lr", new_hp.lr, episode)
            writer.add_scalar(f"{prefix}/clip_param", new_hp.clip_param, episode)
            writer.add_scalar(f"{prefix}/entropy_coeff", new_hp.entropy_coeff, episode)


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

    # Sample env to grab spaces and population count.
    sample_env = PredPreyGrass(env_config)
    obs_space = next(iter(sample_env.observation_spaces.values()))
    act_space = next(iter(sample_env.action_spaces.values()))
    n_pops = int(getattr(sample_env, "n_populations", env_config.get("n_populations", 5)))
    if hasattr(sample_env, "close"):
        sample_env.close()
    if args.n_populations is not None:
        n_pops = int(args.n_populations)
        env_config["n_populations"] = n_pops

    # Build independent PPO per population (10 in total).
    keys: List[str] = []
    predator_keys = []
    prey_keys = []
    hparam_map: Dict[str, PPOHyperparams] = {}
    algos = {}
    policies = {}

    for pop_id in range(n_pops):
        for species in ("predator", "prey"):
            key = _algo_key(species, pop_id)
            keys.append(key)
            (predator_keys if species == "predator" else prey_keys).append(key)
            hparam_map[key] = sample_initial_hparams(rng, args)

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    try:
        for key in keys:
            algo = build_ppo_algo(env_config, obs_space, act_space, hparam_map[key])
            pol = algo.get_policy()
            # Ensure policy reflects our hyperparams.
            apply_ppo_hparams_to_policy(pol, hparam_map[key])
            algos[key] = algo
            policies[key] = pol

        writer = None
        if not args.no_tensorboard and TENSORBOARD_AVAILABLE:
            writer = SummaryWriter(log_dir=str(log_dir))

        # Runtime env (single instance).
        env = PredPreyGrass(env_config)

        # Keep sliding rewards for PBT ranking.
        recent_returns: Dict[str, List[float]] = {k: [] for k in keys}
        best_score: Dict[str, float] = {k: float("-inf") for k in keys}
        best_episode: Dict[str, int] = {k: -1 for k in keys}

        progress = None
        if not args.no_progress:
            progress = ProgressBar(int(args.num_episodes), label="[PBT-PPO] ", stream=sys.stderr)
            progress.update(0)

        for ep in range(int(args.num_episodes)):
            obs, infos = env.reset(seed=int(args.seed) + ep)
            traj: Dict[str, Dict[str, List[dict]]] = {k: {} for k in keys}
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

                actions: Dict[str, np.ndarray] = {}
                extra_cache: Dict[str, dict] = {}
                for key, aids in groups.items():
                    pol = policies[key]
                    obs_batch = np.stack([obs[aid] for aid in aids]).astype(np.float32)
                    act_b, _, extra = pol.compute_actions_from_input_dict(
                        {SampleBatch.OBS: obs_batch}, explore=True
                    )
                    logp_b = extra.get("action_logp", None)
                    vf_b = extra.get("vf_preds", None)
                    for i, aid in enumerate(aids):
                        actions[aid] = act_b[i]
                        extra_cache[aid] = {
                            "logp": float(logp_b[i]) if logp_b is not None else 0.0,
                            "vf": float(vf_b[i]) if vf_b is not None else 0.0,
                            "key": key,
                        }

                next_obs, rewards, dones, truncs, infos = env.step(actions)

                for aid, rew in rewards.items():
                    if aid not in obs:
                        continue
                    info = infos.get(aid, {}) or {}
                    species = info.get("species") or ("predator" if "predator" in aid else "prey")
                    pop_id = info.get("population_id")
                    if pop_id is None:
                        pop_id = getattr(env, "agent_population_id", {}).get(aid, 0)
                    key = _algo_key(species, int(pop_id))
                    meta_by_agent[aid] = {"species": species, "population_id": int(pop_id), "key": key}
                    ep_ret_by_agent[aid] = ep_ret_by_agent.get(aid, 0.0) + float(rew)

                    per_agent_steps = traj[key].setdefault(aid, [])
                    per_agent_steps.append(
                        {
                            "obs": obs[aid],
                            "actions": actions[aid],
                            "rewards": float(rew),
                            "next_obs": next_obs.get(aid, obs[aid]),
                            "terminated": bool(dones.get(aid, False)),
                            "truncated": bool(truncs.get(aid, False)),
                            "logp": extra_cache.get(aid, {}).get("logp", 0.0),
                            "vf": extra_cache.get(aid, {}).get("vf", 0.0),
                        }
                    )

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

            # PPO update per key using collected on-policy trajectories.
            for key in keys:
                agent_batches = _build_agent_batches(traj[key])
                if not agent_batches:
                    continue
                for b in agent_batches:
                    # Ensure dist inputs/VF preds are present for PPO loss computation.
                    _, _, extra = policies[key].compute_actions_from_input_dict(
                        {SampleBatch.OBS: b[SampleBatch.OBS]}, explore=False
                    )
                    if SampleBatch.ACTION_DIST_INPUTS in extra:
                        b[SampleBatch.ACTION_DIST_INPUTS] = np.asarray(extra[SampleBatch.ACTION_DIST_INPUTS])
                    if SampleBatch.VF_PREDS in extra:
                        b[SampleBatch.VF_PREDS] = np.asarray(extra[SampleBatch.VF_PREDS])
                    post = policies[key].postprocess_trajectory(b, other_agent_batches=None, episode=None)
                    policies[key].learn_on_batch(post)

            # Aggregate per-key episode return means.
            ep_ret_by_key: Dict[str, List[float]] = {k: [] for k in keys}
            for aid, ret in ep_ret_by_agent.items():
                meta = meta_by_agent.get(aid)
                if not meta:
                    continue
                ep_ret_by_key[meta["key"]].append(float(ret))

            for key, vals in ep_ret_by_key.items():
                if not vals:
                    continue
                mean_ret = float(np.mean(vals))
                recent_returns[key].append(mean_ret)
                if len(recent_returns[key]) > int(args.pbt_interval):
                    recent_returns[key] = recent_returns[key][-int(args.pbt_interval) :]
                if mean_ret > best_score[key]:
                    best_score[key] = mean_ret
                    best_episode[key] = ep

                if writer is not None:
                    writer.add_scalar(f"episode_return_mean/{key}", mean_ret, ep)

            if writer is not None:
                writer.add_scalar("episode_len", t + 1, ep)
                pop_counts = getattr(env, "population_counts", {})
                if pop_counts:
                    writer.add_scalar(
                        "population/predator/total",
                        sum(v for k, v in pop_counts.items() if k.startswith("predator_")),
                        ep,
                    )
                    writer.add_scalar(
                        "population/prey/total",
                        sum(v for k, v in pop_counts.items() if k.startswith("prey_")),
                        ep,
                    )

            # PBT step.
            if args.pbt_interval > 0 and (ep + 1) % int(args.pbt_interval) == 0:
                window_scores = {k: float(np.mean(recent_returns[k])) if recent_returns[k] else float("-inf") for k in keys}
                select_and_mutate(
                    rng,
                    predator_keys,
                    window_scores,
                    hparam_map,
                    policies,
                    args.elite_fraction,
                    args,
                    writer=writer,
                    episode=ep,
                )
                select_and_mutate(
                    rng,
                    prey_keys,
                    window_scores,
                    hparam_map,
                    policies,
                    args.elite_fraction,
                    args,
                    writer=writer,
                    episode=ep,
                )
                if writer is not None:
                    # Log snapshot of hyperparams.
                    for k in keys:
                        hp = hparam_map[k]
                        writer.add_scalar(f"hparams/{k}/lr", hp.lr, ep)
                        writer.add_scalar(f"hparams/{k}/clip_param", hp.clip_param, ep)
                        writer.add_scalar(f"hparams/{k}/entropy_coeff", hp.entropy_coeff, ep)

            if (ep + 1) % max(1, int(args.pbt_interval)) == 0:
                print(f"[PBT-PPO] episode={ep} done (len={t+1})")
            if progress is not None:
                progress.update(
                    float(ep + 1),
                    detail=f"ep={ep+1}/{int(args.num_episodes)} last_len={t+1}/{int(args.max_steps)}",
                )

        # Final selection: best mean in the last window.
        final_scores = {k: float(np.mean(recent_returns[k])) if recent_returns[k] else float("-inf") for k in keys}
        best_pred = max(predator_keys, key=lambda k: final_scores.get(k, float("-inf")))
        best_prey = max(prey_keys, key=lambda k: final_scores.get(k, float("-inf")))

        summary = {
            "best_predator_key": best_pred,
            "best_prey_key": best_prey,
            "best_predator_hparams": asdict(hparam_map[best_pred]),
            "best_prey_hparams": asdict(hparam_map[best_prey]),
            "final_scores": final_scores,
            "best_score_seen": best_score,
            "best_episode_seen": best_episode,
        }
        (log_dir / "pbt_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("[PBT-PPO] best predator:", best_pred, hparam_map[best_pred])
        print("[PBT-PPO] best prey:", best_prey, hparam_map[best_prey])
        print("[PBT-PPO] summary written to:", (log_dir / "pbt_summary.json"))

        if args.save_best:
            ckpt_dir = log_dir / "checkpoints"
            try:
                algos[best_pred].save(str(ckpt_dir / f"{best_pred}"))
            except Exception as exc:
                print("[WARN] failed to save best_pred algo:", exc)
            try:
                algos[best_prey].save(str(ckpt_dir / f"{best_prey}"))
            except Exception as exc:
                print("[WARN] failed to save best_prey algo:", exc)

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
