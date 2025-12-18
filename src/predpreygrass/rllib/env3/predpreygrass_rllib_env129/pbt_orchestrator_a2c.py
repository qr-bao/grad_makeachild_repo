"""
Population-based hyperparameter optimisation inside a single coexisting multi-agent run (A2C).

Goal:
  - 5 predator populations + 5 prey populations (n_populations in env config)
  - All populations use A2C, but each population has its own independent A2C model
    (weights + optimiser + hyperparameters).
  - All populations act and learn simultaneously in the same environment.
  - Periodically apply PBT: copy weights+hyperparams from elites and mutate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except Exception:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

from predpreygrass.rllib.env3.predpreygrass_rllib_env129.predpreygrass_rllib_env import (
    PredPreyGrass,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.utils.progress import ProgressBar


ALGO_PREFIX = "a2c"


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256), act_last=True):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, out_dim))
        if act_last:
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleA2C:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_low,
        act_high,
        *,
        lr: float,
        gamma: float,
        entropy_coef: float,
        value_coef: float,
        device: str = "cpu",
    ):
        self.device = device
        self.act_low = torch.tensor(act_low, device=device, dtype=torch.float32)
        self.act_high = torch.tensor(act_high, device=device, dtype=torch.float32)
        self.actor = MLP(obs_dim, act_dim, hidden=(256, 256)).to(device)
        self.value = MLP(obs_dim, 1, hidden=(256, 256), act_last=False).to(device)
        self.log_std = nn.Parameter(torch.zeros(act_dim, device=device))
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.value.parameters()) + [self.log_std],
            lr=float(lr),
        )
        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self._logp_eps = 1e-6

    def set_lr(self, lr: float) -> None:
        for group in self.opt.param_groups:
            group["lr"] = float(lr)

    def act(self, obs_np, deterministic: bool = False):
        with torch.no_grad():
            obs = torch.as_tensor(obs_np, device=self.device, dtype=torch.float32)
            mean = self.actor(obs)
            std = self.log_std.exp()
            dist = D.Normal(mean, std)
            pre_tanh = mean if deterministic else dist.sample()
            action_raw = torch.tanh(pre_tanh)
            logp = dist.log_prob(pre_tanh) - torch.log(1.0 - action_raw.pow(2) + self._logp_eps)
            logp = logp.sum(-1).item()
            value = self.value(obs).item()
            action = action_raw * (self.act_high - self.act_low) / 2 + (self.act_high + self.act_low) / 2
            action = torch.max(torch.min(action, self.act_high), self.act_low)
            return action.detach().cpu().numpy(), logp, value, pre_tanh.detach().cpu().numpy()

    def act_batch(self, obs_batch_np, deterministic: bool = False):
        with torch.no_grad():
            obs = torch.as_tensor(obs_batch_np, device=self.device, dtype=torch.float32)
            mean = self.actor(obs)
            std = self.log_std.exp()
            dist = D.Normal(mean, std)
            pre_tanh = mean if deterministic else dist.sample()
            action_raw = torch.tanh(pre_tanh)
            logp = dist.log_prob(pre_tanh) - torch.log(1.0 - action_raw.pow(2) + self._logp_eps)
            logp = logp.sum(-1)
            value = self.value(obs).squeeze(-1)
            action = action_raw * (self.act_high - self.act_low) / 2 + (self.act_high + self.act_low) / 2
            action = torch.max(torch.min(action, self.act_high), self.act_low)
            return (
                action.detach().cpu().numpy(),
                logp.detach().cpu().numpy(),
                value.detach().cpu().numpy(),
                pre_tanh.detach().cpu().numpy(),
            )

    def train_batch(self, batch: dict) -> dict:
        obs = torch.as_tensor(batch["obs"], device=self.device, dtype=torch.float32)
        pre_tanh = torch.as_tensor(batch["pre_tanh_actions"], device=self.device, dtype=torch.float32)
        rews = torch.as_tensor(batch["rewards"], device=self.device, dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], device=self.device, dtype=torch.float32)

        returns = []
        ret = 0.0
        for r, d in zip(reversed(rews), reversed(dones)):
            ret = r + self.gamma * ret * (1.0 - d)
            returns.append(ret)
        returns = torch.stack(list(reversed(returns))).unsqueeze(-1)

        values = self.value(obs)
        advantages = returns - values.detach()

        mean = self.actor(obs)
        std = self.log_std.exp()
        dist = D.Normal(mean, std)
        action_raw = torch.tanh(pre_tanh)
        logp = dist.log_prob(pre_tanh) - torch.log(1.0 - action_raw.pow(2) + self._logp_eps)
        logp = logp.sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)

        policy_loss = -(logp * advantages).mean()
        value_loss = nn.functional.mse_loss(values, returns)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.mean().item()),
        }

    def export_state(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "value": self.value.state_dict(),
            "log_std": self.log_std.detach().cpu(),
            "opt": self.opt.state_dict(),
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
        }

    def import_state(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.value.load_state_dict(state["value"])
        with torch.no_grad():
            self.log_std.copy_(state["log_std"].to(self.device))
        try:
            self.opt.load_state_dict(state["opt"])
        except Exception:
            pass


@dataclass(frozen=True)
class A2CHyperparams:
    lr: float
    gamma: float
    entropy_coef: float
    value_coef: float
    updates_per_episode: int

    def clamp(self) -> "A2CHyperparams":
        return A2CHyperparams(
            lr=float(np.clip(self.lr, 1e-6, 5e-3)),
            gamma=float(np.clip(self.gamma, 0.8, 0.999)),
            entropy_coef=float(np.clip(self.entropy_coef, 0.0, 0.1)),
            value_coef=float(np.clip(self.value_coef, 0.0, 5.0)),
            updates_per_episode=int(np.clip(self.updates_per_episode, 0, 200)),
        )


def _algo_key(species: str, pop_id: int) -> str:
    return f"{ALGO_PREFIX}_{species}_{pop_id}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("PBT orchestrator for 10 independent A2C populations")
    parser.add_argument("--env-config-file", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/pbt_orchestrator_a2c"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--n-populations", type=int, default=None)

    parser.add_argument("--pbt-interval", type=int, default=20)
    parser.add_argument("--elite-fraction", type=float, default=0.4)

    parser.add_argument("--init-lr-log-mean", type=float, default=math.log(3e-4))
    parser.add_argument("--init-lr-log-std", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--init-entropy-mean", type=float, default=0.01)
    parser.add_argument("--init-entropy-std", type=float, default=0.01)
    parser.add_argument("--init-value-coef-mean", type=float, default=0.5)
    parser.add_argument("--init-value-coef-std", type=float, default=0.3)
    parser.add_argument("--init-updates-per-episode", type=int, default=5)

    parser.add_argument("--mutate-lr-sigma", type=float, default=0.25)
    parser.add_argument("--mutate-entropy-sigma", type=float, default=0.005)
    parser.add_argument("--mutate-value-sigma", type=float, default=0.2)

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


def sample_initial_hparams(rng: np.random.Generator, args: argparse.Namespace) -> A2CHyperparams:
    lr = _lognormal(rng, args.init_lr_log_mean, args.init_lr_log_std)
    entropy = float(rng.normal(args.init_entropy_mean, args.init_entropy_std))
    value_coef = float(rng.normal(args.init_value_coef_mean, args.init_value_coef_std))
    return A2CHyperparams(
        lr=lr,
        gamma=float(args.gamma),
        entropy_coef=entropy,
        value_coef=value_coef,
        updates_per_episode=int(args.init_updates_per_episode),
    ).clamp()


def mutate_hparams(rng: np.random.Generator, base: A2CHyperparams, args: argparse.Namespace) -> A2CHyperparams:
    lr = float(base.lr * math.exp(rng.normal(0.0, args.mutate_lr_sigma)))
    entropy = float(base.entropy_coef + rng.normal(0.0, args.mutate_entropy_sigma))
    value_coef = float(base.value_coef + rng.normal(0.0, args.mutate_value_sigma))
    return A2CHyperparams(
        lr=lr,
        gamma=base.gamma,
        entropy_coef=entropy,
        value_coef=value_coef,
        updates_per_episode=base.updates_per_episode,
    ).clamp()


def _score_window(values: List[float], window: int) -> float:
    if not values:
        return float("-inf")
    if window <= 1:
        return float(values[-1])
    return float(np.mean(values[-window:]))


def _select_and_mutate(
    rng: np.random.Generator,
    keys: List[str],
    scores: Dict[str, float],
    hparams: Dict[str, A2CHyperparams],
    models: Dict[str, SimpleA2C],
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
        new_hp = mutate_hparams(rng, hparams[donor], args)
        models[loser].import_state(models[donor].export_state())
        models[loser].gamma = float(new_hp.gamma)
        models[loser].entropy_coef = float(new_hp.entropy_coef)
        models[loser].value_coef = float(new_hp.value_coef)
        models[loser].set_lr(float(new_hp.lr))
        hparams[loser] = new_hp
        if writer is not None:
            writer.add_scalar(f"pbt/{loser}/lr", new_hp.lr, episode)
            writer.add_scalar(f"pbt/{loser}/entropy_coef", new_hp.entropy_coef, episode)
            writer.add_scalar(f"pbt/{loser}/value_coef", new_hp.value_coef, episode)


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

    env_probe = PredPreyGrass(env_config)
    obs_space = next(iter(env_probe.observation_spaces.values()))
    act_space = next(iter(env_probe.action_spaces.values()))
    n_pops = int(getattr(env_probe, "n_populations", env_config.get("n_populations", 5)))
    if hasattr(env_probe, "close"):
        env_probe.close()
    if args.n_populations is not None:
        n_pops = int(args.n_populations)
        env_config["n_populations"] = n_pops

    act_low = np.asarray(act_space.low, dtype=np.float32)
    act_high = np.asarray(act_space.high, dtype=np.float32)
    obs_dim = int(np.prod(obs_space.shape))
    act_dim = int(np.prod(act_space.shape))

    predator_keys = [_algo_key("predator", i) for i in range(n_pops)]
    prey_keys = [_algo_key("prey", i) for i in range(n_pops)]
    keys = predator_keys + prey_keys

    hparam_map: Dict[str, A2CHyperparams] = {k: sample_initial_hparams(rng, args) for k in keys}
    models: Dict[str, SimpleA2C] = {}
    returns_hist: Dict[str, List[float]] = {k: [] for k in keys}
    best_seen: Dict[str, float] = {k: float("-inf") for k in keys}

    for key in keys:
        hp = hparam_map[key]
        models[key] = SimpleA2C(
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_low=act_low,
            act_high=act_high,
            lr=hp.lr,
            gamma=hp.gamma,
            entropy_coef=hp.entropy_coef,
            value_coef=hp.value_coef,
        )

    writer = None
    if not args.no_tensorboard and TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=str(log_dir))

    env = PredPreyGrass(env_config)

    progress = None
    if not args.no_progress:
        progress = ProgressBar(int(args.num_episodes), label="[PBT-A2C] ", stream=sys.stderr)
        progress.update(0)

    for ep in range(int(args.num_episodes)):
        obs, infos = env.reset(seed=int(args.seed) + ep)
        traj: Dict[str, Dict[str, List[dict]]] = {k: {} for k in keys}
        ep_ret_by_agent: Dict[str, float] = {}
        key_by_agent: Dict[str, str] = {}

        for t in range(int(args.max_steps)):
            groups: Dict[str, List[str]] = {}
            for aid in obs.keys():
                info = infos.get(aid, {}) or {}
                species = info.get("species") or ("predator" if "predator" in aid else "prey")
                pop_id = info.get("population_id")
                if pop_id is None:
                    pop_id = getattr(env, "agent_population_id", {}).get(aid, 0)
                key = _algo_key(species, int(pop_id))
                key_by_agent[aid] = key
                groups.setdefault(key, []).append(aid)

            actions: Dict[str, np.ndarray] = {}
            extra_cache: Dict[str, dict] = {}
            for key, aids in groups.items():
                obs_batch = np.stack([obs[aid] for aid in aids]).astype(np.float32)
                act_b, logp_b, vf_b, pre_b = models[key].act_batch(obs_batch, deterministic=False)
                for i, aid in enumerate(aids):
                    actions[aid] = act_b[i]
                    extra_cache[aid] = {
                        "pre_tanh": pre_b[i],
                        "logp": float(logp_b[i]),
                        "vf": float(vf_b[i]),
                    }

            next_obs, rewards, dones, truncs, infos = env.step(actions)
            for aid, rew in rewards.items():
                if aid not in obs:
                    continue
                key = key_by_agent.get(aid)
                if not key:
                    continue
                done = bool(dones.get(aid, False) or truncs.get(aid, False))
                ep_ret_by_agent[aid] = ep_ret_by_agent.get(aid, 0.0) + float(rew)
                steps = traj[key].setdefault(aid, [])
                steps.append(
                    {
                        "obs": obs[aid],
                        "pre_tanh_actions": extra_cache[aid]["pre_tanh"],
                        "rewards": float(rew),
                        "dones": float(done),
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

        # Train per key (on-policy).
        for key in keys:
            hp = hparam_map[key]
            model = models[key]
            updates = int(hp.updates_per_episode)
            if updates <= 0:
                continue
            agent_traj = traj[key]
            if not agent_traj:
                continue
            # Batch all agents' trajectories together (much faster than per-agent updates).
            # Important: force terminal at each agent's last step so returns don't "leak"
            # across concatenated agent sequences.
            flat_obs: List[np.ndarray] = []
            flat_pre: List[np.ndarray] = []
            flat_rew: List[float] = []
            flat_done: List[float] = []
            for _aid, steps in agent_traj.items():
                if not steps:
                    continue
                for s in steps:
                    flat_obs.append(s["obs"])
                    flat_pre.append(s["pre_tanh_actions"])
                    flat_rew.append(float(s["rewards"]))
                    flat_done.append(float(s["dones"]))
                flat_done[-1] = 1.0
            if not flat_obs:
                continue
            mb = {
                "obs": np.stack(flat_obs).astype(np.float32),
                "pre_tanh_actions": np.stack(flat_pre).astype(np.float32),
                "rewards": np.array(flat_rew, dtype=np.float32),
                "dones": np.array(flat_done, dtype=np.float32),
            }

            last_stat = None
            for _ in range(updates):
                last_stat = model.train_batch(mb)
            # Log last batch stat is fine; this is a lightweight tuner.
            if writer is not None and last_stat is not None:
                writer.add_scalar(f"loss/{key}/entropy", float(last_stat["entropy"]), ep)
                writer.add_scalar(f"loss/{key}/policy_loss", float(last_stat["policy_loss"]), ep)
                writer.add_scalar(f"loss/{key}/value_loss", float(last_stat["value_loss"]), ep)

        # Score.
        ep_ret_by_key: Dict[str, List[float]] = {k: [] for k in keys}
        for aid, ret in ep_ret_by_agent.items():
            key = key_by_agent.get(aid)
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

        if writer is not None:
            writer.add_scalar("episode_len", t + 1, ep)

        # PBT step.
        if args.pbt_interval > 0 and (ep + 1) % int(args.pbt_interval) == 0:
            window = int(args.pbt_interval)
            scores = {k: _score_window(returns_hist[k], window) for k in keys}
            _select_and_mutate(rng, predator_keys, scores, hparam_map, models, args, writer=writer, episode=ep)
            _select_and_mutate(rng, prey_keys, scores, hparam_map, models, args, writer=writer, episode=ep)
            if writer is not None:
                for k in keys:
                    hp = hparam_map[k]
                    writer.add_scalar(f"hparams/{k}/lr", hp.lr, ep)
                    writer.add_scalar(f"hparams/{k}/entropy_coef", hp.entropy_coef, ep)
                    writer.add_scalar(f"hparams/{k}/value_coef", hp.value_coef, ep)

        if (ep + 1) % max(1, int(args.pbt_interval)) == 0:
            print(f"[PBT-A2C] episode={ep} done (len={t+1})")
        if progress is not None:
            progress.update(
                float(ep + 1),
                detail=f"ep={ep+1}/{int(args.num_episodes)} last_len={t+1}/{int(args.max_steps)}",
            )

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
    print("[PBT-A2C] best predator:", best_pred, hparam_map[best_pred])
    print("[PBT-A2C] best prey:", best_prey, hparam_map[best_prey])
    print("[PBT-A2C] summary written to:", (log_dir / "pbt_summary.json"))

    if args.save_best:
        ckpt_dir = log_dir / "checkpoints"
        torch.save(
            {"hparams": asdict(hparam_map[best_pred]), "state": models[best_pred].export_state()},
            ckpt_dir / f"{best_pred}.pt",
        )
        torch.save(
            {"hparams": asdict(hparam_map[best_prey]), "state": models[best_prey].export_state()},
            ckpt_dir / f"{best_prey}.pt",
        )

    if writer is not None:
        writer.close()
    if progress is not None:
        progress.close()
    env.close()


if __name__ == "__main__":
    main()
