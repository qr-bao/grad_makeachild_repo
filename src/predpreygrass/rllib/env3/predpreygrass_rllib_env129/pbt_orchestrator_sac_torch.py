"""
Population-based hyperparameter optimisation inside a single coexisting multi-agent run (Torch SAC).

Goal:
  - 5 predator populations + 5 prey populations (n_populations in env config)
  - All populations use SAC, but each population has its own independent SAC model
    (weights + optimiser + hyperparameters + replay buffer).
  - All populations act and learn simultaneously in the same environment.
  - Periodically apply PBT: copy weights+hyperparams from elites and mutate.

This is a lightweight, pure PyTorch SAC implementation designed for fast iteration and
to avoid RLlib/Ray overhead in this repo's multi-population setting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except Exception:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

from predpreygrass.rllib.env3.predpreygrass_rllib_env129.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.utils.progress import ProgressBar


ALGO_PREFIX = "sac"


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256), act_last=False):
        super().__init__()
        layers = []
        last = int(in_dim)
        for h in hidden:
            layers.append(nn.Linear(last, int(h)))
            layers.append(nn.ReLU())
            last = int(h)
        layers.append(nn.Linear(last, int(out_dim)))
        if act_last:
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SACActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.mean = MLP(obs_dim, act_dim, hidden=hidden, act_last=False)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(obs)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std


class SACCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.q = MLP(obs_dim + act_dim, 1, hidden=hidden, act_last=False)

    def forward(self, obs: torch.Tensor, act_raw: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act_raw], dim=-1)
        return self.q(x).squeeze(-1)


class SimpleSAC:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_low,
        act_high,
        *,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        gamma: float,
        tau: float,
        init_alpha: float,
        target_entropy: float,
        device: str = "cpu",
    ):
        self.device = device
        self.act_low = torch.tensor(act_low, device=device, dtype=torch.float32)
        self.act_high = torch.tensor(act_high, device=device, dtype=torch.float32)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        self.actor = SACActor(obs_dim, act_dim, hidden=(256, 256)).to(device)
        self.q1 = SACCritic(obs_dim, act_dim, hidden=(256, 256)).to(device)
        self.q2 = SACCritic(obs_dim, act_dim, hidden=(256, 256)).to(device)
        self.q1_t = SACCritic(obs_dim, act_dim, hidden=(256, 256)).to(device)
        self.q2_t = SACCritic(obs_dim, act_dim, hidden=(256, 256)).to(device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        init_alpha = float(max(1e-6, init_alpha))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha), device=device, dtype=torch.float32))
        self.target_entropy = float(target_entropy)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=float(critic_lr))
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=float(critic_lr))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=float(alpha_lr))

        self.gamma = float(gamma)
        self.tau = float(tau)
        self._logp_eps = 1e-6
        self.total_updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def set_lrs(self, *, actor_lr: float, critic_lr: float, alpha_lr: float) -> None:
        for g in self.actor_opt.param_groups:
            g["lr"] = float(actor_lr)
        for g in self.q1_opt.param_groups:
            g["lr"] = float(critic_lr)
        for g in self.q2_opt.param_groups:
            g["lr"] = float(critic_lr)
        for g in self.alpha_opt.param_groups:
            g["lr"] = float(alpha_lr)

    def _squash_and_scale(self, pre_tanh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        act_raw = torch.tanh(pre_tanh)
        act = act_raw * (self.act_high - self.act_low) / 2 + (self.act_high + self.act_low) / 2
        act = torch.max(torch.min(act, self.act_high), self.act_low)
        return act_raw, act

    def _squash_logp(self, dist: D.Normal, pre_tanh: torch.Tensor) -> torch.Tensor:
        act_raw = torch.tanh(pre_tanh)
        logp = dist.log_prob(pre_tanh) - torch.log(1.0 - act_raw.pow(2) + self._logp_eps)
        return logp.sum(-1)

    def sample_action_batch(self, obs_batch_np, *, deterministic: bool = False):
        with torch.no_grad():
            obs = torch.as_tensor(obs_batch_np, device=self.device, dtype=torch.float32)
            mean, log_std = self.actor(obs)
            log_std = torch.clamp(log_std, -20.0, 2.0)
            std = log_std.exp()
            dist = D.Normal(mean, std)
            pre_tanh = mean if deterministic else dist.rsample()
            logp = self._squash_logp(dist, pre_tanh)
            act_raw, act = self._squash_and_scale(pre_tanh)
            return (
                act.detach().cpu().numpy(),
                act_raw.detach().cpu().numpy(),
                logp.detach().cpu().numpy(),
            )

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1 - self.tau)
                tp.data.add_(self.tau * sp.data)

    def train_step(self, batch: dict) -> dict:
        self.total_updates += 1
        obs = torch.as_tensor(batch["obs"], device=self.device, dtype=torch.float32)
        act_raw = torch.as_tensor(batch["act_raw"], device=self.device, dtype=torch.float32)
        rews = torch.as_tensor(batch["rewards"], device=self.device, dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], device=self.device, dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], device=self.device, dtype=torch.float32)

        # --- Q targets
        with torch.no_grad():
            mean, log_std = self.actor(next_obs)
            log_std = torch.clamp(log_std, -20.0, 2.0)
            std = log_std.exp()
            dist = D.Normal(mean, std)
            pre_tanh = dist.rsample()
            next_logp = self._squash_logp(dist, pre_tanh)
            next_act_raw, _next_act = self._squash_and_scale(pre_tanh)
            target_q = torch.min(self.q1_t(next_obs, next_act_raw), self.q2_t(next_obs, next_act_raw))
            target_v = target_q - self.alpha.detach() * next_logp
            y = rews + self.gamma * (1.0 - dones) * target_v

        q1 = self.q1(obs, act_raw)
        q2 = self.q2(obs, act_raw)
        q1_loss = nn.functional.mse_loss(q1, y)
        q2_loss = nn.functional.mse_loss(q2, y)

        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.q1_opt.step()
        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.q2_opt.step()

        # --- Actor + alpha
        mean, log_std = self.actor(obs)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        std = log_std.exp()
        dist = D.Normal(mean, std)
        pre_tanh = dist.rsample()
        logp = self._squash_logp(dist, pre_tanh)
        new_act_raw, _new_act = self._squash_and_scale(pre_tanh)
        q_new = torch.min(self.q1(obs, new_act_raw), self.q2(obs, new_act_raw))
        actor_loss = (self.alpha.detach() * logp - q_new).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + float(self.target_entropy))).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.q1_t, self.q1)
        self._soft_update(self.q2_t, self.q2)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.detach().cpu().item()),
            "mean_logp": float(logp.detach().mean().cpu().item()),
        }

    def export_state(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_t": self.q1_t.state_dict(),
            "q2_t": self.q2_t.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_opt": self.actor_opt.state_dict(),
            "q1_opt": self.q1_opt.state_dict(),
            "q2_opt": self.q2_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "gamma": self.gamma,
            "tau": self.tau,
            "target_entropy": self.target_entropy,
            "act_low": self.act_low.detach().cpu().numpy(),
            "act_high": self.act_high.detach().cpu().numpy(),
            "total_updates": self.total_updates,
        }

    def import_state(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self.q1_t.load_state_dict(state["q1_t"])
        self.q2_t.load_state_dict(state["q2_t"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        try:
            self.actor_opt.load_state_dict(state["actor_opt"])
            self.q1_opt.load_state_dict(state["q1_opt"])
            self.q2_opt.load_state_dict(state["q2_opt"])
            self.alpha_opt.load_state_dict(state["alpha_opt"])
        except Exception:
            pass
        self.total_updates = int(state.get("total_updates", 0))


@dataclass(frozen=True)
class SACHyperparams:
    actor_lr: float
    critic_lr: float
    alpha_lr: float
    tau: float
    gamma: float
    init_alpha: float
    target_entropy: float
    batch_size: int
    warmup_steps: int
    updates_per_episode: int

    def clamp(self) -> "SACHyperparams":
        def _clip(v, lo, hi):
            return float(np.clip(v, lo, hi))

        return SACHyperparams(
            actor_lr=_clip(self.actor_lr, 1e-6, 5e-3),
            critic_lr=_clip(self.critic_lr, 1e-6, 5e-3),
            alpha_lr=_clip(self.alpha_lr, 1e-6, 5e-3),
            tau=_clip(self.tau, 1e-4, 0.02),
            gamma=_clip(self.gamma, 0.8, 0.999),
            init_alpha=_clip(self.init_alpha, 1e-6, 2.0),
            target_entropy=_clip(self.target_entropy, -20.0, 0.0),
            batch_size=int(np.clip(self.batch_size, 32, 4096)),
            warmup_steps=int(np.clip(self.warmup_steps, 0, 200000)),
            updates_per_episode=int(np.clip(self.updates_per_episode, 0, 500)),
        )


def _algo_key(species: str, pop_id: int) -> str:
    return f"{ALGO_PREFIX}_{species}_{pop_id}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("PBT orchestrator for 10 independent Torch SAC populations")
    p.add_argument("--env-config-file", type=Path, default=None)
    p.add_argument("--log-dir", type=Path, default=Path("logs/pbt_orchestrator_sac_torch"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-episodes", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--n-populations", type=int, default=None)

    p.add_argument("--pbt-interval", type=int, default=20)
    p.add_argument("--elite-fraction", type=float, default=0.4)
    p.add_argument("--buffer-capacity", type=int, default=50000)

    # Init distributions.
    p.add_argument("--init-lr-log-mean", type=float, default=math.log(3e-4))
    p.add_argument("--init-lr-log-std", type=float, default=0.7)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--init-tau-mean", type=float, default=0.005)
    p.add_argument("--init-tau-std", type=float, default=0.002)
    p.add_argument("--init-alpha-mean", type=float, default=0.2)
    p.add_argument("--init-alpha-std", type=float, default=0.2)
    p.add_argument("--init-target-entropy", type=float, default=None)
    p.add_argument("--init-batch-size", type=int, default=256)
    p.add_argument("--init-warmup", type=int, default=2000)
    p.add_argument("--init-updates-per-episode", type=int, default=1)

    # Mutation magnitudes.
    p.add_argument("--mutate-lr-sigma", type=float, default=0.25)
    p.add_argument("--mutate-tau-sigma", type=float, default=0.001)
    p.add_argument("--mutate-alpha-sigma", type=float, default=0.05)
    p.add_argument("--mutate-target-entropy-sigma", type=float, default=0.5)
    p.add_argument("--mutate-batch-scale", type=float, default=0.25)
    p.add_argument("--mutate-warmup-scale", type=float, default=0.25)
    p.add_argument("--mutate-updates-scale", type=float, default=0.5)

    p.add_argument("--torch-num-threads", type=int, default=1)
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--progress-every-steps", type=int, default=50)
    p.add_argument("--save-best", action="store_true")
    return p


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


def sample_initial_hparams(rng: np.random.Generator, args: argparse.Namespace, *, act_dim: int) -> SACHyperparams:
    lr = _lognormal(rng, args.init_lr_log_mean, args.init_lr_log_std)
    tau = float(max(1e-4, rng.normal(args.init_tau_mean, args.init_tau_std)))
    init_alpha = float(max(1e-6, rng.normal(args.init_alpha_mean, args.init_alpha_std)))
    if args.init_target_entropy is None:
        target_entropy = float(-act_dim)
    else:
        target_entropy = float(args.init_target_entropy)
    return SACHyperparams(
        actor_lr=lr,
        critic_lr=lr,
        alpha_lr=lr,
        tau=tau,
        gamma=float(args.gamma),
        init_alpha=init_alpha,
        target_entropy=target_entropy,
        batch_size=int(args.init_batch_size),
        warmup_steps=int(args.init_warmup),
        updates_per_episode=int(args.init_updates_per_episode),
    ).clamp()


def mutate_hparams(rng: np.random.Generator, base: SACHyperparams, args: argparse.Namespace) -> SACHyperparams:
    lr = float(base.actor_lr * math.exp(rng.normal(0.0, args.mutate_lr_sigma)))
    tau = float(base.tau + rng.normal(0.0, args.mutate_tau_sigma))
    init_alpha = float(base.init_alpha + rng.normal(0.0, args.mutate_alpha_sigma))
    target_entropy = float(base.target_entropy + rng.normal(0.0, args.mutate_target_entropy_sigma))
    batch_scale = float(math.exp(rng.normal(0.0, args.mutate_batch_scale)))
    warmup_scale = float(math.exp(rng.normal(0.0, args.mutate_warmup_scale)))
    updates_scale = float(math.exp(rng.normal(0.0, args.mutate_updates_scale)))
    return SACHyperparams(
        actor_lr=lr,
        critic_lr=lr,
        alpha_lr=lr,
        tau=tau,
        gamma=base.gamma,
        init_alpha=init_alpha,
        target_entropy=target_entropy,
        batch_size=int(max(32, round(base.batch_size * batch_scale))),
        warmup_steps=int(max(0, round(base.warmup_steps * warmup_scale))),
        updates_per_episode=int(max(0, round(base.updates_per_episode * updates_scale))),
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
    hparams: Dict[str, SACHyperparams],
    models: Dict[str, SimpleSAC],
    replay: Dict[str, List[dict]],
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
        models[loser].tau = float(new_hp.tau)
        models[loser].gamma = float(new_hp.gamma)
        models[loser].target_entropy = float(new_hp.target_entropy)
        with torch.no_grad():
            models[loser].log_alpha.copy_(
                torch.tensor(math.log(float(max(1e-6, new_hp.init_alpha))), device=models[loser].device)
            )
        models[loser].set_lrs(
            actor_lr=new_hp.actor_lr, critic_lr=new_hp.critic_lr, alpha_lr=new_hp.alpha_lr
        )
        hparams[loser] = new_hp
        replay[loser].clear()
        if writer is not None:
            writer.add_scalar(f"pbt/{loser}/lr", new_hp.actor_lr, episode)
            writer.add_scalar(f"pbt/{loser}/tau", new_hp.tau, episode)
            writer.add_scalar(f"pbt/{loser}/init_alpha", new_hp.init_alpha, episode)
            writer.add_scalar(f"pbt/{loser}/target_entropy", new_hp.target_entropy, episode)
            writer.add_scalar(f"pbt/{loser}/batch_size", new_hp.batch_size, episode)
            writer.add_scalar(f"pbt/{loser}/warmup_steps", new_hp.warmup_steps, episode)
            writer.add_scalar(f"pbt/{loser}/updates_per_episode", new_hp.updates_per_episode, episode)


def main() -> None:
    args = build_parser().parse_args()
    log_dir = args.log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    if int(args.torch_num_threads) > 0:
        torch.set_num_threads(int(args.torch_num_threads))

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(int(args.seed))

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

    hparam_map: Dict[str, SACHyperparams] = {k: sample_initial_hparams(rng, args, act_dim=act_dim) for k in keys}
    models: Dict[str, SimpleSAC] = {}
    replay: Dict[str, List[dict]] = {k: [] for k in keys}
    returns_hist: Dict[str, List[float]] = {k: [] for k in keys}
    best_seen: Dict[str, float] = {k: float("-inf") for k in keys}

    for key in keys:
        hp = hparam_map[key]
        models[key] = SimpleSAC(
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_low=act_low,
            act_high=act_high,
            actor_lr=hp.actor_lr,
            critic_lr=hp.critic_lr,
            alpha_lr=hp.alpha_lr,
            gamma=hp.gamma,
            tau=hp.tau,
            init_alpha=hp.init_alpha,
            target_entropy=hp.target_entropy,
            device="cpu",
        )

    writer = None
    if not args.no_tensorboard and TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=str(log_dir))

    env = PredPreyGrass(env_config)

    progress = None
    if not args.no_progress:
        progress = ProgressBar(int(args.num_episodes), label="[PBT-SAC-TORCH] ", stream=sys.stderr)
        progress.update(0)

    for ep in range(int(args.num_episodes)):
        obs, infos = env.reset(seed=int(args.seed) + ep)
        ep_ret_by_key: Dict[str, List[float]] = {k: [] for k in keys}
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
            raw_actions: Dict[str, np.ndarray] = {}
            for key, aids in groups.items():
                obs_batch = np.stack([obs[aid] for aid in aids]).astype(np.float32)
                act_b, act_raw_b, _logp_b = models[key].sample_action_batch(obs_batch, deterministic=False)
                for i, aid in enumerate(aids):
                    actions[aid] = act_b[i]
                    raw_actions[aid] = act_raw_b[i]

            next_obs, rewards, dones, truncs, infos = env.step(actions)
            for aid, rew in rewards.items():
                if aid not in obs:
                    continue
                key = key_by_agent.get(aid)
                if not key:
                    continue
                done = bool(dones.get(aid, False) or truncs.get(aid, False))
                ep_ret_by_agent[aid] = ep_ret_by_agent.get(aid, 0.0) + float(rew)
                replay[key].append(
                    {
                        "obs": obs[aid],
                        "act_raw": raw_actions[aid],
                        "rewards": float(rew),
                        "next_obs": next_obs.get(aid, obs[aid]),
                        "dones": float(done),
                    }
                )
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

        # SAC updates.
        for key in keys:
            hp = hparam_map[key]
            buf = replay[key]
            if len(buf) < max(int(hp.warmup_steps), int(hp.batch_size)):
                continue
            updates = int(hp.updates_per_episode)
            if updates <= 0:
                continue
            loss_stats = []
            for _u in range(updates):
                batch_size = int(hp.batch_size)
                idx = rng.choice(len(buf), size=batch_size, replace=len(buf) < batch_size)
                sample = [buf[i] for i in idx]
                mb = {
                    "obs": np.stack([s["obs"] for s in sample]).astype(np.float32),
                    "act_raw": np.stack([s["act_raw"] for s in sample]).astype(np.float32),
                    "rewards": np.array([s["rewards"] for s in sample], dtype=np.float32),
                    "next_obs": np.stack([s["next_obs"] for s in sample]).astype(np.float32),
                    "dones": np.array([s["dones"] for s in sample], dtype=np.float32),
                }
                stat = models[key].train_step(mb)
                loss_stats.append(stat)
            if writer is not None and loss_stats:
                writer.add_scalar(f"loss/{key}/q1_loss", float(np.mean([s["q1_loss"] for s in loss_stats])), ep)
                writer.add_scalar(f"loss/{key}/q2_loss", float(np.mean([s["q2_loss"] for s in loss_stats])), ep)
                writer.add_scalar(
                    f"loss/{key}/actor_loss", float(np.mean([s["actor_loss"] for s in loss_stats])), ep
                )
                writer.add_scalar(
                    f"loss/{key}/alpha_loss", float(np.mean([s["alpha_loss"] for s in loss_stats])), ep
                )
                writer.add_scalar(f"alpha/{key}", float(loss_stats[-1]["alpha"]), ep)
                writer.add_scalar(f"mean_logp/{key}", float(loss_stats[-1]["mean_logp"]), ep)

        if writer is not None:
            writer.add_scalar("episode_len", t + 1, ep)

        # PBT step.
        if args.pbt_interval > 0 and (ep + 1) % int(args.pbt_interval) == 0:
            window = int(args.pbt_interval)
            scores = {k: _score_window(returns_hist[k], window) for k in keys}
            _select_and_mutate(
                rng,
                predator_keys,
                scores,
                hparam_map,
                models,
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
                models,
                replay,
                args,
                writer=writer,
                episode=ep,
            )

        if (ep + 1) % max(1, int(args.pbt_interval)) == 0:
            print(f"[PBT-SAC-TORCH] episode={ep} done (len={t+1})")
        if progress is not None:
            progress.update(float(ep + 1), detail=f"ep={ep+1}/{int(args.num_episodes)} last_len={t+1}/{int(args.max_steps)}")

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
    print("[PBT-SAC-TORCH] best predator:", best_pred, hparam_map[best_pred])
    print("[PBT-SAC-TORCH] best prey:", best_prey, hparam_map[best_prey])
    print("[PBT-SAC-TORCH] summary written to:", (log_dir / "pbt_summary.json"))

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

