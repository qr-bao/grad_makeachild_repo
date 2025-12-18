"""
Population-based hyperparameter optimisation inside a single coexisting multi-agent run (TD3).

Goal:
  - 5 predator populations + 5 prey populations (n_populations in env config)
  - All populations use TD3, but each population has its own independent TD3 model
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
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

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


ALGO_PREFIX = "td3"


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


class SimpleTD3:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_low,
        act_high,
        *,
        lr: float,
        gamma: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        policy_delay: int,
        device: str = "cpu",
    ):
        self.device = device
        self.act_low = torch.tensor(act_low, device=device, dtype=torch.float32)
        self.act_high = torch.tensor(act_high, device=device, dtype=torch.float32)
        self.actor = MLP(obs_dim, act_dim).to(device)
        self.actor_target = MLP(obs_dim, act_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.q1 = MLP(obs_dim + act_dim, 1, act_last=False).to(device)
        self.q2 = MLP(obs_dim + act_dim, 1, act_last=False).to(device)
        self.q1_target = MLP(obs_dim + act_dim, 1, act_last=False).to(device)
        self.q2_target = MLP(obs_dim + act_dim, 1, act_last=False).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=lr)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_delay = int(max(1, policy_delay))
        self.total_updates = 0

    def compute_action(self, obs_np, noise_std: float = 0.1):
        with torch.no_grad():
            o = torch.as_tensor(obs_np, device=self.device, dtype=torch.float32)
            a = self.actor(o)
            a = a * (self.act_high - self.act_low) / 2 + (self.act_high + self.act_low) / 2
            if noise_std > 0:
                a = a + torch.randn_like(a) * float(noise_std)
            a = torch.max(torch.min(a, self.act_high), self.act_low)
            return a.cpu().numpy()

    def compute_action_batch(self, obs_batch_np, noise_std: float = 0.1):
        with torch.no_grad():
            o = torch.as_tensor(obs_batch_np, device=self.device, dtype=torch.float32)
            a = self.actor(o)
            a = a * (self.act_high - self.act_low) / 2 + (self.act_high + self.act_low) / 2
            if noise_std > 0:
                a = a + torch.randn_like(a) * float(noise_std)
            a = torch.max(torch.min(a, self.act_high), self.act_low)
            return a.cpu().numpy()

    def train_step(self, batch: dict) -> dict:
        self.total_updates += 1
        obs = torch.as_tensor(batch["obs"], device=self.device, dtype=torch.float32)
        acts = torch.as_tensor(batch["actions"], device=self.device, dtype=torch.float32)
        rews = torch.as_tensor(batch["rewards"], device=self.device, dtype=torch.float32).unsqueeze(-1)
        next_obs = torch.as_tensor(batch["next_obs"], device=self.device, dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], device=self.device, dtype=torch.float32).unsqueeze(-1)

        with torch.no_grad():
            noise = (torch.randn_like(acts) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = self.actor_target(next_obs)
            next_action = next_action * (self.act_high - self.act_low) / 2 + (self.act_high + self.act_low) / 2
            next_action = (next_action + noise).clamp(self.act_low, self.act_high)
            inp_next = torch.cat([next_obs, next_action], dim=-1)
            target_q1 = self.q1_target(inp_next)
            target_q2 = self.q2_target(inp_next)
            target_q = torch.min(target_q1, target_q2)
            target = rews + self.gamma * (1 - dones) * target_q

        inp = torch.cat([obs, acts], dim=-1)
        current_q1 = self.q1(inp)
        current_q2 = self.q2(inp)
        q1_loss = nn.functional.mse_loss(current_q1, target)
        q2_loss = nn.functional.mse_loss(current_q2, target)
        self.q1_opt.zero_grad()
        self.q2_opt.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q1_opt.step()
        self.q2_opt.step()

        actor_loss_val = None
        if self.total_updates % self.policy_delay == 0:
            actor_actions = self.actor(obs)
            actor_actions = actor_actions * (self.act_high - self.act_low) / 2 + (self.act_high + self.act_low) / 2
            actor_inp = torch.cat([obs, actor_actions], dim=-1)
            actor_loss = -self.q1(actor_inp).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.q1_target, self.q1)
            self._soft_update(self.q2_target, self.q2)
            actor_loss_val = float(actor_loss.item())

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": actor_loss_val,
        }

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1 - self.tau)
                tp.data.add_(self.tau * sp.data)

    def export_state(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "q1_opt": self.q1_opt.state_dict(),
            "q2_opt": self.q2_opt.state_dict(),
            "gamma": self.gamma,
            "tau": self.tau,
            "policy_noise": self.policy_noise,
            "noise_clip": self.noise_clip,
            "policy_delay": self.policy_delay,
            "total_updates": self.total_updates,
            "act_low": self.act_low.detach().cpu().numpy(),
            "act_high": self.act_high.detach().cpu().numpy(),
        }

    def import_state(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self.q1_target.load_state_dict(state["q1_target"])
        self.q2_target.load_state_dict(state["q2_target"])
        try:
            self.actor_opt.load_state_dict(state["actor_opt"])
            self.q1_opt.load_state_dict(state["q1_opt"])
            self.q2_opt.load_state_dict(state["q2_opt"])
        except Exception:
            pass
        self.total_updates = int(state.get("total_updates", 0))


@dataclass(frozen=True)
class TD3Hyperparams:
    lr: float
    tau: float
    gamma: float
    policy_noise: float
    noise_clip: float
    policy_delay: int
    batch_size: int
    warmup_steps: int
    updates_per_episode: int

    def clamp(self) -> "TD3Hyperparams":
        def _clip(v, lo, hi):
            return float(np.clip(v, lo, hi))

        return TD3Hyperparams(
            lr=_clip(self.lr, 1e-6, 5e-3),
            tau=_clip(self.tau, 1e-4, 0.02),
            gamma=_clip(self.gamma, 0.8, 0.999),
            policy_noise=_clip(self.policy_noise, 0.0, 1.0),
            noise_clip=_clip(self.noise_clip, 0.0, 1.0),
            policy_delay=int(np.clip(self.policy_delay, 1, 10)),
            batch_size=int(np.clip(self.batch_size, 16, 2048)),
            warmup_steps=int(np.clip(self.warmup_steps, 0, 200000)),
            updates_per_episode=int(np.clip(self.updates_per_episode, 0, 500)),
        )


def _algo_key(species: str, pop_id: int) -> str:
    return f"{ALGO_PREFIX}_{species}_{pop_id}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("PBT orchestrator for 10 independent TD3 populations")
    parser.add_argument("--env-config-file", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/pbt_orchestrator_td3"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--n-populations", type=int, default=None)

    parser.add_argument("--pbt-interval", type=int, default=20)
    parser.add_argument("--elite-fraction", type=float, default=0.4)
    parser.add_argument("--buffer-capacity", type=int, default=50000)

    # Init distributions.
    parser.add_argument("--init-lr-log-mean", type=float, default=math.log(3e-4))
    parser.add_argument("--init-lr-log-std", type=float, default=0.7)
    parser.add_argument("--init-tau-mean", type=float, default=0.005)
    parser.add_argument("--init-tau-std", type=float, default=0.002)
    parser.add_argument("--init-policy-noise-mean", type=float, default=0.1)
    parser.add_argument("--init-policy-noise-std", type=float, default=0.05)
    parser.add_argument("--init-noise-clip-mean", type=float, default=0.2)
    parser.add_argument("--init-noise-clip-std", type=float, default=0.05)
    parser.add_argument("--init-policy-delay", type=int, default=2)
    parser.add_argument("--init-batch-size", type=int, default=256)
    parser.add_argument("--init-warmup", type=int, default=2000)
    parser.add_argument("--init-updates-per-episode", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)

    # Mutation magnitudes.
    parser.add_argument("--mutate-lr-sigma", type=float, default=0.25)
    parser.add_argument("--mutate-tau-sigma", type=float, default=0.001)
    parser.add_argument("--mutate-noise-sigma", type=float, default=0.03)
    parser.add_argument("--mutate-clip-sigma", type=float, default=0.03)
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


def sample_initial_hparams(rng: np.random.Generator, args: argparse.Namespace) -> TD3Hyperparams:
    lr = _lognormal(rng, args.init_lr_log_mean, args.init_lr_log_std)
    tau = float(max(1e-4, rng.normal(args.init_tau_mean, args.init_tau_std)))
    policy_noise = float(max(0.0, rng.normal(args.init_policy_noise_mean, args.init_policy_noise_std)))
    noise_clip = float(max(0.0, rng.normal(args.init_noise_clip_mean, args.init_noise_clip_std)))
    return TD3Hyperparams(
        lr=lr,
        tau=tau,
        gamma=float(args.gamma),
        policy_noise=policy_noise,
        noise_clip=noise_clip,
        policy_delay=int(args.init_policy_delay),
        batch_size=int(args.init_batch_size),
        warmup_steps=int(args.init_warmup),
        updates_per_episode=int(args.init_updates_per_episode),
    ).clamp()


def mutate_hparams(rng: np.random.Generator, base: TD3Hyperparams, args: argparse.Namespace) -> TD3Hyperparams:
    lr = float(base.lr * math.exp(rng.normal(0.0, args.mutate_lr_sigma)))
    tau = float(base.tau + rng.normal(0.0, args.mutate_tau_sigma))
    policy_noise = float(base.policy_noise + rng.normal(0.0, args.mutate_noise_sigma))
    noise_clip = float(base.noise_clip + rng.normal(0.0, args.mutate_clip_sigma))
    batch = int(round(base.batch_size * (1.0 + rng.normal(0.0, args.mutate_batch_scale)) / 32.0) * 32)
    return TD3Hyperparams(
        lr=lr,
        tau=tau,
        gamma=base.gamma,
        policy_noise=policy_noise,
        noise_clip=noise_clip,
        policy_delay=base.policy_delay,
        batch_size=batch,
        warmup_steps=base.warmup_steps,
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
    hparams: Dict[str, TD3Hyperparams],
    models: Dict[str, SimpleTD3],
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
        new_hp = mutate_hparams(rng, hparams[donor], args)
        # Copy donor weights/optimizer states, then apply new hyperparams to optimizers.
        models[loser].import_state(models[donor].export_state())
        for opt in (models[loser].actor_opt, models[loser].q1_opt, models[loser].q2_opt):
            for group in opt.param_groups:
                group["lr"] = float(new_hp.lr)
        models[loser].tau = float(new_hp.tau)
        models[loser].gamma = float(new_hp.gamma)
        models[loser].policy_noise = float(new_hp.policy_noise)
        models[loser].noise_clip = float(new_hp.noise_clip)
        models[loser].policy_delay = int(new_hp.policy_delay)
        hparams[loser] = new_hp
        buffers[loser].clear()

        if writer is not None:
            writer.add_scalar(f"pbt/{loser}/lr", new_hp.lr, episode)
            writer.add_scalar(f"pbt/{loser}/tau", new_hp.tau, episode)
            writer.add_scalar(f"pbt/{loser}/policy_noise", new_hp.policy_noise, episode)
            writer.add_scalar(f"pbt/{loser}/noise_clip", new_hp.noise_clip, episode)
            writer.add_scalar(f"pbt/{loser}/batch_size", new_hp.batch_size, episode)


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

    # TD3 requires continuous action bounds.
    act_low = np.asarray(act_space.low, dtype=np.float32)
    act_high = np.asarray(act_space.high, dtype=np.float32)
    obs_dim = int(np.prod(obs_space.shape))
    act_dim = int(np.prod(act_space.shape))

    predator_keys = [_algo_key("predator", i) for i in range(n_pops)]
    prey_keys = [_algo_key("prey", i) for i in range(n_pops)]
    keys = predator_keys + prey_keys

    hparam_map: Dict[str, TD3Hyperparams] = {k: sample_initial_hparams(rng, args) for k in keys}
    models: Dict[str, SimpleTD3] = {}
    buffers: Dict[str, List[dict]] = {k: [] for k in keys}
    returns_hist: Dict[str, List[float]] = {k: [] for k in keys}
    best_seen: Dict[str, float] = {k: float("-inf") for k in keys}

    writer = None
    if not args.no_tensorboard and TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=str(log_dir))

    env = PredPreyGrass(env_config)
    for key in keys:
        hp = hparam_map[key]
        models[key] = SimpleTD3(
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_low=act_low,
            act_high=act_high,
            lr=hp.lr,
            gamma=hp.gamma,
            tau=hp.tau,
            policy_noise=hp.policy_noise,
            noise_clip=hp.noise_clip,
            policy_delay=hp.policy_delay,
            device="cpu",
        )

    progress = None
    if not args.no_progress:
        progress = ProgressBar(int(args.num_episodes), label="[PBT-TD3] ", stream=sys.stderr)
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
            for key, aids in groups.items():
                obs_batch = np.stack([obs[aid] for aid in aids]).astype(np.float32)
                act_batch = models[key].compute_action_batch(obs_batch, noise_std=0.1)
                for i, aid in enumerate(aids):
                    actions[aid] = act_batch[i]

            next_obs, rewards, dones, truncs, infos = env.step(actions)

            for aid, rew in rewards.items():
                if aid not in obs:
                    continue
                key = key_by_agent.get(aid)
                if not key:
                    continue
                done = bool(dones.get(aid, False) or truncs.get(aid, False))
                ep_ret_by_agent[aid] = ep_ret_by_agent.get(aid, 0.0) + float(rew)
                buffers[key].append(
                    {
                        "obs": obs[aid],
                        "actions": actions[aid],
                        "rewards": float(rew),
                        "next_obs": next_obs.get(aid, obs[aid]),
                        "dones": float(done),
                    }
                )
                if len(buffers[key]) > int(args.buffer_capacity):
                    buffers[key] = buffers[key][-int(args.buffer_capacity) :]

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

        # Score per key.
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

        # Train each TD3 model.
        for key in keys:
            hp = hparam_map[key]
            buf = buffers[key]
            if len(buf) < max(int(hp.warmup_steps), int(hp.batch_size)):
                continue
            loss_stats = []
            n_updates = int(hp.updates_per_episode)
            if progress is not None and n_updates > 0:
                progress.update(float(ep + 1), detail=f"training {key} updates={n_updates} buf={len(buf)}")
            for u in range(n_updates):
                batch_size = int(hp.batch_size)
                idx = rng.choice(len(buf), size=batch_size, replace=len(buf) < batch_size)
                sample = [buf[i] for i in idx]
                mb = {
                    "obs": np.stack([s["obs"] for s in sample]).astype(np.float32),
                    "actions": np.stack([s["actions"] for s in sample]).astype(np.float32),
                    "rewards": np.array([s["rewards"] for s in sample], dtype=np.float32),
                    "next_obs": np.stack([s["next_obs"] for s in sample]).astype(np.float32),
                    "dones": np.array([s["dones"] for s in sample], dtype=np.float32),
                }
                stat = models[key].train_step(mb)
                loss_stats.append(stat)
                if progress is not None and n_updates >= 10:
                    every_u = max(1, n_updates // 3)
                    if (u + 1) % every_u == 0:
                        progress.update(float(ep + 1), detail=f"training {key} upd={u+1}/{n_updates}")
            if writer is not None and loss_stats:
                q1 = [s["q1_loss"] for s in loss_stats if s.get("q1_loss") is not None]
                q2 = [s["q2_loss"] for s in loss_stats if s.get("q2_loss") is not None]
                al = [s["actor_loss"] for s in loss_stats if s.get("actor_loss") is not None]
                if q1:
                    writer.add_scalar(f"loss/{key}/q1_loss", float(np.mean(q1)), ep)
                if q2:
                    writer.add_scalar(f"loss/{key}/q2_loss", float(np.mean(q2)), ep)
                if al:
                    writer.add_scalar(f"loss/{key}/actor_loss", float(np.mean(al)), ep)

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
                buffers,
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
                buffers,
                args,
                writer=writer,
                episode=ep,
            )
            if writer is not None:
                for k in keys:
                    hp = hparam_map[k]
                    writer.add_scalar(f"hparams/{k}/lr", hp.lr, ep)
                    writer.add_scalar(f"hparams/{k}/tau", hp.tau, ep)
                    writer.add_scalar(f"hparams/{k}/policy_noise", hp.policy_noise, ep)
                    writer.add_scalar(f"hparams/{k}/noise_clip", hp.noise_clip, ep)
                    writer.add_scalar(f"hparams/{k}/batch_size", hp.batch_size, ep)

        if (ep + 1) % max(1, int(args.pbt_interval)) == 0:
            print(f"[PBT-TD3] episode={ep} done (len={t+1})")
        if progress is not None:
            progress.update(
                float(ep + 1),
                detail=f"ep={ep+1}/{int(args.num_episodes)} last_len={t+1}/{int(args.max_steps)}",
            )

    # Final selection: best by last-window mean per species.
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
    print("[PBT-TD3] best predator:", best_pred, hparam_map[best_pred])
    print("[PBT-TD3] best prey:", best_prey, hparam_map[best_prey])
    print("[PBT-TD3] summary written to:", (log_dir / "pbt_summary.json"))

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
