"""
Population-based hyperparameter optimisation inside a single coexisting multi-agent run (Torch PPO).

Goal:
  - 5 predator populations + 5 prey populations (n_populations in env config)
  - All populations use PPO, but each population has its own independent PPO model
    (weights + optimiser + hyperparameters).
  - All populations act and learn simultaneously in the same environment.
  - Periodically apply PBT: copy weights+hyperparams from elites and mutate.

This is a lightweight, pure PyTorch PPO implementation designed for fast iteration and
to avoid RLlib/Ray overhead in this repo's multi-population setting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

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


ALGO_PREFIX = "ppo"


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256), act_last=True):
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


class SimplePPO:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_low,
        act_high,
        *,
        lr: float,
        gamma: float,
        lambda_: float,
        entropy_coef: float,
        value_coef: float,
        max_grad_norm: float,
        device: str = "cpu",
    ):
        self.device = device
        self.act_low = torch.tensor(act_low, device=device, dtype=torch.float32)
        self.act_high = torch.tensor(act_high, device=device, dtype=torch.float32)
        self.actor = MLP(obs_dim, act_dim, hidden=(256, 256), act_last=False).to(device)
        self.value = MLP(obs_dim, 1, hidden=(256, 256), act_last=False).to(device)
        self.log_std = nn.Parameter(torch.zeros(act_dim, device=device))
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.value.parameters()) + [self.log_std],
            lr=float(lr),
        )
        self.gamma = float(gamma)
        self.lambda_ = float(lambda_)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self._logp_eps = 1e-6

    def set_lr(self, lr: float) -> None:
        for group in self.opt.param_groups:
            group["lr"] = float(lr)

    def _dist(self, mean: torch.Tensor) -> D.Normal:
        std = self.log_std.exp()
        return D.Normal(mean, std)

    def _squash_and_scale(self, pre_tanh: torch.Tensor) -> torch.Tensor:
        a_raw = torch.tanh(pre_tanh)
        a = a_raw * (self.act_high - self.act_low) / 2 + (self.act_high + self.act_low) / 2
        return torch.max(torch.min(a, self.act_high), self.act_low)

    def _squash_logp(self, dist: D.Normal, pre_tanh: torch.Tensor) -> torch.Tensor:
        a_raw = torch.tanh(pre_tanh)
        logp = dist.log_prob(pre_tanh) - torch.log(1.0 - a_raw.pow(2) + self._logp_eps)
        return logp.sum(-1)

    def act_batch(self, obs_batch_np, deterministic: bool = False):
        with torch.no_grad():
            obs = torch.as_tensor(obs_batch_np, device=self.device, dtype=torch.float32)
            mean = self.actor(obs)
            dist = self._dist(mean)
            pre_tanh = mean if deterministic else dist.sample()
            action = self._squash_and_scale(pre_tanh)
            logp = self._squash_logp(dist, pre_tanh)
            value = self.value(obs).squeeze(-1)
            return (
                action.detach().cpu().numpy(),
                logp.detach().cpu().numpy(),
                value.detach().cpu().numpy(),
                pre_tanh.detach().cpu().numpy(),
            )

    def value_batch(self, obs_batch_np) -> np.ndarray:
        with torch.no_grad():
            obs = torch.as_tensor(obs_batch_np, device=self.device, dtype=torch.float32)
            v = self.value(obs).squeeze(-1)
            return v.detach().cpu().numpy()

    def evaluate_batch(self, obs_batch, pre_tanh_batch):
        obs = torch.as_tensor(obs_batch, device=self.device, dtype=torch.float32)
        pre_tanh = torch.as_tensor(pre_tanh_batch, device=self.device, dtype=torch.float32)
        mean = self.actor(obs)
        dist = self._dist(mean)
        logp = self._squash_logp(dist, pre_tanh)
        entropy = dist.entropy().sum(-1)
        value = self.value(obs).squeeze(-1)
        return logp, entropy, value

    def train_ppo(
        self,
        batch: dict,
        *,
        clip_param: float,
        num_epochs: int,
        minibatch_size: int,
        normalize_advantages: bool = True,
    ) -> dict:
        obs = torch.as_tensor(batch["obs"], device=self.device, dtype=torch.float32)
        pre_tanh = torch.as_tensor(batch["pre_tanh_actions"], device=self.device, dtype=torch.float32)
        old_logp = torch.as_tensor(batch["old_logp"], device=self.device, dtype=torch.float32)
        returns = torch.as_tensor(batch["returns"], device=self.device, dtype=torch.float32)
        adv = torch.as_tensor(batch["advantages"], device=self.device, dtype=torch.float32)

        if normalize_advantages:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        n = int(obs.shape[0])
        if n == 0:
            return {}

        clip_param = float(clip_param)
        minibatch_size = int(max(16, minibatch_size))
        num_epochs = int(max(1, num_epochs))

        last = {}
        idx_all = torch.arange(n, device=self.device)
        for _ in range(num_epochs):
            perm = idx_all[torch.randperm(n, device=self.device)]
            for start in range(0, n, minibatch_size):
                mb_idx = perm[start : start + minibatch_size]
                new_logp, entropy, value = self.evaluate_batch(obs[mb_idx], pre_tanh[mb_idx])
                ratio = torch.exp(new_logp - old_logp[mb_idx])
                surr1 = ratio * adv[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(value, returns[mb_idx])
                entropy_mean = entropy.mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_mean

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.value.parameters()) + [self.log_std],
                        max_norm=self.max_grad_norm,
                    )
                self.opt.step()

                approx_kl = 0.5 * ((new_logp - old_logp[mb_idx]) ** 2).mean()
                last = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy_mean.item()),
                    "approx_kl": float(approx_kl.item()),
                }
        return last

    def export_state(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "value": self.value.state_dict(),
            "log_std": self.log_std.detach().cpu(),
            "opt": self.opt.state_dict(),
            "gamma": self.gamma,
            "lambda_": self.lambda_,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "max_grad_norm": self.max_grad_norm,
            "act_low": self.act_low.detach().cpu().numpy(),
            "act_high": self.act_high.detach().cpu().numpy(),
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
class PPOHyperparams:
    lr: float
    clip_param: float
    entropy_coef: float
    value_coef: float
    gamma: float
    lambda_: float
    num_epochs: int
    minibatch_size: int
    max_grad_norm: float

    def clamp(self) -> "PPOHyperparams":
        return PPOHyperparams(
            lr=float(np.clip(self.lr, 1e-6, 5e-3)),
            clip_param=float(np.clip(self.clip_param, 0.02, 0.4)),
            entropy_coef=float(np.clip(self.entropy_coef, 0.0, 0.05)),
            value_coef=float(np.clip(self.value_coef, 0.0, 5.0)),
            gamma=float(np.clip(self.gamma, 0.8, 0.999)),
            lambda_=float(np.clip(self.lambda_, 0.8, 0.999)),
            num_epochs=int(np.clip(self.num_epochs, 1, 20)),
            minibatch_size=int(np.clip(self.minibatch_size, 32, 8192)),
            max_grad_norm=float(np.clip(self.max_grad_norm, 0.0, 10.0)),
        )


def _algo_key(species: str, pop_id: int) -> str:
    return f"{ALGO_PREFIX}_{species}_{pop_id}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("PBT orchestrator for 10 independent Torch PPO populations")
    p.add_argument("--env-config-file", type=Path, default=None)
    p.add_argument("--log-dir", type=Path, default=Path("logs/pbt_orchestrator_ppo_torch"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-episodes", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--n-populations", type=int, default=None)

    p.add_argument("--pbt-interval", type=int, default=20)
    p.add_argument("--elite-fraction", type=float, default=0.4)

    # PPO init distributions / fixed params.
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lambda", dest="lambda_", type=float, default=0.95)
    p.add_argument("--init-lr-log-mean", type=float, default=math.log(3e-4))
    p.add_argument("--init-lr-log-std", type=float, default=0.6)
    p.add_argument("--init-clip-mean", type=float, default=0.2)
    p.add_argument("--init-clip-std", type=float, default=0.06)
    p.add_argument("--init-entropy-mean", type=float, default=0.005)
    p.add_argument("--init-entropy-std", type=float, default=0.005)
    p.add_argument("--init-value-coef-mean", type=float, default=0.5)
    p.add_argument("--init-value-coef-std", type=float, default=0.3)
    p.add_argument("--num-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=2048)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--normalize-advantages", action="store_true", default=True)
    p.add_argument("--no-normalize-advantages", action="store_true")

    # Mutation magnitudes.
    p.add_argument("--mutate-lr-sigma", type=float, default=0.25)
    p.add_argument("--mutate-clip-sigma", type=float, default=0.03)
    p.add_argument("--mutate-entropy-sigma", type=float, default=0.003)
    p.add_argument("--mutate-value-sigma", type=float, default=0.2)

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


def sample_initial_hparams(rng: np.random.Generator, args: argparse.Namespace) -> PPOHyperparams:
    lr = _lognormal(rng, args.init_lr_log_mean, args.init_lr_log_std)
    clip_param = float(rng.normal(args.init_clip_mean, args.init_clip_std))
    entropy = float(rng.normal(args.init_entropy_mean, args.init_entropy_std))
    value_coef = float(rng.normal(args.init_value_coef_mean, args.init_value_coef_std))
    return PPOHyperparams(
        lr=lr,
        clip_param=clip_param,
        entropy_coef=entropy,
        value_coef=value_coef,
        gamma=float(args.gamma),
        lambda_=float(args.lambda_),
        num_epochs=int(args.num_epochs),
        minibatch_size=int(args.minibatch_size),
        max_grad_norm=float(args.max_grad_norm),
    ).clamp()


def mutate_hparams(rng: np.random.Generator, base: PPOHyperparams, args: argparse.Namespace) -> PPOHyperparams:
    lr = float(base.lr * math.exp(rng.normal(0.0, args.mutate_lr_sigma)))
    clip_param = float(base.clip_param + rng.normal(0.0, args.mutate_clip_sigma))
    entropy = float(base.entropy_coef + rng.normal(0.0, args.mutate_entropy_sigma))
    value_coef = float(base.value_coef + rng.normal(0.0, args.mutate_value_sigma))
    return PPOHyperparams(
        lr=lr,
        clip_param=clip_param,
        entropy_coef=entropy,
        value_coef=value_coef,
        gamma=base.gamma,
        lambda_=base.lambda_,
        num_epochs=base.num_epochs,
        minibatch_size=base.minibatch_size,
        max_grad_norm=base.max_grad_norm,
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
    hparams: Dict[str, PPOHyperparams],
    models: Dict[str, SimplePPO],
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
        models[loser].lambda_ = float(new_hp.lambda_)
        models[loser].entropy_coef = float(new_hp.entropy_coef)
        models[loser].value_coef = float(new_hp.value_coef)
        models[loser].max_grad_norm = float(new_hp.max_grad_norm)
        models[loser].set_lr(float(new_hp.lr))
        hparams[loser] = new_hp
        if writer is not None:
            writer.add_scalar(f"pbt/{loser}/lr", new_hp.lr, episode)
            writer.add_scalar(f"pbt/{loser}/clip_param", new_hp.clip_param, episode)
            writer.add_scalar(f"pbt/{loser}/entropy_coef", new_hp.entropy_coef, episode)
            writer.add_scalar(f"pbt/{loser}/value_coef", new_hp.value_coef, episode)


def _compute_gae_for_agent(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    bootstrap_value: float,
    *,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    t = int(len(rewards))
    adv = np.zeros((t,), dtype=np.float32)
    gae = 0.0
    for i in range(t - 1, -1, -1):
        next_v = float(values[i + 1]) if i + 1 < t else float(bootstrap_value)
        mask = 1.0 - float(dones[i])
        delta = float(rewards[i]) + gamma * mask * next_v - float(values[i])
        gae = delta + gamma * lam * mask * gae
        adv[i] = gae
    ret = adv + values.astype(np.float32)
    return adv, ret


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

    hparam_map: Dict[str, PPOHyperparams] = {k: sample_initial_hparams(rng, args) for k in keys}
    models: Dict[str, SimplePPO] = {}
    returns_hist: Dict[str, List[float]] = {k: [] for k in keys}
    best_seen: Dict[str, float] = {k: float("-inf") for k in keys}

    for key in keys:
        hp = hparam_map[key]
        models[key] = SimplePPO(
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_low=act_low,
            act_high=act_high,
            lr=hp.lr,
            gamma=hp.gamma,
            lambda_=hp.lambda_,
            entropy_coef=hp.entropy_coef,
            value_coef=hp.value_coef,
            max_grad_norm=hp.max_grad_norm,
            device="cpu",
        )

    writer = None
    if not args.no_tensorboard and TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=str(log_dir))

    env = PredPreyGrass(env_config)

    progress = None
    if not args.no_progress:
        progress = ProgressBar(int(args.num_episodes), label="[PBT-PPO-TORCH] ", stream=sys.stderr)
        progress.update(0)

    normalize_adv = bool(args.normalize_advantages) and (not bool(args.no_normalize_advantages))

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
                        "old_logp": float(extra_cache[aid]["logp"]),
                        "value": float(extra_cache[aid]["vf"]),
                        "rewards": float(rew),
                        "dones": float(done),
                        "next_obs": next_obs.get(aid, obs[aid]),
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

        # Train per key (PPO).
        for key in keys:
            hp = hparam_map[key]
            model = models[key]
            agent_traj = traj[key]
            if not agent_traj:
                continue

            # Bootstrap values for agents whose last transition is non-terminal.
            bootstrap_obs = []
            bootstrap_agents = []
            for aid, steps in agent_traj.items():
                if not steps:
                    continue
                if float(steps[-1]["dones"]) < 0.5:
                    bootstrap_obs.append(steps[-1]["next_obs"])
                    bootstrap_agents.append(aid)
            bootstrap_vals = {}
            if bootstrap_obs:
                v = model.value_batch(np.stack(bootstrap_obs).astype(np.float32))
                for i, aid in enumerate(bootstrap_agents):
                    bootstrap_vals[aid] = float(v[i])

            flat_obs = []
            flat_pre = []
            flat_old_logp = []
            flat_adv = []
            flat_ret = []

            for aid, steps in agent_traj.items():
                if not steps:
                    continue
                rewards = np.array([s["rewards"] for s in steps], dtype=np.float32)
                dones_arr = np.array([s["dones"] for s in steps], dtype=np.float32)
                values = np.array([s["value"] for s in steps], dtype=np.float32)
                bootstrap_value = float(bootstrap_vals.get(aid, 0.0))
                adv, ret = _compute_gae_for_agent(
                    rewards,
                    dones_arr,
                    values,
                    bootstrap_value,
                    gamma=float(hp.gamma),
                    lam=float(hp.lambda_),
                )
                for i, s in enumerate(steps):
                    flat_obs.append(s["obs"])
                    flat_pre.append(s["pre_tanh_actions"])
                    flat_old_logp.append(float(s["old_logp"]))
                    flat_adv.append(float(adv[i]))
                    flat_ret.append(float(ret[i]))

            if not flat_obs:
                continue

            batch = {
                "obs": np.stack(flat_obs).astype(np.float32),
                "pre_tanh_actions": np.stack(flat_pre).astype(np.float32),
                "old_logp": np.array(flat_old_logp, dtype=np.float32),
                "advantages": np.array(flat_adv, dtype=np.float32),
                "returns": np.array(flat_ret, dtype=np.float32),
            }

            stats = model.train_ppo(
                batch,
                clip_param=float(hp.clip_param),
                num_epochs=int(hp.num_epochs),
                minibatch_size=int(hp.minibatch_size),
                normalize_advantages=normalize_adv,
            )
            if writer is not None and stats:
                writer.add_scalar(f"loss/{key}/policy_loss", float(stats["policy_loss"]), ep)
                writer.add_scalar(f"loss/{key}/value_loss", float(stats["value_loss"]), ep)
                writer.add_scalar(f"loss/{key}/entropy", float(stats["entropy"]), ep)
                writer.add_scalar(f"loss/{key}/approx_kl", float(stats["approx_kl"]), ep)

        # Score per key.
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
            _select_and_mutate(
                rng,
                predator_keys,
                scores,
                hparam_map,
                models,
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
                args,
                writer=writer,
                episode=ep,
            )
            if writer is not None:
                for k in keys:
                    hp = hparam_map[k]
                    writer.add_scalar(f"hparams/{k}/lr", hp.lr, ep)
                    writer.add_scalar(f"hparams/{k}/clip_param", hp.clip_param, ep)
                    writer.add_scalar(f"hparams/{k}/entropy_coef", hp.entropy_coef, ep)
                    writer.add_scalar(f"hparams/{k}/value_coef", hp.value_coef, ep)

        if (ep + 1) % max(1, int(args.pbt_interval)) == 0:
            print(f"[PBT-PPO-TORCH] episode={ep} done (len={t+1})")
        if progress is not None:
            progress.update(float(ep + 1), detail=f"ep={ep+1}/{int(args.num_episodes)} last_len={t+1}/{int(args.max_steps)}")

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
    print("[PBT-PPO-TORCH] best predator:", best_pred, hparam_map[best_pred])
    print("[PBT-PPO-TORCH] best prey:", best_prey, hparam_map[best_prey])
    print("[PBT-PPO-TORCH] summary written to:", (log_dir / "pbt_summary.json"))

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

