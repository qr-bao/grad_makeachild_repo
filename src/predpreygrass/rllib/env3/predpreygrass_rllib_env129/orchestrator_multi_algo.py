"""
Multi-algorithm orchestrator (PPO + SAC + TD3 + A2C training; Random inference).

- Uses per-population mapping (species_pop -> algo).
- Builds lightweight RLlib Algorithm instances (num_workers=0) to reuse policies.
- PPO policies are trained via policy.postprocess_trajectory + learn_on_batch.
- SAC policies are trained off-policy from a simple replay buffer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.distributions as D
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    try:
        from tensorboardX import SummaryWriter  # type: ignore
    except Exception:
        SummaryWriter = None  # type: ignore
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.policy.sample_batch import SampleBatch

try:
    from ray.rllib.algorithms.td3 import TD3Config

    TD3_AVAILABLE = True
except Exception:
    TD3Config = None
    TD3_AVAILABLE = False

# Allow running this file directly without installing the package by ensuring the
# repository `src/` root is on sys.path.
_SRC_ROOT = Path(__file__).resolve().parents[4]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from predpreygrass.rllib.env3.predpreygrass_rllib_env129.predpreygrass_rllib_env import (
    PredPreyGrass,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.config.config_env_train import (
    config_env as config_env_train,
)


def create_env_config():
    return config_env_train.copy()

ALGO_CHOICES = ("ppo", "random", "sac", "td3", "a2c")


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


def _tanh_squash_scale(pre_tanh: torch.Tensor, act_low: torch.Tensor, act_high: torch.Tensor) -> torch.Tensor:
    a_raw = torch.tanh(pre_tanh)
    a = a_raw * (act_high - act_low) / 2 + (act_high + act_low) / 2
    return torch.max(torch.min(a, act_high), act_low)


def _tanh_squash_logp(dist: D.Normal, pre_tanh: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a_raw = torch.tanh(pre_tanh)
    logp = dist.log_prob(pre_tanh) - torch.log(1.0 - a_raw.pow(2) + eps)
    return logp.sum(-1)


def _compute_gae_for_agent(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    bootstrap_value: float,
    *,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    t = int(rewards.shape[0])
    adv = np.zeros(t, dtype=np.float32)
    last_gae = 0.0
    for i in reversed(range(t)):
        next_value = float(bootstrap_value) if i == t - 1 else float(values[i + 1])
        nonterminal = 1.0 - float(dones[i])
        delta = float(rewards[i]) + float(gamma) * next_value * nonterminal - float(values[i])
        last_gae = delta + float(gamma) * float(lam) * nonterminal * last_gae
        adv[i] = last_gae
    ret = adv + values.astype(np.float32)
    return adv, ret


class SimplePPO:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_low,
        act_high,
        *,
        device: str = "cpu",
        lr: float = 3e-4,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        entropy_coef: float = 0.005,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
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
        self._hparams: dict = {}

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = float(lr)

    def act_batch(self, obs_batch_np, deterministic: bool = False):
        with torch.no_grad():
            obs = torch.as_tensor(obs_batch_np, device=self.device, dtype=torch.float32)
            mean = self.actor(obs)
            std = self.log_std.exp()
            dist = D.Normal(mean, std)
            pre_tanh = mean if deterministic else dist.sample()
            action = _tanh_squash_scale(pre_tanh, self.act_low, self.act_high)
            logp = _tanh_squash_logp(dist, pre_tanh, eps=self._logp_eps)
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
            return self.value(obs).squeeze(-1).detach().cpu().numpy()

    def _evaluate_batch(self, obs_batch, pre_tanh_batch):
        obs = torch.as_tensor(obs_batch, device=self.device, dtype=torch.float32)
        pre_tanh = torch.as_tensor(pre_tanh_batch, device=self.device, dtype=torch.float32)
        mean = self.actor(obs)
        std = self.log_std.exp()
        dist = D.Normal(mean, std)
        logp = _tanh_squash_logp(dist, pre_tanh, eps=self._logp_eps)
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

        n = int(obs.shape[0])
        if n <= 0:
            return {}

        if normalize_advantages:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        clip_param = float(clip_param)
        num_epochs = int(max(1, num_epochs))
        minibatch_size = int(max(16, minibatch_size))

        last = {}
        idx_all = torch.arange(n, device=self.device)
        for _ in range(num_epochs):
            perm = idx_all[torch.randperm(n, device=self.device)]
            for start in range(0, n, minibatch_size):
                mb = perm[start : start + minibatch_size]
                new_logp, entropy, value = self._evaluate_batch(obs[mb], pre_tanh[mb])
                ratio = torch.exp(new_logp - old_logp[mb])
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(value, returns[mb])
                entropy_mean = entropy.mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_mean

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.value.parameters()) + [self.log_std],
                        max_norm=float(self.max_grad_norm),
                    )
                self.opt.step()

                approx_kl = 0.5 * ((new_logp - old_logp[mb]) ** 2).mean()
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


class SACActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.mean = MLP(obs_dim, act_dim, hidden=hidden, act_last=False)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        device: str = "cpu",
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        init_alpha: float = 0.2,
        target_entropy: float = -2.0,
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
        self.log_alpha = nn.Parameter(torch.tensor(np.log(init_alpha), device=device, dtype=torch.float32))
        self.target_entropy = float(target_entropy)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=float(critic_lr))
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=float(critic_lr))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=float(alpha_lr))

        self.gamma = float(gamma)
        self.tau = float(tau)
        self._logp_eps = 1e-6
        self.total_updates = 0
        self._hparams: dict = {}

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

    def _squash_and_scale(self, pre_tanh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        act_raw = torch.tanh(pre_tanh)
        act = act_raw * (self.act_high - self.act_low) / 2 + (self.act_high + self.act_low) / 2
        act = torch.max(torch.min(act, self.act_high), self.act_low)
        return act_raw, act

    def sample_action_batch(self, obs_batch_np, *, deterministic: bool = False):
        with torch.no_grad():
            obs = torch.as_tensor(obs_batch_np, device=self.device, dtype=torch.float32)
            mean, log_std = self.actor(obs)
            log_std = torch.clamp(log_std, -20.0, 2.0)
            std = log_std.exp()
            dist = D.Normal(mean, std)
            pre_tanh = mean if deterministic else dist.rsample()
            logp = _tanh_squash_logp(dist, pre_tanh, eps=self._logp_eps)
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

        with torch.no_grad():
            mean, log_std = self.actor(next_obs)
            log_std = torch.clamp(log_std, -20.0, 2.0)
            std = log_std.exp()
            dist = D.Normal(mean, std)
            pre_tanh = dist.rsample()
            next_logp = _tanh_squash_logp(dist, pre_tanh, eps=self._logp_eps)
            next_act_raw, _ = self._squash_and_scale(pre_tanh)
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

        mean, log_std = self.actor(obs)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        std = log_std.exp()
        dist = D.Normal(mean, std)
        pre_tanh = dist.rsample()
        logp = _tanh_squash_logp(dist, pre_tanh, eps=self._logp_eps)
        new_act_raw, _ = self._squash_and_scale(pre_tanh)
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


class SimpleTD3:
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_low,
        act_high,
        device="cpu",
        *,
        hidden: tuple[int, int] = (256, 256),
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.1,
        noise_clip: float = 0.2,
        policy_delay: int = 2,
    ):
        self.device = device
        self.act_low = torch.tensor(act_low, device=device, dtype=torch.float32)
        self.act_high = torch.tensor(act_high, device=device, dtype=torch.float32)
        self.actor = MLP(obs_dim, act_dim, hidden=hidden).to(device)
        self.actor_target = MLP(obs_dim, act_dim, hidden=hidden).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.q1 = MLP(obs_dim + act_dim, 1, hidden=hidden, act_last=False).to(device)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden=hidden, act_last=False).to(device)
        self.q1_target = MLP(obs_dim + act_dim, 1, hidden=hidden, act_last=False).to(device)
        self.q2_target = MLP(obs_dim + act_dim, 1, hidden=hidden, act_last=False).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=float(critic_lr))
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=float(critic_lr))
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_delay = int(policy_delay)
        self.total_updates = 0

    def compute_action(self, obs, noise_std=0.1):
        with torch.no_grad():
            o = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
            a = self.actor(o)
            a = a * (self.act_high - self.act_low) / 2 + (self.act_high + self.act_low) / 2
            if noise_std > 0:
                a = a + torch.randn_like(a) * noise_std
            a = torch.max(torch.min(a, self.act_high), self.act_low)
            return a.cpu().numpy()

    def compute_action_batch(self, obs_batch, noise_std=0.1):
        with torch.no_grad():
            o = torch.as_tensor(obs_batch, device=self.device, dtype=torch.float32)
            a = self.actor(o)
            a = a * (self.act_high - self.act_low) / 2 + (self.act_high + self.act_low) / 2
            if noise_std > 0:
                a = a + torch.randn_like(a) * noise_std
            a = torch.max(torch.min(a, self.act_high), self.act_low)
            return a.cpu().numpy()

    def train_step(self, batch):
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
        # Diagnostics
        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()) if self.total_updates % self.policy_delay == 0 else None,
        }

    def _soft_update(self, target, source):
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


class SimpleA2C:
    def __init__(self, obs_dim, act_dim, act_low, act_high, device="cpu"):
        self.device = device
        self.act_low = torch.tensor(act_low, device=device, dtype=torch.float32)
        self.act_high = torch.tensor(act_high, device=device, dtype=torch.float32)
        self.actor = MLP(obs_dim, act_dim, hidden=(256, 256)).to(device)
        self.value = MLP(obs_dim, 1, hidden=(256, 256), act_last=False).to(device)
        self.log_std = nn.Parameter(torch.zeros(act_dim, device=device))
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.value.parameters()) + [self.log_std], lr=3e-4
        )
        self.gamma = 0.99
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self._logp_eps = 1e-6
        self._hparams: dict = {}

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = float(lr)

    def act(self, obs_np, deterministic: bool = False):
        with torch.no_grad():
            obs = torch.as_tensor(obs_np, device=self.device, dtype=torch.float32)
            mean = self.actor(obs)
            std = self.log_std.exp()
            dist = D.Normal(mean, std)
            if deterministic:
                pre_tanh = mean
            else:
                pre_tanh = dist.sample()
            action_raw = torch.tanh(pre_tanh)
            # Squashed Gaussian logp (ignore final linear scale constant; does not affect gradients).
            logp = dist.log_prob(pre_tanh) - torch.log(1.0 - action_raw.pow(2) + self._logp_eps)
            logp = logp.sum(-1).item()
            value = self.value(obs).item()
            # scale to bounds
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

    def train_batch(self, batch):
        obs = torch.as_tensor(batch["obs"], device=self.device, dtype=torch.float32)
        pre_tanh_acts = torch.as_tensor(batch["pre_tanh_actions"], device=self.device, dtype=torch.float32)
        rews = torch.as_tensor(batch["rewards"], device=self.device, dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], device=self.device, dtype=torch.float32)

        # compute returns (discounted)
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
        action_raw = torch.tanh(pre_tanh_acts)
        logp = dist.log_prob(pre_tanh_acts) - torch.log(1.0 - action_raw.pow(2) + self._logp_eps)
        logp = logp.sum(-1, keepdim=True)
        # Entropy of the unsquashed Normal (approximate; stable and sufficient for tuning).
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Multi-algo orchestrator (PPO train + Random/SAC infer)")
    parser.add_argument("--mode", type=str, default="train", choices=("train", "eval"))
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mapping-json", type=str, default=None)
    parser.add_argument("--env-config-file", type=str, default=None)
    parser.add_argument(
        "--torch-ppo-sac",
        action="store_true",
        help="Use pure-Torch PPO/SAC (instead of RLlib). Enables loading Torch-PBT `.pt` checkpoints.",
    )
    parser.add_argument(
        "--init-checkpoints-json",
        type=str,
        default=None,
        help="Optional JSON dict: {algo_key: path_to_pt}. Loads Torch checkpoints for PPO/SAC/TD3/A2C keys.",
    )
    parser.add_argument(
        "--strict-one-pop-per-key",
        action="store_true",
        help="Error if one algo_key is assigned to multiple populations (mapping values must be unique).",
    )
    parser.add_argument("--torch-device", type=str, default="cpu")
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument(
        "--algo-config-json",
        type=str,
        default=None,
        help="Optional JSON with algo hyperparams, e.g. {\"td3\": {...}, \"td3_train\": {...}}.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/orchestrator",
        help="Base directory for TensorBoard/event logs.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (default: <log-dir>/checkpoints).",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=5,
        help="Save checkpoints every N episodes (<=0 disables).",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Optional path to write per-episode metrics as JSON (useful for eval/tuning).",
    )
    return parser.parse_args()


def load_mapping(path: str | None, n_pops: int) -> Dict[str, str]:
    if path:
        with open(path, "r", encoding="utf-8") as fp:
            mapping = json.load(fp)
    else:
        mapping = {f"predator_{i}": "ppo" for i in range(n_pops)}
        mapping.update({f"prey_{i}": "random" for i in range(n_pops)})
    for v in mapping.values():
        if v not in ALGO_CHOICES:
            # 允许自定义名称（例如 ppo_predator）；前缀必须是支持的算法
            if not any(v.startswith(base) for base in ALGO_CHOICES):
                raise ValueError(f"Unsupported algo {v}, allowed prefixes {ALGO_CHOICES}")
    return mapping


def _algo_instance_key(algo_name: str, meta: dict) -> str:
    """Return a key to select the correct algo/model instance.

    - 如果映射值已经是自定义键（例如 ppo_predator），直接使用；
    - 如果使用了基础名称（ppo/sac/td3/a2c/random），为 td3/a2c 自动按物种拆分，防止捕食/逃逸混用。
    """
    if any(algo_name.startswith(base) and algo_name != base for base in ALGO_CHOICES):
        return algo_name
    species = meta.get("species")
    if algo_name in ("td3", "a2c") and species:
        return f"{algo_name}_{species}"
    return algo_name


def _algo_base_name(algo_key: str) -> str:
    for base in ALGO_CHOICES:
        if algo_key.startswith(base):
            return base
    raise ValueError(f"Unsupported algo key {algo_key}, expected prefix in {ALGO_CHOICES}")


def build_algorithms(mapping: Dict[str, str], env_config: dict):
    sample_env = PredPreyGrass(env_config)
    first_obs_space = next(iter(sample_env.observation_spaces.values()))
    first_act_space = next(iter(sample_env.action_spaces.values()))
    if hasattr(sample_env, "close"):
        sample_env.close()

    algos = {}
    for algo_key in set(mapping.values()):
        base = _algo_base_name(algo_key)
        if base == "ppo":
            cfg = PPOConfig()
            if hasattr(cfg, "api_stack"):
                cfg = cfg.api_stack(
                    enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
                )
            if hasattr(cfg, "_enable_rl_module_api"):
                cfg._enable_rl_module_api = False
            if hasattr(cfg, "_enable_learner_api"):
                cfg._enable_learner_api = False
            cfg = cfg.framework("torch").environment(
                env=PredPreyGrass,
                env_config=env_config,
                disable_env_checking=True,
                observation_space=first_obs_space,
                action_space=first_act_space,
            )
            if hasattr(cfg, "auto_wrap_old_gym_envs"):
                cfg.auto_wrap_old_gym_envs = False
            if hasattr(cfg, "env_runners"):
                cfg = cfg.env_runners(num_env_runners=0)
            elif hasattr(cfg, "rollouts"):
                cfg = cfg.rollouts(num_rollout_workers=0, num_envs_per_worker=1)
            algos[algo_key] = cfg.build()
        elif base == "sac":
            cfg = SACConfig()
            if hasattr(cfg, "api_stack"):
                cfg = cfg.api_stack(
                    enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
                )
            if hasattr(cfg, "_enable_rl_module_api"):
                cfg._enable_rl_module_api = False
            if hasattr(cfg, "_enable_learner_api"):
                cfg._enable_learner_api = False
            cfg = cfg.framework("torch").environment(
                env=PredPreyGrass,
                env_config=env_config,
                disable_env_checking=True,
                observation_space=first_obs_space,
                action_space=first_act_space,
            )
            if hasattr(cfg, "auto_wrap_old_gym_envs"):
                cfg.auto_wrap_old_gym_envs = False
            if hasattr(cfg, "env_runners"):
                cfg = cfg.env_runners(num_env_runners=0)
            elif hasattr(cfg, "rollouts"):
                cfg = cfg.rollouts(num_rollout_workers=0, num_envs_per_worker=1)
            cfg = cfg.training(
                train_batch_size=64,
                replay_buffer_config={
                    "type": "MultiAgentPrioritizedReplayBuffer",
                    "capacity": int(1e5),
                    "prioritized_replay_alpha": 0.6,
                    "prioritized_replay_beta": 0.4,
                    "prioritized_replay_eps": 1e-6,
                },
                num_steps_sampled_before_learning_starts=32,
                n_step=1,
            )
            algos[algo_key] = cfg.build()
        elif base == "a2c":
            algos[algo_key] = None
        elif base == "td3":
            # Custom TD3 (no Ray algo)
            algos[algo_key] = None
        elif base == "random":
            # no algorithm needed
            continue
        else:
            raise ValueError(f"Unsupported algo {algo_key}")
    return algos


def _assign_algo(agent_id: str, info: dict, mapping: Dict[str, str]) -> str:
    pop_id = info.get("population_id")
    species = info.get("species")
    key = f"{species}_{pop_id}" if species is not None and pop_id is not None else None
    if key and key in mapping:
        return mapping[key]
    if "predator" in agent_id:
        return mapping.get("predator_0", "ppo")
    if "prey" in agent_id:
        return mapping.get("prey_0", "random")
    return "random"


def _enrich_info(agent_id: str, info: dict, env) -> dict:
    """Best-effort fill species/population_id for mapping/logging."""
    meta = dict(info) if info else {}
    if "species" not in meta:
        if "predator" in agent_id:
            meta["species"] = "predator"
        elif "prey" in agent_id:
            meta["species"] = "prey"
    if "population_id" not in meta:
        try:
            pop_map = getattr(env, "agent_population_id", {})
            if agent_id in pop_map:
                meta["population_id"] = pop_map[agent_id]
        except Exception:
            pass
    return meta


def _maybe_save_checkpoints(
    checkpoint_dir: Path,
    ep: int,
    algos: dict,
    ppo_models: dict,
    sac_models: dict,
    td3_models: dict,
    a2c_models: dict,
    replay_buffers: dict,
    checkpoint_freq: int,
):
    if checkpoint_freq <= 0:
        return
    if (ep + 1) % checkpoint_freq != 0:
        return
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Save RLlib algos (ppo/sac)
    for name, algo in algos.items():
        if algo is None:
            continue
        try:
            algo.save(str(checkpoint_dir / f"{name}_ep{ep+1}"))
        except Exception as exc:
            print(f"[WARN] Failed to checkpoint algo {name}: {exc}")
    # Save Torch models (PPO/SAC/TD3/A2C)
    for name, model in ppo_models.items():
        path = checkpoint_dir / f"{name}_ep{ep+1}.pt"
        torch.save({"hparams": getattr(model, "_hparams", {}), "state": model.export_state()}, path)
    for name, model in sac_models.items():
        path = checkpoint_dir / f"{name}_ep{ep+1}.pt"
        torch.save(
            {
                "hparams": getattr(model, "_hparams", {}),
                "state": model.export_state(),
                "buffer": replay_buffers.get(name, []),
            },
            path,
        )
    for name, model in td3_models.items():
        path = checkpoint_dir / f"{name}_ep{ep+1}.pt"
        torch.save(
            {
                "hparams": getattr(model, "_hparams", {}),
                "state": model.export_state(),
                "buffer": replay_buffers.get(name, []),
            },
            path,
        )
    for name, model in a2c_models.items():
        path = checkpoint_dir / f"{name}_ep{ep+1}.pt"
        torch.save({"hparams": getattr(model, "_hparams", {}), "state": model.export_state()}, path)


def orchestrate(
    algo_map: Dict[str, str],
    env_config: dict,
    num_episodes: int,
    max_steps: int,
    log_dir: Path,
    checkpoint_dir: Path,
    checkpoint_freq: int,
    *,
    mode: str = "train",
    seed: int = 0,
    export_json: str | None = None,
    td3_hparams: dict | None = None,
    td3_train: dict | None = None,
    torch_ppo_sac: bool = False,
    torch_device: str = "cpu",
    torch_num_threads: int = 1,
    init_checkpoints: dict | None = None,
):
    env = PredPreyGrass(env_config)
    torch.set_num_threads(int(torch_num_threads))
    init_checkpoints = dict(init_checkpoints or {})

    # Build RLlib algorithms only if we are using RLlib PPO/SAC.
    if torch_ppo_sac:
        algos: dict = {}
    else:
        algos = build_algorithms(algo_map, env_config)
    policies = {name: algo.get_policy() for name, algo in algos.items() if algo is not None}
    ppo_models: Dict[str, SimplePPO] = {}
    sac_models: Dict[str, SimpleSAC] = {}
    td3_models = {}
    a2c_models = {}
    off_buffers: Dict[str, List[dict]] = {
        name: []
        for name in set(algo_map.values())
        if _algo_base_name(name) in ("sac", "td3")
    }
    td3_train = dict(td3_train or {})
    off_max_buffer = int(td3_train.get("buffer_size", 5e4))
    writer = SummaryWriter(log_dir=str(log_dir)) if mode == "train" else None
    agent_reward_sums: Dict[str, float] = {}
    agent_meta: Dict[str, Dict] = {}
    action_norms: Dict[str, List[float]] = {}
    loss_stats: Dict[str, List[dict]] = {}
    exported_episodes: List[dict] = []

    init_payloads: Dict[str, dict] = {}
    init_status: Dict[str, str] = {}
    for algo_key in set(algo_map.values()):
        if _algo_base_name(algo_key) == "random":
            init_status[algo_key] = "baseline_random"
            continue
        path = init_checkpoints.get(algo_key)
        if not path:
            init_status[algo_key] = "missing"
            continue
        p = Path(path).expanduser()
        if not p.exists():
            init_status[algo_key] = f"missing_file: {p}"
            continue
        try:
            # PyTorch 2.6+ defaults to `weights_only=True`, which can reject checkpoints
            # that contain non-tensor objects (e.g., numpy arrays) even if they are trusted
            # local artifacts. Our PBT checkpoints are produced by this codebase, so we
            # explicitly allow full deserialization here.
            try:
                init_payloads[algo_key] = torch.load(str(p), map_location=torch_device, weights_only=False)
            except TypeError:
                init_payloads[algo_key] = torch.load(str(p), map_location=torch_device)
            init_status[algo_key] = "loaded"
        except Exception as exc:
            init_status[algo_key] = f"load_failed: {exc}"

    def _apply_checkpoint(algo_key: str, model, *, kind: str) -> None:
        status = init_status.get(algo_key, "missing")
        if status.startswith("load_failed") or status.startswith("missing_file"):
            print(f"[INIT][WARN] {algo_key}: {status} -> random init ({kind})")
            return
        payload = init_payloads.get(algo_key)
        if payload is None:
            print(f"[INIT] {algo_key}: no checkpoint provided -> random init ({kind})")
            return
        state = payload.get("state", payload) if isinstance(payload, dict) else payload
        hparams = payload.get("hparams", {}) if isinstance(payload, dict) else {}
        try:
            if isinstance(state, dict) and hasattr(model, "import_state"):
                model.import_state(state)
            else:
                raise ValueError("Unsupported checkpoint format (expected dict with 'state')")
            if isinstance(hparams, dict) and hparams:
                setattr(model, "_hparams", dict(hparams))
            init_status[algo_key] = "applied"
            print(f"[INIT] {algo_key}: loaded + applied ({kind})")
        except Exception as exc:
            init_status[algo_key] = f"apply_failed: {exc}"
            print(f"[INIT][WARN] {algo_key}: apply_failed: {exc} -> random init ({kind})")

    for ep in range(num_episodes):
        obs, infos = env.reset(seed=seed + ep)
        if ep == 0:
            # Create all algo instances (one key per population) up-front so we can load checkpoints
            # and print a full init report before training starts.
            species_sample_aid: Dict[str, str] = {}
            for aid in obs.keys():
                if "predator" in aid and "predator" not in species_sample_aid:
                    species_sample_aid["predator"] = aid
                if "prey" in aid and "prey" not in species_sample_aid:
                    species_sample_aid["prey"] = aid
            if not species_sample_aid:
                raise RuntimeError("No agents in initial reset; cannot infer observation/action shapes.")

            species_by_algo_key: Dict[str, str] = {}
            for pop_key, algo_key in algo_map.items():
                if _algo_base_name(algo_key) == "random":
                    continue
                if algo_key in species_by_algo_key:
                    continue
                if pop_key.startswith("predator_"):
                    species_by_algo_key[algo_key] = "predator"
                elif pop_key.startswith("prey_"):
                    species_by_algo_key[algo_key] = "prey"

            for algo_key in sorted(set(algo_map.values())):
                base = _algo_base_name(algo_key)
                if base == "random":
                    continue
                species = species_by_algo_key.get(algo_key)
                aid = species_sample_aid.get(species or "", next(iter(species_sample_aid.values())))
                ob = obs[aid]
                act_space = env.action_spaces[aid]
                act_low = act_space.low
                act_high = act_space.high
                act_dim = int(act_space.shape[0])

                if base == "ppo":
                    if algo_key not in ppo_models:
                        ppo_models[algo_key] = SimplePPO(
                            obs_dim=int(ob.shape[0]),
                            act_dim=act_dim,
                            act_low=act_low,
                            act_high=act_high,
                            device=torch_device,
                        )
                    _apply_checkpoint(algo_key, ppo_models[algo_key], kind="ppo")
                    hp = getattr(ppo_models[algo_key], "_hparams", {})
                    if isinstance(hp, dict) and hp:
                        if "lr" in hp:
                            ppo_models[algo_key].set_lr(float(hp["lr"]))
                        ppo_models[algo_key].gamma = float(hp.get("gamma", ppo_models[algo_key].gamma))
                        ppo_models[algo_key].lambda_ = float(hp.get("lambda_", ppo_models[algo_key].lambda_))
                        ppo_models[algo_key].entropy_coef = float(hp.get("entropy_coef", ppo_models[algo_key].entropy_coef))
                        ppo_models[algo_key].value_coef = float(hp.get("value_coef", ppo_models[algo_key].value_coef))
                        ppo_models[algo_key].max_grad_norm = float(hp.get("max_grad_norm", ppo_models[algo_key].max_grad_norm))
                elif base == "sac":
                    if algo_key not in sac_models:
                        # default target_entropy = -|A|
                        sac_models[algo_key] = SimpleSAC(
                            obs_dim=int(ob.shape[0]),
                            act_dim=act_dim,
                            act_low=act_low,
                            act_high=act_high,
                            device=torch_device,
                            target_entropy=-float(act_dim),
                        )
                    _apply_checkpoint(algo_key, sac_models[algo_key], kind="sac")
                    hp = getattr(sac_models[algo_key], "_hparams", {})
                    if isinstance(hp, dict) and hp:
                        sac_models[algo_key].gamma = float(hp.get("gamma", sac_models[algo_key].gamma))
                        sac_models[algo_key].tau = float(hp.get("tau", sac_models[algo_key].tau))
                        sac_models[algo_key].target_entropy = float(hp.get("target_entropy", sac_models[algo_key].target_entropy))
                        sac_models[algo_key].set_lrs(
                            actor_lr=float(hp.get("actor_lr", 3e-4)),
                            critic_lr=float(hp.get("critic_lr", 3e-4)),
                            alpha_lr=float(hp.get("alpha_lr", 3e-4)),
                        )
                elif base == "td3":
                    if algo_key not in td3_models:
                        hparams = dict(td3_hparams or {})
                        td3_models[algo_key] = SimpleTD3(
                            obs_dim=int(ob.shape[0]),
                            act_dim=act_dim,
                            act_low=act_low,
                            act_high=act_high,
                            device=torch_device,
                            **hparams,
                        )
                    _apply_checkpoint(algo_key, td3_models[algo_key], kind="td3")
                    hp = getattr(td3_models[algo_key], "_hparams", {})
                    if isinstance(hp, dict) and hp:
                        if "lr" in hp:
                            lr = float(hp["lr"])
                            for g in td3_models[algo_key].actor_opt.param_groups:
                                g["lr"] = lr
                            for g in td3_models[algo_key].q1_opt.param_groups:
                                g["lr"] = lr
                            for g in td3_models[algo_key].q2_opt.param_groups:
                                g["lr"] = lr
                        td3_models[algo_key].gamma = float(hp.get("gamma", td3_models[algo_key].gamma))
                        td3_models[algo_key].tau = float(hp.get("tau", td3_models[algo_key].tau))
                        td3_models[algo_key].policy_noise = float(hp.get("policy_noise", td3_models[algo_key].policy_noise))
                        td3_models[algo_key].noise_clip = float(hp.get("noise_clip", td3_models[algo_key].noise_clip))
                        td3_models[algo_key].policy_delay = int(hp.get("policy_delay", td3_models[algo_key].policy_delay))
                elif base == "a2c":
                    if algo_key not in a2c_models:
                        a2c_models[algo_key] = SimpleA2C(
                            obs_dim=int(ob.shape[0]),
                            act_dim=act_dim,
                            act_low=act_low,
                            act_high=act_high,
                            device=torch_device,
                        )
                    _apply_checkpoint(algo_key, a2c_models[algo_key], kind="a2c")
                    hp = getattr(a2c_models[algo_key], "_hparams", {})
                    if isinstance(hp, dict) and hp:
                        if "lr" in hp:
                            a2c_models[algo_key].set_lr(float(hp["lr"]))
                        a2c_models[algo_key].gamma = float(hp.get("gamma", a2c_models[algo_key].gamma))
                        a2c_models[algo_key].entropy_coef = float(hp.get("entropy_coef", a2c_models[algo_key].entropy_coef))
                        a2c_models[algo_key].value_coef = float(hp.get("value_coef", a2c_models[algo_key].value_coef))

            # Print a consolidated report.
            for algo_key in sorted(set(algo_map.values())):
                print(f"[INIT][REPORT] {algo_key}: {init_status.get(algo_key, 'missing')}")
        traj: Dict[str, Dict[str, List[dict]]] = {}
        ep_reward_per_algo: Dict[str, float] = {}
        agent_reward_sums.clear()
        agent_meta.clear()
        action_norms.clear()
        loss_stats = {}
        for step in range(max_steps + 1):
            actions: Dict[str, np.ndarray] = {}
            extra_cache: Dict[str, dict] = {}
            a2c_groups: Dict[str, List[str]] = {}
            td3_groups: Dict[str, List[str]] = {}
            rllib_groups: Dict[str, List[str]] = {}
            ppo_groups: Dict[str, List[str]] = {}
            sac_groups: Dict[str, List[str]] = {}

            for aid, ob in obs.items():
                meta = _enrich_info(aid, infos.get(aid, {}), env)
                algo_name = _assign_algo(aid, meta, algo_map)
                algo_key = _algo_instance_key(algo_name, meta)
                base = _algo_base_name(algo_key)
                if base == "random":
                    act = env.action_spaces[aid].sample()
                    logp = 0.0
                    vf = 0.0
                    pre_tanh = None
                elif base == "a2c":
                    a2c_groups.setdefault(algo_key, []).append(aid)
                    continue
                elif base == "td3":
                    td3_groups.setdefault(algo_key, []).append(aid)
                    continue
                elif base == "ppo":
                    if torch_ppo_sac:
                        ppo_groups.setdefault(algo_key, []).append(aid)
                    else:
                        rllib_groups.setdefault(algo_key, []).append(aid)
                    continue
                elif base == "sac":
                    if torch_ppo_sac:
                        sac_groups.setdefault(algo_key, []).append(aid)
                    else:
                        rllib_groups.setdefault(algo_key, []).append(aid)
                    continue
                else:
                    pol = policies[algo_key]
                    act, _, extra = pol.compute_single_action(ob, explore=(mode == "train"))
                    logp = extra.get("action_logp", 0.0)
                    vf = extra.get("vf_preds", 0.0)
                    pre_tanh = None
                actions[aid] = act
                extra_cache[aid] = {"logp": logp, "vf": vf, "algo_key": algo_key, "pre_tanh": pre_tanh}
                try:
                    action_norms.setdefault(algo_key, []).append(float(np.linalg.norm(act)))
                except Exception:
                    pass

            if rllib_groups:
                explore = mode == "train"
                for algo_key, aids in rllib_groups.items():
                    pol = policies[algo_key]
                    obs_batch = np.stack([obs[aid] for aid in aids]).astype(np.float32)
                    act_b, _, extra = pol.compute_actions_from_input_dict({SampleBatch.OBS: obs_batch}, explore=explore)
                    logp_b = extra.get("action_logp", None)
                    vf_b = extra.get("vf_preds", None)
                    for i, aid in enumerate(aids):
                        actions[aid] = act_b[i]
                        extra_cache[aid] = {
                            "logp": float(logp_b[i]) if logp_b is not None else 0.0,
                            "vf": float(vf_b[i]) if vf_b is not None else 0.0,
                            "algo_key": algo_key,
                            "pre_tanh": None,
                        }
                        try:
                            action_norms.setdefault(algo_key, []).append(float(np.linalg.norm(act_b[i])))
                        except Exception:
                            pass

            if ppo_groups:
                deterministic = mode != "train"
                for algo_key, aids in ppo_groups.items():
                    model = ppo_models.get(algo_key)
                    if model is None:
                        raise RuntimeError(f"Missing Torch PPO model for key {algo_key}")
                    obs_batch = np.stack([obs[aid] for aid in aids]).astype(np.float32)
                    act_b, logp_b, vf_b, pre_b = model.act_batch(obs_batch, deterministic=deterministic)
                    for i, aid in enumerate(aids):
                        actions[aid] = act_b[i]
                        extra_cache[aid] = {
                            "logp": float(logp_b[i]),
                            "vf": float(vf_b[i]),
                            "algo_key": algo_key,
                            "pre_tanh": pre_b[i],
                        }
                        try:
                            action_norms.setdefault(algo_key, []).append(float(np.linalg.norm(act_b[i])))
                        except Exception:
                            pass

            if sac_groups:
                deterministic = mode != "train"
                for algo_key, aids in sac_groups.items():
                    model = sac_models.get(algo_key)
                    if model is None:
                        raise RuntimeError(f"Missing Torch SAC model for key {algo_key}")
                    obs_batch = np.stack([obs[aid] for aid in aids]).astype(np.float32)
                    act_b, act_raw_b, logp_b = model.sample_action_batch(obs_batch, deterministic=deterministic)
                    for i, aid in enumerate(aids):
                        actions[aid] = act_b[i]
                        extra_cache[aid] = {
                            "logp": float(logp_b[i]),
                            "vf": 0.0,
                            "algo_key": algo_key,
                            "pre_tanh": None,
                            "act_raw": act_raw_b[i],
                        }
                        try:
                            action_norms.setdefault(algo_key, []).append(float(np.linalg.norm(act_b[i])))
                        except Exception:
                            pass

            if a2c_groups:
                deterministic = mode != "train"
                for algo_key, aids in a2c_groups.items():
                    if algo_key not in a2c_models:
                        sample_aid = aids[0]
                        act_low = env.action_spaces[sample_aid].low
                        act_high = env.action_spaces[sample_aid].high
                        a2c_models[algo_key] = SimpleA2C(
                            obs_dim=obs[sample_aid].shape[0],
                            act_dim=env.action_spaces[sample_aid].shape[0],
                            act_low=act_low,
                            act_high=act_high,
                            device=torch_device,
                        )
                    obs_batch = np.stack([obs[aid] for aid in aids]).astype(np.float32)
                    act_b, logp_b, vf_b, pre_b = a2c_models[algo_key].act_batch(obs_batch, deterministic=deterministic)
                    for i, aid in enumerate(aids):
                        actions[aid] = act_b[i]
                        extra_cache[aid] = {
                            "logp": float(logp_b[i]),
                            "vf": float(vf_b[i]),
                            "algo_key": algo_key,
                            "pre_tanh": pre_b[i],
                        }
                        try:
                            action_norms.setdefault(algo_key, []).append(float(np.linalg.norm(act_b[i])))
                        except Exception:
                            pass

            if td3_groups:
                noise_std = float(td3_train.get("exploration_noise_std", 0.1)) if mode == "train" else 0.0
                for algo_key, aids in td3_groups.items():
                    if algo_key not in td3_models:
                        sample_aid = aids[0]
                        act_low = env.action_spaces[sample_aid].low
                        act_high = env.action_spaces[sample_aid].high
                        hparams = dict(td3_hparams or {})
                        td3_models[algo_key] = SimpleTD3(
                            obs_dim=obs[sample_aid].shape[0],
                            act_dim=env.action_spaces[sample_aid].shape[0],
                            act_low=act_low,
                            act_high=act_high,
                            device=torch_device,
                            **hparams,
                        )
                    obs_batch = np.stack([obs[aid] for aid in aids]).astype(np.float32)
                    act_b = td3_models[algo_key].compute_action_batch(obs_batch, noise_std=noise_std)
                    for i, aid in enumerate(aids):
                        actions[aid] = act_b[i]
                        extra_cache[aid] = {"logp": 0.0, "vf": 0.0, "algo_key": algo_key, "pre_tanh": None}
                        try:
                            action_norms.setdefault(algo_key, []).append(float(np.linalg.norm(act_b[i])))
                        except Exception:
                            pass

            next_obs, rewards, dones, truncs, infos = env.step(actions)
            for aid, rew in rewards.items():
                meta = _enrich_info(aid, infos.get(aid, {}), env)
                algo_name = _assign_algo(aid, meta, algo_map)
                algo_key = _algo_instance_key(algo_name, meta)
                if algo_key not in traj:
                    traj[algo_key] = {}
                if aid not in obs:
                    continue
                agent_meta[aid] = meta
                buf = traj[algo_key].setdefault(aid, [])
                transition = {
                    "obs": obs[aid],
                    "actions": actions[aid],
                    "rewards": rew,
                    "dones": dones.get(aid, False) or truncs.get(aid, False),
                    "next_obs": next_obs.get(aid, obs[aid]),
                    "logp": extra_cache[aid]["logp"],
                    "vf": extra_cache[aid]["vf"],
                    "pre_tanh_actions": extra_cache[aid]["pre_tanh"],
                    "act_raw": extra_cache[aid].get("act_raw", None),
                    "t": len(buf),
                }
                buf.append(transition)
                ep_reward_per_algo[algo_key] = ep_reward_per_algo.get(algo_key, 0.0) + rew
                agent_reward_sums[aid] = agent_reward_sums.get(aid, 0.0) + rew
                if _algo_base_name(algo_key) in ("sac", "td3"):
                    off_buffers.setdefault(algo_key, []).append(transition)
                    if len(off_buffers[algo_key]) > off_max_buffer:
                        off_buffers[algo_key] = off_buffers[algo_key][-off_max_buffer:]

            obs = next_obs
            if dones.get("__all__") or truncs.get("__all__"):
                break

        # Training (PPO/SAC via Ray, TD3/A2C via custom)
        if mode == "train":
            for algo_key, agents_steps in traj.items():
                base = _algo_base_name(algo_key)
                if base == "random":
                    continue
                if base == "ppo":
                    if torch_ppo_sac:
                        model = ppo_models.get(algo_key)
                        if model is None:
                            continue
                        hp = getattr(model, "_hparams", {}) or {}
                        clip_param = float(hp.get("clip_param", 0.2))
                        num_epochs = int(hp.get("num_epochs", 4))
                        minibatch_size = int(hp.get("minibatch_size", 2048))
                        normalize_adv = True

                        bootstrap_obs = []
                        bootstrap_agents = []
                        for aid, steps in agents_steps.items():
                            if steps and not bool(steps[-1]["dones"]):
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
                        for aid, steps in agents_steps.items():
                            if not steps:
                                continue
                            rewards = np.array([float(s["rewards"]) for s in steps], dtype=np.float32)
                            dones_arr = np.array([float(bool(s["dones"])) for s in steps], dtype=np.float32)
                            values = np.array([float(s["vf"]) for s in steps], dtype=np.float32)
                            bootstrap_value = float(bootstrap_vals.get(aid, 0.0))
                            adv, ret = _compute_gae_for_agent(
                                rewards,
                                dones_arr,
                                values,
                                bootstrap_value,
                                gamma=float(model.gamma),
                                lam=float(model.lambda_),
                            )
                            for i, s in enumerate(steps):
                                if s.get("pre_tanh_actions") is None:
                                    continue
                                flat_obs.append(s["obs"])
                                flat_pre.append(s["pre_tanh_actions"])
                                flat_old_logp.append(float(s["logp"]))
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
                        stat = model.train_ppo(
                            batch,
                            clip_param=clip_param,
                            num_epochs=num_epochs,
                            minibatch_size=minibatch_size,
                            normalize_advantages=normalize_adv,
                        )
                        if stat:
                            loss_stats.setdefault(algo_key, []).append(stat)
                    else:
                        policy = policies.get(algo_key)
                        for steps in agents_steps.values():
                            if not steps:
                                continue
                            batch = SampleBatch(
                                {
                                    SampleBatch.OBS: np.stack([s["obs"] for s in steps]).astype(np.float32),
                                    SampleBatch.ACTIONS: np.stack([s["actions"] for s in steps]).astype(np.float32),
                                    SampleBatch.REWARDS: np.array([s["rewards"] for s in steps], dtype=np.float32),
                                    SampleBatch.DONES: np.array([s["dones"] for s in steps], dtype=np.bool_),
                                    SampleBatch.TERMINATEDS: np.array([s["dones"] for s in steps], dtype=np.bool_),
                                    SampleBatch.TRUNCATEDS: np.zeros(len(steps), dtype=np.bool_),
                                    SampleBatch.NEXT_OBS: np.stack([s["next_obs"] for s in steps]).astype(np.float32),
                                    SampleBatch.ACTION_LOGP: np.array([s["logp"] for s in steps], dtype=np.float32),
                                    SampleBatch.VF_PREDS: np.array([s["vf"] for s in steps], dtype=np.float32),
                                    SampleBatch.EPS_ID: np.zeros(len(steps), dtype=np.int32),
                                    SampleBatch.AGENT_INDEX: np.zeros(len(steps), dtype=np.int32),
                                    SampleBatch.T: np.array([s["t"] for s in steps], dtype=np.int32),
                                    SampleBatch.UNROLL_ID: np.zeros(len(steps), dtype=np.int32),
                                }
                            )
                            _, _, extra = policy.compute_actions_from_input_dict(
                                {SampleBatch.OBS: batch[SampleBatch.OBS]}, explore=False
                            )
                            if SampleBatch.ACTION_DIST_INPUTS in extra:
                                batch[SampleBatch.ACTION_DIST_INPUTS] = np.array(extra[SampleBatch.ACTION_DIST_INPUTS])
                            if SampleBatch.VF_PREDS in extra:
                                batch[SampleBatch.VF_PREDS] = np.array(extra[SampleBatch.VF_PREDS])
                            post = policy.postprocess_trajectory(batch, other_agent_batches=None, episode=None)
                            policy.learn_on_batch(post)
                elif base == "sac":
                    if torch_ppo_sac:
                        model = sac_models.get(algo_key)
                        if model is None:
                            continue
                        hp = getattr(model, "_hparams", {}) or {}
                        warmup = int(hp.get("warmup_steps", 2000))
                        batch_size = int(hp.get("batch_size", 256))
                        updates = int(hp.get("updates_per_episode", 1))
                        buffer = off_buffers.get(algo_key, [])
                        if updates <= 0 or len(buffer) < max(warmup, batch_size):
                            continue
                        for _ in range(updates):
                            idx = np.random.choice(len(buffer), size=batch_size, replace=len(buffer) < batch_size)
                            sample = [buffer[i] for i in idx]
                            mb = {
                                "obs": np.stack([s["obs"] for s in sample]).astype(np.float32),
                                "act_raw": np.stack([s["act_raw"] for s in sample]).astype(np.float32),
                                "rewards": np.array([float(s["rewards"]) for s in sample], dtype=np.float32),
                                "next_obs": np.stack([s["next_obs"] for s in sample]).astype(np.float32),
                                "dones": np.array([float(bool(s["dones"])) for s in sample], dtype=np.float32),
                            }
                            stat = model.train_step(mb)
                            if stat:
                                loss_stats.setdefault(algo_key, []).append(stat)
                    else:
                        policy = policies.get(algo_key)
                        buffer = off_buffers.get(algo_key, [])
                        batch_size = policy.config.get("train_batch_size", 64)
                        warmup = policy.config.get(
                            "num_steps_sampled_before_learning_starts",
                            policy.config.get("learning_starts", 0),
                        )
                        if len(buffer) >= max(batch_size, warmup):
                            updates = max(1, len(buffer) // batch_size // 2)
                            for _ in range(updates):
                                idx = np.random.choice(len(buffer), size=batch_size, replace=len(buffer) < batch_size)
                                sample = [buffer[i] for i in idx]
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
                                post = policy.postprocess_trajectory(batch, other_agent_batches=None, episode=None)
                                policy.learn_on_batch(post)
                                if hasattr(policy, "update_target"):
                                    policy.update_target()
                elif base == "a2c":
                    model = a2c_models.get(algo_key)
                    if not model:
                        continue
                    hp = getattr(model, "_hparams", {}) or {}
                    updates = int(hp.get("updates_per_episode", 1))
                    if updates <= 0:
                        continue
                    flat_obs = []
                    flat_pre = []
                    flat_rew = []
                    flat_done = []
                    for steps in agents_steps.values():
                        if not steps:
                            continue
                        for s in steps:
                            flat_obs.append(s["obs"])
                            flat_pre.append(s["pre_tanh_actions"])
                            flat_rew.append(float(s["rewards"]))
                            flat_done.append(float(s["dones"]))
                        flat_done[-1] = 1.0
                    if flat_obs:
                        mb = {
                            "obs": np.stack(flat_obs).astype(np.float32),
                            "pre_tanh_actions": np.stack(flat_pre).astype(np.float32),
                            "rewards": np.array(flat_rew, dtype=np.float32),
                            "dones": np.array(flat_done, dtype=np.float32),
                        }
                        for _ in range(updates):
                            stat = model.train_batch(mb)
                            if stat:
                                loss_stats.setdefault(algo_key, []).append(stat)
                elif base == "td3":
                    model = td3_models.get(algo_key)
                    buffer = off_buffers.get(algo_key, [])
                    hp = getattr(model, "_hparams", {}) if model is not None else {}
                    batch_size = int((hp or {}).get("batch_size", td3_train.get("batch_size", 64)))
                    warmup = int((hp or {}).get("warmup_steps", td3_train.get("warmup", 32)))
                    if model and len(buffer) >= max(batch_size, warmup):
                        updates = int((hp or {}).get("updates_per_episode", td3_train.get("updates_per_episode", 0)))
                        if updates <= 0:
                            updates = max(1, len(buffer) // batch_size // 2)
                        for _ in range(updates):
                            idx = np.random.choice(len(buffer), size=batch_size, replace=len(buffer) < batch_size)
                            sample = [buffer[i] for i in idx]
                            mb = {
                                "obs": np.stack([s["obs"] for s in sample]).astype(np.float32),
                                "actions": np.stack([s["actions"] for s in sample]).astype(np.float32),
                                "rewards": np.array([s["rewards"] for s in sample], dtype=np.float32),
                                "next_obs": np.stack([s["next_obs"] for s in sample]).astype(np.float32),
                                "dones": np.array([s["dones"] for s in sample], dtype=np.float32),
                            }
                            stat = model.train_step(mb)
                            if stat:
                                loss_stats.setdefault(algo_key, []).append(stat)

        # TensorBoard logging
        bucket: Dict[tuple, List[float]] = {}
        species_bucket: Dict[str, List[float]] = {}
        for aid, rsum in agent_reward_sums.items():
            meta = agent_meta.get(aid, {})
            species = meta.get("species") or ("predator" if "predator" in aid else "prey")
            algo_name = _assign_algo(aid, meta, algo_map)
            algo_key_for_log = _algo_instance_key(algo_name, meta)
            bucket.setdefault((species, algo_key_for_log), []).append(rsum)
            species_bucket.setdefault(species, []).append(rsum)

        episode_len = int(getattr(env, "current_step", step + 1))
        episode_payload = {
            "episode": ep,
            "episode_len": episode_len,
            "episode_reward_sum_per_algo": {k: float(v) for k, v in ep_reward_per_algo.items()},
            "episode_reward_mean_per_species": {k: float(np.mean(v)) for k, v in species_bucket.items() if v},
            "env_episode_metrics": (infos.get("__common__", {}) or {}).get("episode_metrics"),
        }
        exported_episodes.append(episode_payload)

        if writer is not None:
            for (species, algo_key_for_log), vals in bucket.items():
                writer.add_scalar(f"episode_reward_mean/{species}/{algo_key_for_log}", float(np.mean(vals)), ep)
                v = np.array(vals, dtype=np.float32)
                writer.add_scalar(f"episode_reward_p50/{species}/{algo_key_for_log}", float(np.median(v)), ep)
                writer.add_scalar(f"episode_reward_p25/{species}/{algo_key_for_log}", float(np.percentile(v, 25)), ep)
                writer.add_scalar(f"episode_reward_p75/{species}/{algo_key_for_log}", float(np.percentile(v, 75)), ep)
            for species, vals in species_bucket.items():
                writer.add_scalar(f"episode_reward_mean/{species}/ALL", float(np.mean(vals)), ep)
            for algo_key_for_log, rew_sum in ep_reward_per_algo.items():
                writer.add_scalar(f"episode_reward_sum/{algo_key_for_log}", rew_sum, ep)
            writer.add_scalar("episode_len", episode_len, ep)
            for algo_key_for_log, vals in action_norms.items():
                if vals:
                    writer.add_scalar(f"action_norm_mean/{algo_key_for_log}", float(np.mean(vals)), ep)
            pop_counts = getattr(env, "population_counts", {})
            if pop_counts:
                pred_total = sum(v for k, v in pop_counts.items() if k.startswith("predator_"))
                prey_total = sum(v for k, v in pop_counts.items() if k.startswith("prey_"))
                writer.add_scalar("population/predator/total", pred_total, ep)
                writer.add_scalar("population/prey/total", prey_total, ep)
                algo_population: Dict[tuple, float] = {}
                for key, val in pop_counts.items():
                    if key.startswith("predator_"):
                        writer.add_scalar(f"population/predator/{key}", val, ep)
                        try:
                            pop_id = int(key.split("_")[-1])
                            algo_label = algo_map.get(f"predator_{pop_id}")
                            if algo_label:
                                algo_population[("predator", algo_label)] = algo_population.get(("predator", algo_label), 0.0) + val
                        except Exception:
                            pass
                    elif key.startswith("prey_"):
                        writer.add_scalar(f"population/prey/{key}", val, ep)
                        try:
                            pop_id = int(key.split("_")[-1])
                            algo_label = algo_map.get(f"prey_{pop_id}")
                            if algo_label:
                                algo_population[("prey", algo_label)] = algo_population.get(("prey", algo_label), 0.0) + val
                        except Exception:
                            pass
                for (species, algo_label), val in algo_population.items():
                    # Log aggregated by算法名字，方便直接在 population/ 下看到按算法汇总的人口曲线
                    writer.add_scalar(f"population/{species}/{algo_label}", val, ep)
                    # 保留旧的 population_algo 路径以兼容历史日志
                    writer.add_scalar(f"population_algo/{species}/{algo_label}", val, ep)
            for key, stats in loss_stats.items():
                if key.startswith("td3"):
                    q1 = [s["q1_loss"] for s in stats if s.get("q1_loss") is not None]
                    q2 = [s["q2_loss"] for s in stats if s.get("q2_loss") is not None]
                    al = [s["actor_loss"] for s in stats if s.get("actor_loss") is not None]
                    prefix = f"loss/{key}"
                    if q1:
                        writer.add_scalar(f"{prefix}/q1_loss", float(np.mean(q1)), ep)
                    if q2:
                        writer.add_scalar(f"{prefix}/q2_loss", float(np.mean(q2)), ep)
                    if al:
                        writer.add_scalar(f"{prefix}/actor_loss", float(np.mean(al)), ep)
                if key.startswith("a2c"):
                    pl = [s["policy_loss"] for s in stats if s.get("policy_loss") is not None]
                    vl = [s["value_loss"] for s in stats if s.get("value_loss") is not None]
                    ent = [s["entropy"] for s in stats if s.get("entropy") is not None]
                    prefix = f"loss/{key}"
                    if pl:
                        writer.add_scalar(f"{prefix}/policy_loss", float(np.mean(pl)), ep)
                    if vl:
                        writer.add_scalar(f"{prefix}/value_loss", float(np.mean(vl)), ep)
                    if ent:
                        writer.add_scalar(f"{prefix}/entropy", float(np.mean(ent)), ep)

        print(f"Episode {ep} done at env_step {episode_len}")
        if mode == "train":
            _maybe_save_checkpoints(
                checkpoint_dir,
                ep,
                algos,
                ppo_models,
                sac_models,
                td3_models,
                a2c_models,
                off_buffers,
                checkpoint_freq,
            )

    for name, algo in algos.items():
        if algo is not None and hasattr(algo, "stop"):
            algo.stop()
    env.close()
    if writer is not None:
        writer.close()
    if export_json:
        export_path = Path(export_json).expanduser().resolve()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps({"episodes": exported_episodes}, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    log_dir = Path(args.log_dir).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir or os.path.join(log_dir, "checkpoints")).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    env_config = create_env_config()
    if args.env_config_file:
        with open(args.env_config_file, "r", encoding="utf-8") as fp:
            env_config.update(json.load(fp))
    if args.max_steps:
        env_config["max_steps"] = args.max_steps
    # Silence verbose env logging by default for cleaner console output.
    env_config["verbose_spawning"] = False
    env_config["verbose_engagement"] = False
    env_config["verbose_movement"] = False

    sample_env = PredPreyGrass(env_config)
    n_pops = sample_env.n_populations
    if hasattr(sample_env, "close"):
        sample_env.close()

    mapping = load_mapping(args.mapping_json, n_pops)

    if args.strict_one_pop_per_key:
        rev: Dict[str, List[str]] = {}
        for pop_key, algo_key in mapping.items():
            rev.setdefault(algo_key, []).append(pop_key)
        bad = {k: v for k, v in rev.items() if len(v) > 1}
        if bad:
            raise ValueError(f"--strict-one-pop-per-key violated (duplicates): {bad}")

    init_checkpoints = {}
    if args.init_checkpoints_json:
        with open(args.init_checkpoints_json, "r", encoding="utf-8") as fp:
            init_checkpoints = json.load(fp) or {}

    use_rllib = not bool(args.torch_ppo_sac)
    if use_rllib:
        ray.init(ignore_reinit_error=True, log_to_driver=False)
    algo_cfg = {}
    if args.algo_config_json:
        with open(args.algo_config_json, "r", encoding="utf-8") as fp:
            algo_cfg = json.load(fp) or {}
    orchestrate(
        mapping,
        env_config,
        args.num_episodes,
        args.max_steps,
        log_dir,
        checkpoint_dir,
        args.checkpoint_freq,
        mode=args.mode,
        seed=args.seed,
        export_json=args.export_json,
        td3_hparams=algo_cfg.get("td3"),
        td3_train=algo_cfg.get("td3_train"),
        torch_ppo_sac=bool(args.torch_ppo_sac),
        torch_device=str(args.torch_device),
        torch_num_threads=int(args.torch_num_threads),
        init_checkpoints=init_checkpoints,
    )
    if use_rllib:
        ray.shutdown()


if __name__ == "__main__":
    main()
