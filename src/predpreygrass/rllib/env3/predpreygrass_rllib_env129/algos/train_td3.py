#!/usr/bin/env python3
"""Custom TD3 trainer for PredPreyGrass (predators learn, prey random)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from predpreygrass.rllib.env3.predpreygrass_rllib_env129.predpreygrass_rllib_env import (
    PredPreyGrass,
)

Tensor = torch.Tensor


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--num-iterations", type=int, default=2000)
    p.add_argument("--max-env-steps", type=int, default=2000)
    p.add_argument("--checkpoint-freq", type=int, default=20)
    p.add_argument(
        "--env-config-file",
        type=Path,
        default=Path("predpreygrass_rllib_env129/config/config_env_base.json"),
    )
    p.add_argument("--log-dir", type=Path, default=Path("logs_td3"))
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--replay-capacity", type=int, default=1_000_000)
    p.add_argument("--warmup-steps", type=int, default=10_000)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--policy-delay", type=int, default=2)
    p.add_argument("--exploration-noise", type=float, default=0.1)
    p.add_argument("--noise-decay", type=float, default=0.999)
    p.add_argument("--min-noise", type=float, default=0.01)
    p.add_argument("--target-policy-noise", type=float, default=0.2)
    p.add_argument("--target-noise-clip", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    return p


def load_env_config(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Env config file not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    scope: Dict[str, dict] = {}
    exec(path.read_text(), scope)
    return scope.get("config_env_base") or scope.get("config") or {}


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.ptr

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        max_idx = self.capacity if self.full else self.ptr
        idx = np.random.randint(0, max_idx, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, act_dim)
        self.act_limit = act_limit

    def forward(self, obs: Tensor) -> Tensor:
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.act_limit


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs: Tensor, act: Tensor) -> Tensor:
        x = torch.cat([obs, act], dim=-1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


class TD3Agent:
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float, args, device):
        self.device = device
        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_delay = args.policy_delay
        self.target_policy_noise = args.target_policy_noise
        self.target_noise_clip = args.target_noise_clip

        self.actor = Actor(obs_dim, act_dim, act_limit).to(device)
        self.actor_target = Actor(obs_dim, act_dim, act_limit).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        self.critic1 = Critic(obs_dim, act_dim).to(device)
        self.critic2 = Critic(obs_dim, act_dim).to(device)
        self.critic1_target = Critic(obs_dim, act_dim).to(device)
        self.critic2_target = Critic(obs_dim, act_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=args.critic_lr,
        )

        self.total_updates = 0
        self.act_limit = act_limit

    def act(self, obs: np.ndarray, noise_scale: float) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action = self.actor(obs_t.unsqueeze(0)).squeeze(0).cpu().numpy()
        self.actor.train()
        if noise_scale > 0.0:
            action += noise_scale * np.random.randn(*action.shape)
        return np.clip(action, -self.act_limit, self.act_limit)

    def update(self, buffer: ReplayBuffer, batch_size: int, updates: int):
        if len(buffer) < batch_size:
            return
        for _ in range(updates):
            obs, action, reward, next_obs, done = buffer.sample(batch_size)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
            reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(-1)
            next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
            done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device).unsqueeze(-1)

            with torch.no_grad():
                noise = (
                    torch.randn_like(action_t) * self.target_policy_noise
                ).clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (
                    self.actor_target(next_obs_t) + noise
                ).clamp(-self.act_limit, self.act_limit)
                q1_target = self.critic1_target(next_obs_t, next_actions)
                q2_target = self.critic2_target(next_obs_t, next_actions)
                target_q = torch.min(q1_target, q2_target)
                y = reward_t + (1.0 - done_t) * self.gamma * target_q

            q1 = self.critic1(obs_t, action_t)
            q2 = self.critic2(obs_t, action_t)
            critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            self.total_updates += 1
            if self.total_updates % self.policy_delay == 0:
                actor_loss = -self.critic1(obs_t, self.actor(obs_t)).mean()
                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()
                self._soft_update(self.actor_target, self.actor)
                self._soft_update(self.critic1_target, self.critic1)
                self._soft_update(self.critic2_target, self.critic2)

    def _soft_update(self, target: nn.Module, source: nn.Module):
        for tgt_param, src_param in zip(target.parameters(), source.parameters()):
            tgt_param.data.mul_(1.0 - self.tau)
            tgt_param.data.add_(self.tau * src_param.data)

    def save(self, path: Path):
        payload = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
        }
        torch.save(payload, path)


def make_log_dir(base: Path) -> Path:
    base = base.expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / f"run_{int(Path().stat().st_mtime_ns % 1e12)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_cfg = load_env_config(args.env_config_file)
    base_cfg["max_steps"] = args.max_env_steps

    sample_env = PredPreyGrass(dict(base_cfg))
    obs_dim = sample_env.observation_spaces["predator_0"].shape[0]
    act_space = sample_env.action_spaces["predator_0"]
    prey_act_space = sample_env.action_spaces["prey_0"]
    act_dim = act_space.shape[0]
    act_limit = float(np.max(np.abs(act_space.high)))
    prey_low = prey_act_space.low
    prey_high = prey_act_space.high
    zero_obs = np.zeros(obs_dim, dtype=np.float32)
    sample_env.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buffer = ReplayBuffer(args.replay_capacity, obs_dim, act_dim)
    agent = TD3Agent(obs_dim, act_dim, act_limit, args, device)

    log_dir = make_log_dir(args.log_dir)
    serializable_args = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
    }
    (log_dir / "config.json").write_text(json.dumps(serializable_args, indent=2))

    env = PredPreyGrass(dict(base_cfg))
    noise_scale = args.exploration_noise

    for it in range(1, args.num_iterations + 1):
        obs, _ = env.reset(seed=args.seed + it)
        ep_reward = 0.0
        ep_steps = 0
        while ep_steps < args.max_env_steps:
            actions: Dict[str, np.ndarray] = {}
            stored: List[Tuple[str, np.ndarray, np.ndarray]] = []
            for agent_id, agent_obs in obs.items():
                if "predator" in agent_id:
                    action = agent.act(agent_obs, noise_scale)
                    actions[agent_id] = action
                    stored.append((agent_id, agent_obs, action))
                else:
                    actions[agent_id] = np.random.uniform(prey_low, prey_high).astype(
                        np.float32
                    )

            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            for agent_id, agent_obs, act in stored:
                reward = rewards.get(agent_id, 0.0)
                done = terminations.get(agent_id, False) or truncations.get(
                    agent_id, False
                )
                nxt = next_obs.get(agent_id, zero_obs)
                buffer.add(agent_obs, act, reward, nxt, done)
                ep_reward += reward

            if len(buffer) > args.warmup_steps:
                agent.update(buffer, args.batch_size, args.updates_per_step)

            obs = next_obs
            ep_steps += 1
            if terminations.get("__all__", False) or truncations.get("__all__", False):
                break

        noise_scale = max(args.min_noise, noise_scale * args.noise_decay)
        print(
            f"[TD3] Iteration {it:04d} | steps={ep_steps:4d} | reward={ep_reward:8.2f} | noise={noise_scale:.4f}"
        )

        if it % args.checkpoint_freq == 0:
            agent.save(log_dir / f"checkpoint_{it:06d}.pt")

    agent.save(log_dir / "checkpoint_final.pt")
    env.close()


def main() -> int:
    args = build_parser().parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
