#!/usr/bin/env python3
"""Custom DDPG trainer (predators learn, prey acts randomly)."""

from __future__ import annotations

import argparse
import json
import math
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
    p.add_argument("--log-dir", type=Path, default=Path("logs_ddpg_custom"))
    p.add_argument(
        "--env-config-file",
        type=Path,
        default=Path("predpreygrass_rllib_env129/config/config_env_base.json"),
    )
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--replay-capacity", type=int, default=500_000)
    p.add_argument("--warmup-steps", type=int, default=5000)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--exploration-noise", type=float, default=0.1)
    p.add_argument("--noise-decay", type=float, default=0.999)
    p.add_argument("--min-noise", type=float, default=0.01)
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
        indices = np.random.randint(0, max_idx, size=batch_size)
        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.dones[indices],
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


class DDPGAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_limit: float,
        args: argparse.Namespace,
        device: torch.device,
    ):
        self.device = device
        self.gamma = args.gamma
        self.tau = args.tau
        self.actor = Actor(obs_dim, act_dim, act_limit).to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.target_actor = Actor(obs_dim, act_dim, act_limit).to(device)
        self.target_critic = Critic(obs_dim, act_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

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
                target_actions = self.target_actor(next_obs_t)
                target_q = self.target_critic(next_obs_t, target_actions)
                y = reward_t + (1.0 - done_t) * self.gamma * target_q

            current_q = self.critic(obs_t, action_t)
            critic_loss = F.mse_loss(current_q, y)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            actor_actions = self.actor(obs_t)
            actor_loss = -self.critic(obs_t, actor_actions).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self._soft_update(self.target_actor, self.actor)
            self._soft_update(self.target_critic, self.critic)

    def _soft_update(self, target: nn.Module, source: nn.Module):
        for tgt_param, src_param in zip(target.parameters(), source.parameters()):
            tgt_param.data.mul_(1.0 - self.tau)
            tgt_param.data.add_(self.tau * src_param.data)

    def act(self, obs: np.ndarray, noise_scale: float) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action = self.actor(obs_t.unsqueeze(0)).squeeze(0).cpu().numpy()
        self.actor.train()
        if noise_scale > 0.0:
            action += noise_scale * np.random.randn(*action.shape)
        return action

    def save(self, path: Path):
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
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
    agent = DDPGAgent(obs_dim, act_dim, act_limit, args, device)

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
                    action = np.clip(action, act_space.low, act_space.high)
                    actions[agent_id] = action
                    stored.append((agent_id, agent_obs, action))
                else:
                    random_action = np.random.uniform(prey_low, prey_high).astype(np.float32)
                    actions[agent_id] = random_action

            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            for agent_id, agent_obs, act in stored:
                reward = rewards.get(agent_id, 0.0)
                done = terminations.get(agent_id, False) or truncations.get(agent_id, False)
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
            f"[DDPG] Iteration {it:04d} | steps={ep_steps:4d} | reward={ep_reward:8.2f} | noise={noise_scale:.4f}"
        )

        if it % args.checkpoint_freq == 0:
            ckpt_path = log_dir / f"checkpoint_{it:06d}.pt"
            agent.save(ckpt_path)

    agent.save(log_dir / "checkpoint_final.pt")
    env.close()


def main() -> int:
    args = build_parser().parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
