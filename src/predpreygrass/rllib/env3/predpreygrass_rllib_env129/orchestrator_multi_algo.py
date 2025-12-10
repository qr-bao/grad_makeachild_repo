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
from pathlib import Path
from typing import Dict, List

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.distributions as D
from torch.utils.tensorboard import SummaryWriter
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


class SimpleTD3:
    def __init__(self, obs_dim, act_dim, act_low, act_high, device="cpu"):
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
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=3e-4)
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.1
        self.noise_clip = 0.2
        self.policy_delay = 2
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

    def act(self, obs_np):
        obs = torch.as_tensor(obs_np, device=self.device, dtype=torch.float32)
        mean = self.actor(obs)
        std = self.log_std.exp()
        dist = D.Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1).item()
        value = self.value(obs).item()
        # scale to bounds
        action = action * (self.act_high - self.act_low) / 2 + (self.act_high + self.act_low) / 2
        action = torch.max(torch.min(action, self.act_high), self.act_low)
        return action.cpu().numpy(), logp, value

    def train_batch(self, batch):
        obs = torch.as_tensor(batch["obs"], device=self.device, dtype=torch.float32)
        acts = torch.as_tensor(batch["actions"], device=self.device, dtype=torch.float32)
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
        logp = dist.log_prob(acts).sum(-1, keepdim=True)
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Multi-algo orchestrator (PPO train + Random/SAC infer)")
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--mapping-json", type=str, default=None)
    parser.add_argument("--env-config-file", type=str, default=None)
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
    td3_models: dict,
    a2c_models: dict,
    off_buffers: dict,
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
    # Save custom TD3/A2C
    for name, model in td3_models.items():
        path = checkpoint_dir / f"{name}_ep{ep+1}.pt"
        torch.save(
            {
                "actor": model.actor.state_dict(),
                "actor_target": model.actor_target.state_dict(),
                "q1": model.q1.state_dict(),
                "q2": model.q2.state_dict(),
                "q1_target": model.q1_target.state_dict(),
                "q2_target": model.q2_target.state_dict(),
                "act_low": model.act_low.cpu(),
                "act_high": model.act_high.cpu(),
                "buffer": off_buffers.get(name, []),
            },
            path,
        )
    for name, model in a2c_models.items():
        path = checkpoint_dir / f"{name}_ep{ep+1}.pt"
        torch.save(
            {
                "actor": model.actor.state_dict(),
                "value": model.value.state_dict(),
                "log_std": model.log_std.detach().cpu(),
            },
            path,
        )


def orchestrate(
    algo_map: Dict[str, str],
    env_config: dict,
    num_episodes: int,
    max_steps: int,
    log_dir: Path,
    checkpoint_dir: Path,
    checkpoint_freq: int,
):
    env = PredPreyGrass(env_config)
    algos = build_algorithms(algo_map, env_config)
    policies = {name: algo.get_policy() for name, algo in algos.items() if algo is not None}
    td3_models = {}
    a2c_models = {}
    off_buffers: Dict[str, List[dict]] = {
        name: []
        for name, algo in algos.items()
        if _algo_base_name(name) in ("sac", "td3")
    }
    off_max_buffer = int(5e4)
    writer = SummaryWriter(log_dir=str(log_dir))
    agent_reward_sums: Dict[str, float] = {}
    agent_meta: Dict[str, Dict] = {}
    action_norms: Dict[str, List[float]] = {}
    loss_stats: Dict[str, List[dict]] = {}

    for ep in range(num_episodes):
        obs, infos = env.reset()
        traj: Dict[str, Dict[str, List[dict]]] = {}
        ep_reward_per_algo: Dict[str, float] = {}
        agent_reward_sums.clear()
        agent_meta.clear()
        action_norms.clear()
        loss_stats = {}
        for step in range(max_steps):
            actions = {}
            extra_cache = {}
            for aid, ob in obs.items():
                meta = _enrich_info(aid, infos.get(aid, {}), env)
                algo_name = _assign_algo(aid, meta, algo_map)
                algo_key = _algo_instance_key(algo_name, meta)
                base = _algo_base_name(algo_key)
                if base == "random":
                    act = env.action_spaces[aid].sample()
                    logp = 0.0
                    vf = 0.0
                elif base == "a2c":
                    if algo_key not in a2c_models:
                        act_low = env.action_spaces[aid].low
                        act_high = env.action_spaces[aid].high
                        a2c_models[algo_key] = SimpleA2C(
                            obs_dim=ob.shape[0],
                            act_dim=env.action_spaces[aid].shape[0],
                            act_low=act_low,
                            act_high=act_high,
                        )
                    act, logp, vf = a2c_models[algo_key].act(ob)
                elif base == "td3":
                    if algo_key not in td3_models:
                        act_low = env.action_spaces[aid].low
                        act_high = env.action_spaces[aid].high
                        td3_models[algo_key] = SimpleTD3(
                            obs_dim=ob.shape[0],
                            act_dim=env.action_spaces[aid].shape[0],
                            act_low=act_low,
                            act_high=act_high,
                        )
                    act = td3_models[algo_key].compute_action(ob, noise_std=0.1)
                    logp = 0.0
                    vf = 0.0
                else:
                    pol = policies[algo_key]
                    act, _, extra = pol.compute_single_action(ob, explore=True)
                    logp = extra.get("action_logp", 0.0)
                    vf = extra.get("vf_preds", 0.0)
                actions[aid] = act
                extra_cache[aid] = {"logp": logp, "vf": vf, "algo_key": algo_key}
                try:
                    action_norms.setdefault(algo_key, []).append(float(np.linalg.norm(act)))
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
                    # Newborn agents won't have pre-step obs/action; skip logging this transition.
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
                    "t": len(buf),
                }
                buf.append(transition)
                ep_reward_per_algo[algo_key] = ep_reward_per_algo.get(algo_key, 0.0) + rew
                agent_reward_sums[aid] = agent_reward_sums.get(aid, 0.0) + rew
                if algo_name in ("sac", "td3"):
                    off_buffers.setdefault(algo_key, []).append(transition)
                    if len(off_buffers[algo_key]) > off_max_buffer:
                        off_buffers[algo_key] = off_buffers[algo_key][-off_max_buffer:]

            obs = next_obs
            if dones.get("__all__") or truncs.get("__all__"):
                break

        # Training (PPO/SAC via Ray, TD3 via custom)
        for algo_key, agents_steps in traj.items():
            base = _algo_base_name(algo_key)
            if base == "random":
                continue
            if base == "ppo":
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
                    acts, _, extra = policy.compute_actions_from_input_dict(
                        {SampleBatch.OBS: batch[SampleBatch.OBS]}, explore=False
                    )
                    if SampleBatch.ACTION_DIST_INPUTS in extra:
                        batch[SampleBatch.ACTION_DIST_INPUTS] = np.array(extra[SampleBatch.ACTION_DIST_INPUTS])
                    if SampleBatch.VF_PREDS in extra:
                        batch[SampleBatch.VF_PREDS] = np.array(extra[SampleBatch.VF_PREDS])
                    post = policy.postprocess_trajectory(batch, other_agent_batches=None, episode=None)
                    policy.learn_on_batch(post)
            elif base == "sac":
                policy = policies.get(algo_key)
                buffer = off_buffers.get(algo_key, [])
                batch_size = policy.config.get("train_batch_size", 64)
                warmup = policy.config.get("num_steps_sampled_before_learning_starts", policy.config.get("learning_starts", 0))
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
                for steps in agents_steps.values():
                    if not steps:
                        continue
                    mb = {
                        "obs": np.stack([s["obs"] for s in steps]).astype(np.float32),
                        "actions": np.stack([s["actions"] for s in steps]).astype(np.float32),
                        "rewards": np.array([s["rewards"] for s in steps], dtype=np.float32),
                        "dones": np.array([s["dones"] for s in steps], dtype=np.float32),
                        "next_obs": np.stack([s["next_obs"] for s in steps]).astype(np.float32),
                    }
                    stat = model.train_batch(mb)
                    if stat:
                        loss_stats.setdefault(algo_key, []).append(stat)
            elif base == "td3":
                model = td3_models.get(algo_key)
                buffer = off_buffers.get(algo_key, [])
                batch_size = 64
                warmup = 32
                if model and len(buffer) >= max(batch_size, warmup):
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
            algo_key = _algo_instance_key(algo_name, meta)
            bucket.setdefault((species, algo_key), []).append(rsum)
            species_bucket.setdefault(species, []).append(rsum)
        for (species, algo_key), vals in bucket.items():
            writer.add_scalar(f"episode_reward_mean/{species}/{algo_key}", float(np.mean(vals)), ep)
            v = np.array(vals, dtype=np.float32)
            writer.add_scalar(f"episode_reward_p50/{species}/{algo_key}", float(np.median(v)), ep)
            writer.add_scalar(f"episode_reward_p25/{species}/{algo_key}", float(np.percentile(v, 25)), ep)
            writer.add_scalar(f"episode_reward_p75/{species}/{algo_key}", float(np.percentile(v, 75)), ep)
        for species, vals in species_bucket.items():
            writer.add_scalar(f"episode_reward_mean/{species}/ALL", float(np.mean(vals)), ep)
        for algo_key, rew_sum in ep_reward_per_algo.items():
            writer.add_scalar(f"episode_reward_sum/{algo_key}", rew_sum, ep)
        writer.add_scalar("episode_len", step + 1, ep)
        # Action norms
        for algo_key, vals in action_norms.items():
            if vals:
                writer.add_scalar(f"action_norm_mean/{algo_key}", float(np.mean(vals)), ep)
        # Population counts
        pop_counts = getattr(env, "population_counts", {})
        if pop_counts:
            pred_total = sum(v for k, v in pop_counts.items() if k.startswith("predator_"))
            prey_total = sum(v for k, v in pop_counts.items() if k.startswith("prey_"))
            writer.add_scalar("population/predator/total", pred_total, ep)
            writer.add_scalar("population/prey/total", prey_total, ep)
            for key, val in pop_counts.items():
                if key.startswith("predator_"):
                    writer.add_scalar(f"population/predator/{key}", val, ep)
                elif key.startswith("prey_"):
                    writer.add_scalar(f"population/prey/{key}", val, ep)
        # Loss diagnostics
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

        print(f"Episode {ep} done at step {step+1}")
        _maybe_save_checkpoints(checkpoint_dir, ep, algos, td3_models, a2c_models, off_buffers, checkpoint_freq)

    for name, algo in algos.items():
        if algo is not None and hasattr(algo, "stop"):
            algo.stop()
    env.close()
    writer.close()


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

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    orchestrate(
        mapping,
        env_config,
        args.num_episodes,
        args.max_steps,
        log_dir,
        checkpoint_dir,
        args.checkpoint_freq,
    )
    ray.shutdown()


if __name__ == "__main__":
    main()
