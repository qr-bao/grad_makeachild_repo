"""
离线评估 checkpoint（无 Ray、无渲染）。
流程：
  - 从 checkpoint_*/policies/<policy>/policy_state.pkl 读出权重
  - 重建同结构的 MLP + DiagGaussian，支持 deterministic / stochastic
  - 逐个 checkpoint 跑若干 episode，统计 predator/prey/grass 数量与平均能量
  - 输出 CSV 并生成 2x3 曲线图

示例（在 env3 目录）:
PYTHONPATH=../../.. python predpreygrass_rllib_env129/evaluate_checkpoints.py \
  --trial-dir logs/run_20251124_204510/ray_results/PPO_PredPreyGrass_continuous_simple/PPO_PredPreyGrass-continuous_09900_00000_0_2025-11-24_20-45-13 \
  --env-config predpreygrass_rllib_env129/config/config_env_base.json \
  --episodes 3 --eval-max-steps 2000 --stochastic-policy \
  --output-csv eval_metrics.csv --output-fig eval_metrics.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from predpreygrass.rllib.env3.predpreygrass_rllib_env129.predpreygrass_rllib_env import (
    PredPreyGrass,
)


def policy_mapping_fn(agent_id: str) -> str | None:
    if "predator" in agent_id:
        return "predator_policy"
    if "prey" in agent_id:
        return "prey_policy"
    return None


def list_checkpoints(trial_dir: Path) -> List[Path]:
    ckpts = sorted(trial_dir.glob("checkpoint_*"), key=lambda p: p.name)
    return [
        p
        for p in ckpts
        if (p / "policies" / "predator_policy" / "policy_state.pkl").exists()
        and (p / "policies" / "prey_policy" / "policy_state.pkl").exists()
    ]


def load_env_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fp:
        if path.suffix.lower() == ".json":
            return json.load(fp)
        scope: Dict = {}
        exec(path.read_text(), scope)
        return scope.get("config_env_base") or scope.get("config") or {}


class DiagGaussianPolicy(nn.Module):
    """简易三层 MLP + DiagGaussian."""

    def __init__(self, obs_dim: int, hidden: List[int], act_dim: int, weights: Dict[str, np.ndarray]):
        super().__init__()
        h1, h2, h3 = hidden
        self.h1 = nn.Linear(obs_dim, h1)
        self.h2 = nn.Linear(h1, h2)
        self.h3 = nn.Linear(h2, h3)
        self.logits = nn.Linear(h3, act_dim * 2)  # mean + log_std
        self.relu = nn.ReLU()
        self.load_weights(weights)

    def load_weights(self, w: Dict[str, np.ndarray]):
        def set_(layer: nn.Linear, kw, kb):
            layer.weight.data = torch.tensor(w[kw])
            layer.bias.data = torch.tensor(w[kb])

        set_(self.h1, "_hidden_layers.0._model.0.weight", "_hidden_layers.0._model.0.bias")
        set_(self.h2, "_hidden_layers.1._model.0.weight", "_hidden_layers.1._model.0.bias")
        set_(self.h3, "_hidden_layers.2._model.0.weight", "_hidden_layers.2._model.0.bias")
        set_(self.logits, "_logits._model.0.weight", "_logits._model.0.bias")

    @torch.no_grad()
    def act(self, obs: torch.Tensor, stochastic: bool, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.h1(obs))
        x = self.relu(self.h2(x))
        x = self.relu(self.h3(x))
        out = self.logits(x)
        act_dim = out.shape[-1] // 2
        mean, log_std = out[..., :act_dim], out[..., act_dim:]
        if stochastic:
            std = torch.exp(log_std).clamp(min=1e-6, max=5.0)
            action = torch.normal(mean, std)
        else:
            action = mean
        return torch.max(torch.min(action, high), low)


def build_policy(policy_state_path: Path) -> DiagGaussianPolicy:
    import pickle

    state = pickle.load(open(policy_state_path, "rb"))
    w = state["weights"]
    obs_dim = w["_hidden_layers.0._model.0.weight"].shape[1]
    hidden = [
        w["_hidden_layers.0._model.0.weight"].shape[0],
        w["_hidden_layers.1._model.0.weight"].shape[0],
        w["_hidden_layers.2._model.0.weight"].shape[0],
    ]
    act_dim = w["_logits._model.0.weight"].shape[0] // 2
    return DiagGaussianPolicy(obs_dim, hidden, act_dim, w).eval()


def evaluate_checkpoint_local(
    ckpt: Path, env_cfg: Dict, episodes: int, max_steps: int, stochastic: bool
) -> Dict:
    pred_policy = build_policy(ckpt / "policies" / "predator_policy" / "policy_state.pkl")
    prey_policy = build_policy(ckpt / "policies" / "prey_policy" / "policy_state.pkl")
    torch.set_grad_enabled(False)

    results = []
    for _ in range(episodes):
        cfg = env_cfg.copy()
        if max_steps > 0:
            cfg["max_steps"] = max_steps
        env = PredPreyGrass(cfg)
        obs, _ = env.reset()
        step = 0
        done = False
        while not done and step < max_steps:
            actions = {}
            for aid, ob in obs.items():
                pol_id = policy_mapping_fn(aid)
                if pol_id is None:
                    continue
                policy = pred_policy if pol_id == "predator_policy" else prey_policy
                ob_t = torch.tensor(ob, dtype=torch.float32)
                low = torch.tensor(env.action_spaces[aid].low, dtype=torch.float32)
                high = torch.tensor(env.action_spaces[aid].high, dtype=torch.float32)
                act = policy.act(ob_t, stochastic=stochastic, low=low, high=high).numpy()
                actions[aid] = act
            obs, _r, term, trunc, _info = env.step(actions)
            done = term.get("__all__", False) or trunc.get("__all__", False)
            step += 1

        predators = [a for a in env.agents if "predator" in a]
        preys = [a for a in env.agents if "prey" in a]
        predator_energy = [env.agent_energies.get(a, 0.0) for a in predators]
        prey_energy = [env.agent_energies.get(a, 0.0) for a in preys]
        grass_energy = list(env.grass_energies.values())
        results.append(
            {
                "predator_count": len(predators),
                "prey_count": len(preys),
                "grass_count": len(env.grass_positions),
                "predator_energy_mean": float(np.mean(predator_energy)) if predator_energy else 0.0,
                "prey_energy_mean": float(np.mean(prey_energy)) if prey_energy else 0.0,
                "grass_energy_mean": float(np.mean(grass_energy)) if grass_energy else 0.0,
            }
        )
        env.close()

    agg = {k: float(np.mean([r[k] for r in results])) for k in results[0].keys()}
    agg["episodes"] = episodes
    return agg


def plot_metrics(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    xs = df["iteration"]
    axes = axes.ravel()
    pairs = [
        ("predator_count", "Predator Count"),
        ("prey_count", "Prey Count"),
        ("grass_count", "Grass Count"),
        ("predator_energy_mean", "Predator Avg Energy"),
        ("prey_energy_mean", "Prey Avg Energy"),
        ("grass_energy_mean", "Grass Avg Energy"),
    ]
    for ax, (col, title) in zip(axes, pairs):
        ax.plot(xs, df[col], marker="o")
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints locally (no Ray)")
    parser.add_argument("--trial-dir", required=True)
    parser.add_argument(
        "--env-config",
        default="predpreygrass_rllib_env129/config/config_env_base.json",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--eval-max-steps", type=int, default=2000)
    parser.add_argument("--stochastic-policy", action="store_true")
    parser.add_argument("--output-csv", default="eval_metrics.csv")
    parser.add_argument("--output-fig", default="eval_metrics.png")
    args = parser.parse_args()

    trial_dir = Path(args.trial_dir).expanduser().resolve()
    env_cfg = load_env_config(Path(args.env_config).expanduser().resolve())

    checkpoints = list_checkpoints(trial_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {trial_dir}")

    rows = []
    for ckpt in checkpoints:
        m = re.search(r"checkpoint_(\d+)$", ckpt.name)
        iteration = int(m.group(1)) if m else len(rows)
        agg = evaluate_checkpoint_local(
            ckpt, env_cfg, episodes=args.episodes, max_steps=args.eval_max_steps, stochastic=args.stochastic_policy
        )
        agg.update({"iteration": iteration, "checkpoint": ckpt.name})
        rows.append(agg)
        print(f"[EVAL] {ckpt.name}: {agg}")

    df = pd.DataFrame(rows).sort_values("iteration")
    out_csv = trial_dir / args.output_csv
    df.to_csv(out_csv, index=False)
    print(f"Saved metrics to {out_csv}")

    out_fig = trial_dir / args.output_fig
    plot_metrics(df, out_fig)
    print(f"Saved figure to {out_fig}")


if __name__ == "__main__":
    main()

