#!/usr/bin/env python3
"""
可视化已训练策略的脚本。
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import Dict
import numpy as np
import torch  # ← 新增导入
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.rllib.core.columns import Columns  # ← 新增导入

from predpreygrass.rllib.env3.predpreygrass_rllib_env124.predpreygrass_rllib_env import (
    PredPreyGrass,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env124.visualizer import (
    PredPreyVisualizer,
)

def infer_policy_id(agent_id: str) -> str:
    """与训练时一致的策略映射规则。"""
    return "predator_policy" if "predator" in agent_id else "prey_policy"

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualise a trained PredPreyGrass policy checkpoint.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="RLlib checkpoint directory (e.g. .../checkpoint_000020)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20000,
        help="Maximum simulation steps before exiting (default: 20000)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target visualisation frame rate (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional environment reset seed (default: 42)",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Enable exploration when computing actions (default: disabled)",
    )
    return parser

def main() -> None:
    args = build_argument_parser().parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    
    # 注册训练阶段使用的环境名称
    def _env_creator(env_config):
        return PredPreyGrass(dict(env_config))
    
    for env_name in ("PredPreyGrass-continuous", "PredPreyGrass-discrete"):
        try:
            register_env(env_name, _env_creator)
        except Exception:
            pass
    
    algo = Algorithm.from_checkpoint(str(checkpoint_path))
    
    # 获取 RLModule 实例
    rl_modules = {}
    try:
        rl_modules["predator_policy"] = algo.get_module("predator_policy")
        rl_modules["prey_policy"] = algo.get_module("prey_policy")
    except Exception as e:
        print(f"[WARNING] Could not get RLModules: {e}")
        print("[INFO] Trying alternative method...")
        try:
            rl_modules["predator_policy"] = algo.env_runner.module["predator_policy"]
            rl_modules["prey_policy"] = algo.env_runner.module["prey_policy"]
        except Exception as e2:
            raise RuntimeError(f"Failed to get RLModules: {e2}")
    
    # ============ 新增：确定设备 ============
    # 检查模型在哪个设备上
    device = None
    for policy_id, rl_module in rl_modules.items():
        # 尝试获取模型的设备
        try:
            # 获取模型的第一个参数来确定设备
            device = next(rl_module.parameters()).device
            print(f"[INFO] Model device: {device}")
            break
        except Exception:
            device = torch.device("cpu")
    
    if device is None:
        device = torch.device("cpu")
    # ======================================
    
    env_config = dict(algo.config.get("env_config", {}))
    env_config["debug_logging"] = False
    env = PredPreyGrass(env_config)
    
    latest_obs: Dict[str, np.ndarray] = {}
    
    original_reset = env.reset
    def reset_and_capture(*reset_args, **reset_kwargs):
        observations, info = original_reset(*reset_args, **reset_kwargs)
        latest_obs.clear()
        latest_obs.update(observations)
        return observations, info
    
    env.reset = reset_and_capture  # type: ignore[assignment]
    observations, _ = env.reset(seed=args.seed)
    latest_obs.update(observations)
    
    visualizer = PredPreyVisualizer(env, fps=args.fps)
    step_count = 0
    frame_interval = 1.0 / max(args.fps, 1)
    last_frame_time = time.time()
    
    try:
        while visualizer.render() and step_count < args.max_steps:
            if visualizer.paused:
                time.sleep(0.01)
                continue
            
            now = time.time()
            elapsed = now - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.time()
            
            actions: Dict[str, np.ndarray] = {}
            
            for agent_id in list(env.agents):
                obs_vector = latest_obs.get(agent_id)
                if obs_vector is None:
                    continue
                
                policy_id = infer_policy_id(agent_id)
                rl_module = rl_modules[policy_id]
                
                # ============ 转换为 PyTorch tensor ============
                # 1. 添加 batch 维度
                obs_batch = np.expand_dims(obs_vector, axis=0)
                
                # 2. 转换为 torch tensor 并移到正确的设备
                obs_tensor = torch.from_numpy(obs_batch).float().to(device)
                
                # 3. 使用正确的输入格式（使用 Columns.OBS 键）
                input_dict = {Columns.OBS: obs_tensor}
                
                # 4. 进行推理
                with torch.no_grad():  # 推理时不需要计算梯度
                    output = rl_module.forward_inference(input_dict)
                
                # 5. 提取动作并转换回 numpy
                if "actions" in output:
                    action_tensor = output["actions"]
                elif Columns.ACTIONS in output:
                    action_tensor = output[Columns.ACTIONS]
                elif "action_dist_inputs" in output:
                    action_tensor = output["action_dist_inputs"]
                else:
                    raise ValueError(f"Could not find actions in output keys: {output.keys()}")
                
                # 转换为 numpy 并移除 batch 维度
                action = action_tensor.cpu().numpy()[0]
                # =============================================
                
                actions[agent_id] = np.asarray(action, dtype=np.float32)
            
            new_obs, rewards, terminations, truncations, infos = env.step(actions)
            
            for agent_id, obs_vector in new_obs.items():
                latest_obs[agent_id] = obs_vector
            
            # 移除离开环境的智能体
            for agent_id in list(latest_obs.keys()):
                if agent_id not in env.agents:
                    latest_obs.pop(agent_id, None)
            
            step_count += 1
            
            if truncations.get("__all__", False) or terminations.get("__all__", False):
                break
        
        print(f"[INFO] Simulation finished after {step_count} steps.")
    
    finally:
        visualizer.close()
        algo.stop()
        ray.shutdown()

if __name__ == "__main__":
    main()
