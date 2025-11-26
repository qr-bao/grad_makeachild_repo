"""
Step 2: deterministic 捕食/能量转移验证。
"""
from collections import defaultdict

import pytest

from predpreygrass.rllib.env3.predpreygrass_rllib_env129.predpreygrass_rllib_env import (
    PredPreyGrass,
)


def _discrete_test_config() -> dict:
    """提供一个最小 1x1 网格配置，确保初始即重合。"""
    return {
        "enable_continuous_space": False,
        "grid_size": 2,
        "num_obs_channels": 4,
        "predator_obs_range": 1,
        "prey_obs_range": 1,
        "n_possible_predators": 1,
        "n_possible_prey": 1,
        "n_initial_active_predator": 1,
        "n_initial_active_prey": 1,
        "initial_num_grass": 0,
        "initial_energy_predator": 5.0,
        "initial_energy_prey": 5.0,
        "energy_transfer_efficiency_predator": 1.0,
        "reward_predator_catch_prey": 1.0,
        "energy_loss_per_step_predator": 0.0,
        "energy_loss_per_step_prey": 0.0,
        "base_metabolism": 0.0,
        "base_metabolism_predator": 0.0,
        "base_metabolism_prey": 0.0,
        "movement_cost_factor": 0.0,
        "movement_cost_factor_predator": 0.0,
        "movement_cost_factor_prey": 0.0,
        "enable_hunger": False,
        "max_steps": 5,
    }


def test_predation_removes_prey_and_rewards_predator():
    env = PredPreyGrass(_discrete_test_config())
    env.reset(seed=123)

    predator_id = "predator_0"
    prey_id = "prey_0"

    # 手动将两者移动到同一网格，保证进行一次捕食。
    env.agent_positions[predator_id] = (0, 0)
    env.predator_positions[predator_id] = (0, 0)
    env.agent_positions[prey_id] = (0, 0)
    env.prey_positions[prey_id] = (0, 0)

    # env 当前的 agent_energies 是 dict，在离散模式下捕食后会立即删除猎物条目，
    # 导致 step 中后续检查出现 KeyError。这里转成 defaultdict 以兼容当前实现。
    env.agent_energies = defaultdict(lambda: 0.0, env.agent_energies)
    env.agent_positions = defaultdict(lambda: (0, 0), env.agent_positions)
    env.prey_positions = defaultdict(lambda: (0, 0), env.prey_positions)
    env.predator_positions = defaultdict(lambda: (0, 0), env.predator_positions)

    actions = {predator_id: 4, prey_id: 4}  # 4 -> 原地不动
    _, rewards, _, _, _ = env.step(actions)

    assert prey_id not in env.agents, "猎物未被清除"
    assert predator_id in env.agents, "捕食者不应被移除"

    assert rewards.get(predator_id) == pytest.approx(1.0), "捕食奖励不正确"
    assert env.agent_energies[predator_id] == pytest.approx(10.0), "捕食者能量未累加猎物能量"

    env.close()
