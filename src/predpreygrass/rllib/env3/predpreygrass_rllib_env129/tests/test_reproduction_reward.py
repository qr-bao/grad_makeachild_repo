"""
Step 3: reproduction 奖励 / 后代生成验证。
"""
import pytest

from predpreygrass.rllib.env3.predpreygrass_rllib_env129.predpreygrass_rllib_env import (
    PredPreyGrass,
)


def _repro_test_config() -> dict:
    """提供一个最小离散环境，允许两只捕食者立即繁殖。"""
    return {
        "enable_continuous_space": False,
        "grid_size": 3,
        "num_obs_channels": 4,
        "predator_obs_range": 3,
        "prey_obs_range": 3,
        "n_populations": 1,
        "n_possible_predators": 4,
        "n_possible_prey": 0,
        "n_initial_active_predator": 2,
        "n_initial_active_prey": 0,
        "initial_num_grass": 0,
        "initial_energy_predator": 50.0,
        "reward_predator_step": 0.0,
        "reward_predator_catch_prey": 0.0,
        "reproduction_reward_predator": 7.5,
        "enable_paired_reproduction": True,
        "predator_creation_energy_threshold": 5.0,
        "reproduction_mode_predator": "fixed_ratio",
        "reproduction_fixed_cost_predator": 0.0,
        "reproduction_transfer_ratio_predator": 0.0,
        "offspring_min_energy_predator": 20.0,
        "min_reproduction_age_predator": 0,
        "max_reproduction_age_predator": 10_000,
        "reproduction_cooldown_predator": 0,
        "max_population_size_predator": 5,
        "energy_loss_per_step_predator": 0.0,
        "base_metabolism_predator": 0.0,
        "movement_cost_factor_predator": 0.0,
        "enable_hunger": False,
        "max_steps": 5,
    }


def test_reproduction_rewards_and_spawns_offspring():
    env = PredPreyGrass(_repro_test_config())
    env.reset(seed=123)

    parent_a = "predator_0"
    parent_b = "predator_1"

    # 将两只捕食者放在相邻网格，以满足离散模式下的配对距离判定。
    env.agent_positions[parent_a] = (0, 0)
    env.predator_positions[parent_a] = (0, 0)
    env.agent_positions[parent_b] = (0, 1)
    env.predator_positions[parent_b] = (0, 1)

    actions = {parent_a: 4, parent_b: 4}  # 原地不动
    _, rewards, _, _, _ = env.step(actions)

    offspring_id = "predator_2"
    assert offspring_id in env.agents, "未生成新的捕食者后代"
    assert env.agent_generation[offspring_id] == 1
    assert env.agent_population_id[offspring_id] == env.agent_population_id[parent_a]
    assert env.agent_energies[offspring_id] == pytest.approx(20.0)

    expected_reward = env.reproduction_reward_predator
    assert rewards.get(parent_a) == pytest.approx(expected_reward)
    assert rewards.get(parent_b) == pytest.approx(expected_reward)

    env.close()
