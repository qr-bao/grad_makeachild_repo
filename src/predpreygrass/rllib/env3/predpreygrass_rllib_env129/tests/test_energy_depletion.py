"""
Step 5: 能量耗尽应立即移除捕食者。
"""
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.predpreygrass_rllib_env import (
    PredPreyGrass,
)


def _energy_drop_config() -> dict:
    return {
        "enable_continuous_space": False,
        "grid_size": 3,
        "num_obs_channels": 4,
        "predator_obs_range": 3,
        "prey_obs_range": 3,
        "n_possible_predators": 1,
        "n_possible_prey": 0,
        "n_initial_active_predator": 1,
        "n_initial_active_prey": 0,
        "initial_num_grass": 0,
        "initial_energy_predator": 0.1,
        "energy_loss_per_step_predator": 1.0,
        "base_metabolism_predator": 0.0,
        "movement_cost_factor_predator": 0.0,
        "enable_hunger": False,
        "max_steps": 5,
    }


def test_predator_removed_when_energy_runs_out():
    env = PredPreyGrass(_energy_drop_config())
    env.reset(seed=0)

    predator_id = "predator_0"
    assert predator_id in env.agents

    env.step({predator_id: 4})

    assert predator_id not in env.agents
    assert predator_id not in env.agent_positions
    assert predator_id not in env.agent_energies
    assert env.current_num_predators == 0

    env.close()
