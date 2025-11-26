"""
Step 4: 当捕食者与猎物全部灭绝时，应全局终止。
"""
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.predpreygrass_rllib_env import (
    PredPreyGrass,
)


def _config():
    return {
        "enable_continuous_space": False,
        "grid_size": 3,
        "num_obs_channels": 4,
        "predator_obs_range": 3,
        "prey_obs_range": 3,
        "n_possible_predators": 1,
        "n_possible_prey": 1,
        "n_initial_active_predator": 1,
        "n_initial_active_prey": 1,
        "initial_num_grass": 0,
        "energy_loss_per_step_predator": 0.0,
        "energy_loss_per_step_prey": 0.0,
        "base_metabolism_predator": 0.0,
        "base_metabolism_prey": 0.0,
        "movement_cost_factor_predator": 0.0,
        "movement_cost_factor_prey": 0.0,
        "enable_hunger": False,
        "max_steps": 10,
    }


def test_extinction_of_all_agents():
    env = PredPreyGrass(_config())
    env.reset(seed=21)

    # 直接清空所有捕食者、猎物，以模拟整群灭绝。
    to_remove = list(env.agents)
    for agent in to_remove:
        env.agents.remove(agent)
        env.agent_positions.pop(agent, None)
        env.agent_energies.pop(agent, None)
        if agent.startswith("predator"):
            env.predator_positions.pop(agent, None)
        else:
            env.prey_positions.pop(agent, None)

    env.current_num_predators = 0
    env.current_num_prey = 0

    _, _, terminations, truncations, _ = env.step({})

    assert terminations.get("__all__", False) is True
    assert truncations.get("__all__", False) is False

    env.close()
