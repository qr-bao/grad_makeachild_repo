"""
Step 6: 饥饿清除行为不应被重复触发。
"""
import numpy as np

from predpreygrass.rllib.env3.predpreygrass_rllib_env127.predpreygrass_rllib_env import (
    PredPreyGrass,
)


def _hunger_config() -> dict:
    """Minimal continuous-space configuration where hunger drives removal."""
    return {
        "enable_continuous_space": True,
        "world_width": 50,
        "world_height": 50,
        "n_possible_predators": 1,
        "n_possible_prey": 0,
        "n_initial_active_predator": 1,
        "n_initial_active_prey": 0,
        "initial_num_grass": 0,
        "initial_energy_predator": 5.0,
        "energy_loss_per_step_predator": 0.0,
        "base_metabolism_predator": 0.0,
        "movement_cost_factor_predator": 0.0,
        "thrust_cost_factor_predator": 0.0,
        "drag_coefficient": 0.0,
        "enable_hunger": True,
        "hunger_damage": 10.0,
        "max_steps_without_food_predator": 1,
        "max_steps": 5,
    }


def test_starvation_removal_occurs_once():
    env = PredPreyGrass(_hunger_config())
    try:
        env.reset(seed=2024)
        predator_id = "predator_0"
        idle_action = np.zeros(2, dtype=np.float32)

        # First step only increments hunger counter; predator is still alive.
        env.step({predator_id: idle_action})
        assert predator_id in env.agents

        # Second step crosses the hunger threshold and should remove the predator exactly once.
        env.step({predator_id: idle_action})

        assert predator_id not in env.agents
        assert predator_id not in env.agent_positions
        assert predator_id not in env.agent_energies
        assert predator_id not in env.agent_steps_since_last_meal
        assert env.current_num_predators == 0
        assert env.retired_agents == {predator_id}

        # An extra no-op step should keep populations stable (no duplicate removals logged).
        env.step({})
        assert env.current_num_predators == 0
        assert env.retired_agents == {predator_id}
    finally:
        env.close()
