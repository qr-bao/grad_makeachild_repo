"""
Step 1: basic RLlib compliance check for the env124 variant.
"""
import pytest

from predpreygrass.rllib.env3.predpreygrass_rllib_env127.predpreygrass_rllib_env import (
    PredPreyGrass,
)

try:  # RLlib is optional in some dev setups
    from ray.rllib.env import check_env

    HAS_RLLIB = True
except ImportError:
    HAS_RLLIB = False


def _minimal_config() -> dict:
    """Return a tiny deterministic config to make check_env fast."""
    return {
        "enable_continuous_space": False,  # use the simpler discrete grid for the probe
        "grid_size": 5,
        "num_obs_channels": 4,
        "predator_obs_range": 3,
        "prey_obs_range": 3,
        "n_possible_predators": 2,
        "n_possible_prey": 2,
        "n_initial_active_predator": 1,
        "n_initial_active_prey": 1,
        "initial_num_grass": 0,
        "max_steps": 5,
        "debug_logging": False,
    }


@pytest.mark.skipif(not HAS_RLLIB, reason="RLlib not installed")
def test_rllib_env_compliance():
    env = PredPreyGrass(_minimal_config())
    check_env(env)
