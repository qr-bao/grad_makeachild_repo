"""
Step 7: PPO smoke test to ensure RLlib can roll out the env once.
"""
import socket

import pytest
from packaging.version import Version

from predpreygrass.rllib.env3.predpreygrass_rllib_env127.predpreygrass_rllib_env import (
    PredPreyGrass,
)

try:  # noqa: SIM105 - explicit import gate keeps test skip-friendly
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - if torch missing we skip
    torch = None

ray = pytest.importorskip("ray")
ppo_module = pytest.importorskip("ray.rllib.algorithms.ppo")
PPOConfig = ppo_module.PPOConfig
_RAY_VERSION = Version(ray.__version__)


def _can_open_udp_socket() -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.close()
        return True
    except OSError:
        return False


_CAN_INIT_RAY = _can_open_udp_socket()


def _env_config() -> dict:
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
        "initial_energy_predator": 5.0,
        "initial_energy_prey": 5.0,
        "energy_loss_per_step_predator": 0.0,
        "energy_loss_per_step_prey": 0.0,
        "base_metabolism": 0.0,
        "movement_cost_factor": 0.0,
        "enable_hunger": False,
        "reward_predator_catch_prey": 0.5,
        "reward_prey_step": 0.1,
        "max_steps": 5,
    }


@pytest.mark.skipif(torch is None, reason="PyTorch not installed for RLlib PPO test")
@pytest.mark.skipif(
    _RAY_VERSION >= Version("2.5.0"),
    reason="Ray>=2.5 defaults to the new API stack; run train_simple.py for the full multi-agent configuration",
)
@pytest.mark.skipif(not _CAN_INIT_RAY, reason="UDP sockets disabled; cannot start Ray locally")
def test_ppo_trainer_runs_single_iteration():
    ray.init(local_mode=True, include_dashboard=False, ignore_reinit_error=True)
    try:
        config = (
            PPOConfig()
            .environment(env=PredPreyGrass, env_config=_env_config())
            .env_runners(num_env_runners=0)
            .framework("torch")
            .debugging(seed=1234)
            .resources(num_gpus=0)
        )

        algo = config.build()
        try:
            result = algo.train()
        finally:
            algo.cleanup()

        assert "episode_reward_mean" in result
        assert result.get("episodes_this_iter", 0) > 0
    finally:
        ray.shutdown()
