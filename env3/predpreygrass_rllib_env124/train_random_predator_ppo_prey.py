"""
Entry point where predators stay Random while prey learn with PPO.
"""

from predpreygrass.rllib.env3.predpreygrass_rllib_env124.train_strategy_common import (
    launch_strategy_training,
)


if __name__ == "__main__":
    launch_strategy_training(
        default_predator_strategy="random",
        default_prey_strategy="ppo",
        experiment_name="PredPreyGrass_randomPred_ppoPrey",
    )

