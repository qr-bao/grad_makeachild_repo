"""
Entry point where predators learn with PPO and prey remain Random.
"""

from predpreygrass.rllib.env3.predpreygrass_rllib_env124.train_strategy_common import (
    launch_strategy_training,
)


if __name__ == "__main__":
    launch_strategy_training(
        default_predator_strategy="ppo",
        default_prey_strategy="random",
        experiment_name="PredPreyGrass_ppoPred_randomPrey",
    )

