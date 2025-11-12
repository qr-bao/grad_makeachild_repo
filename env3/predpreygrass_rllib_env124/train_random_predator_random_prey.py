"""
Entry point for Random-vs-Random training runs.

Predators and prey both use the Random strategy by default, but the shared helper
still exposes ``--predator-strategy``/``--prey-strategy`` if manual overrides are
needed when launching this script.
"""

from predpreygrass.rllib.env3.predpreygrass_rllib_env124.train_strategy_common import (
    launch_strategy_training,
)


if __name__ == "__main__":
    launch_strategy_training(
        default_predator_strategy="random",
        default_prey_strategy="random",
        experiment_name="PredPreyGrass_randomPred_randomPrey",
    )

