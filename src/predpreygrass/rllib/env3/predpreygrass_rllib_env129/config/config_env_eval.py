"""Evaluation configuration with slightly longer episodes and logging enabled."""

from .config_env_train import config_env as training_env

config_env = {
    **training_env,
    "max_steps": 6000,
    "debug_logging": True,
    "allow_empty_predator_population": True,
}
