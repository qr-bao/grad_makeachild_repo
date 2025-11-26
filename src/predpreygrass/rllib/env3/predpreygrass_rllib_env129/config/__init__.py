"""Config shortcuts for PredPreyGrass env129."""

from .config_env_base import config_env_base
from .config_env_train import config_env as config_env_train
from .config_env_eval import config_env as config_env_eval

__all__ = [
    "config_env_base",
    "config_env_train",
    "config_env_eval",
]
