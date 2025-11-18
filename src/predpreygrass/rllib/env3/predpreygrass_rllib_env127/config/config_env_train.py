"""Training environment configuration built on top of :mod:`config_env_base`."""

from .config_env_base import config_env_base

config_env = {
    **config_env_base,
    # Keep training runs deterministic unless overridden by CLI.
    "seed": 42,
    # Enable debug logging toggles to be overwritten by scripts; default off.
    "debug_logging": False,
    "debug_log_level": "INFO",
}
