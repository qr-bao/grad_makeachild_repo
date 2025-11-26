"""Helpers to build RLModule specs for env129 policies."""

from __future__ import annotations

from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec


DEFAULT_MODEL_CONFIG = {
    "fcnet_hiddens": [256, 256, 128],
    "fcnet_activation": "relu",
    "vf_share_layers": False,
}


def build_module_spec(obs_space, act_space, model_config: dict | None = None) -> RLModuleSpec:
    """Create a :class:`RLModuleSpec` for the provided spaces.

    The env129 observations are vector-based so a fully-connected torso is
    sufficient; callers may override hidden sizes via ``model_config``.
    """

    cfg = {**DEFAULT_MODEL_CONFIG, **(model_config or {})}
    return RLModuleSpec(
        module_class=DefaultPPOTorchRLModule,
        observation_space=obs_space,
        action_space=act_space,
        inference_only=False,
        model_config=cfg,
        catalog_class=None,
    )


def build_multi_module_spec(
    obs_spaces_by_policy: dict,
    act_spaces_by_policy: dict,
    model_config: dict | None = None,
) -> MultiRLModuleSpec:
    """Build a MultiRLModuleSpec for arbitrary policy dictionaries."""

    rl_module_specs = {}
    for policy_id, obs_space in obs_spaces_by_policy.items():
        rl_module_specs[policy_id] = build_module_spec(
            obs_space,
            act_spaces_by_policy[policy_id],
            model_config=model_config,
        )
    return MultiRLModuleSpec(rl_module_specs=rl_module_specs)
