"""Torch-based random RLModule used for inference-only policies."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import torch
import tree  # type: ignore
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override


class TorchRandomRLModule(TorchRLModule):
    """Generates uniform random actions for Box/Discrete spaces on torch tensors."""

    framework = "torch"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Dummy learnable tensor ensures torch.optim sees a non-empty parameter list
        # even though this module never runs gradient updates.
        self.register_parameter("_dummy_param", torch.nn.Parameter(torch.zeros(1)))

        if isinstance(self.action_space, gym.spaces.Box):
            low = torch.as_tensor(self.action_space.low, dtype=torch.float32)
            high = torch.as_tensor(self.action_space.high, dtype=torch.float32)
            self.register_buffer("_action_low", low, persistent=False)
            self.register_buffer("_action_high", high, persistent=False)
            self._action_type = "box"
        elif isinstance(self.action_space, gym.spaces.Discrete):
            self._action_type = "discrete"
            self._num_actions = torch.tensor(self.action_space.n, dtype=torch.long)
        else:
            raise ValueError(
                f"TorchRandomRLModule only supports Box or Discrete action spaces, "
                f"got: {self.action_space}"
            )

    def _batch_size_and_device(self, batch: Dict[str, Any]) -> Tuple[int, torch.device]:
        obs = batch.get(Columns.OBS, batch.get(SampleBatch.OBS))
        if obs is None:
            raise ValueError("TorchRandomRLModule requires observations to infer batch size.")
        first = tree.flatten(obs)[0]
        if not torch.is_tensor(first):
            first = torch.as_tensor(first)
        return first.shape[0], first.device

    def _sample_random_actions(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self._action_type == "box":
            low = self._action_low.to(device)
            high = self._action_high.to(device)
            rand = torch.rand((batch_size,) + low.shape, device=device)
            return low + (high - low) * rand
        actions = torch.randint(
            low=0,
            high=int(self._num_actions.item()),
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )
        return actions

    @override(TorchRLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        batch_size, device = self._batch_size_and_device(batch)
        actions = self._sample_random_actions(batch_size, device)
        return {Columns.ACTIONS: actions}

    @override(TorchRLModule)
    def _forward_train(self, *args, **kwargs):
        raise NotImplementedError("TorchRandomRLModule is inference-only.")
