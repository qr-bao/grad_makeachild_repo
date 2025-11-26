"""Lightweight RLlib callback that tracks episode returns/lengths."""

from __future__ import annotations

import time
from collections import defaultdict

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger


class EpisodeReturn(DefaultCallbacks):
    """Aggregate rewards per episode and log summary statistics."""

    def __init__(self):
        super().__init__()
        self.overall_sum_of_rewards = 0.0
        self.num_episodes = 0
        self._pending_episode_metrics = []
        self.start_time = time.time()
        self.last_iteration_time = self.start_time
        self.episode_lengths = {}

    def on_episode_step(self, *, episode, **kwargs):  # noqa: D401
        eid = episode.id_
        self.episode_lengths[eid] = self.episode_lengths.get(eid, 0) + 1

    def on_episode_end(self, *, episode, metrics_logger: MetricsLogger, **kwargs):
        self.num_episodes += 1
        episode_return = episode.get_return()
        episode_id = episode.id_
        episode_length = self.episode_lengths.pop(episode_id, 0)
        self.overall_sum_of_rewards += episode_return

        group_rewards = defaultdict(list)
        predator_total = 0.0
        prey_total = 0.0
        for agent_id, rewards in episode.get_rewards().items():
            total = sum(rewards)
            if "predator" in agent_id:
                predator_total += total
            elif "prey" in agent_id:
                prey_total += total
            for group in [
                "type_1_predator",
                "type_2_predator",
                "type_1_prey",
                "type_2_prey",
            ]:
                if group in agent_id:
                    group_rewards[group].append(total)
                    break

        print(
            f"Episode {self.num_episodes}: Length={episode_length} | Return={episode_return:.2f} | "
            f"Pred={predator_total:.2f} | Prey={prey_total:.2f}"
        )
        for group, totals in group_rewards.items():
            print(f"  - {group}: {sum(totals):.2f}")

        metrics_logger.log_value("episode_length", episode_length, reduce="mean")

    def on_train_result(self, *, result, **kwargs):  # noqa: D401
        now = time.time()
        total_elapsed = now - self.start_time
        iter_num = result.get("training_iteration", 1)
        iter_time = now - self.last_iteration_time
        self.last_iteration_time = now

        result["timing/iter_minutes"] = iter_time / 60.0
        result["timing/avg_minutes_per_iter"] = total_elapsed / 60.0 / iter_num
        result["timing/total_hours_elapsed"] = total_elapsed / 3600.0
