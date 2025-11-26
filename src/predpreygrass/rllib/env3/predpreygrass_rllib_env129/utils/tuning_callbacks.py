"""Shared Tune/RLlib callbacks for hyper-parameter experiments."""

from __future__ import annotations

import csv
import os

from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.stopper import Stopper


class PredatorScore(DefaultCallbacks):
    """Expose predator rewards via ``score_pred`` and log 100+ milestones."""

    def on_train_result(self, *, algorithm, result, **kwargs):  # noqa: D401
        agent_metrics = result.get("env_runners", {}).get("agent_episode_returns_mean", {})
        pred = float(
            agent_metrics.get("predator_policy", agent_metrics.get("predator_0", 0.0))
        )
        score_pred = pred / 100.0
        result["score_pred"] = score_pred
        algorithm._last_score_pred = score_pred
        algorithm._last_iter = int(result.get("training_iteration", -1))
        cfg_from_result = result.get("config", {}) or {}
        algorithm._last_lr = float(cfg_from_result.get("lr", algorithm.config.get("lr")))
        algorithm._last_num_epochs = int(cfg_from_result.get("num_epochs", algorithm.config.get("num_epochs")))
        if score_pred >= 1.0 and not getattr(algorithm, "_saved_pred100_once", False):
            trial_dir = getattr(algorithm, "local_path", None) or getattr(algorithm, "logdir", None)
            if not trial_dir:
                return
            experiment_dir = os.path.dirname(trial_dir)
            csv_path = os.path.join(experiment_dir, "predator_100_hits.csv")
            row = {
                "trial_name": os.path.basename(trial_dir),
                "iteration": algorithm._last_iter,
                "score_pred": score_pred,
                "lr": algorithm._last_lr,
                "num_epochs": algorithm._last_num_epochs,
            }
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["trial_name", "iteration", "score_pred", "lr", "num_epochs"],
                )
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            setattr(algorithm, "_saved_pred100_once", True)


class DropStopper(Stopper):
    """Stop a trial if score_pred drops sharply over a window."""

    def __init__(self, window: int = 5, drop_threshold: float = 0.2, name: str = "drop"):
        self.window = int(window)
        self.drop_threshold = float(drop_threshold)
        self.history = {}
        self.name = name

    def __call__(self, trial_id: str, result: dict) -> bool:
        score = result.get("score_pred")
        if score is None:
            return False
        hist = self.history.setdefault(trial_id, [])
        hist.append(float(score))
        if len(hist) > self.window:
            hist.pop(0)
        if len(hist) == self.window and (hist[-1] < hist[0] - self.drop_threshold):
            return True
        return False

    def stop_all(self) -> bool:  # noqa: D401
        return False


class ReasonedStopper(Stopper):
    """Wrap multiple stoppers and record the stop reason."""

    def __init__(self, *stoppers: Stopper):
        self.stoppers = list(stoppers)
        for stopper in self.stoppers:
            if not hasattr(stopper, "name"):
                stopper.name = stopper.__class__.__name__.lower()
        self._reasons = {}

    def __call__(self, trial_id: str, result: dict) -> bool:
        for stopper in self.stoppers:
            if stopper(trial_id, result):
                reason = getattr(stopper, "name", stopper.__class__.__name__)
                self._reasons[trial_id] = reason
                result["__stop_reason"] = reason
                return True
        return False

    def stop_all(self) -> bool:  # noqa: D401
        return any(getattr(s, "stop_all", lambda: False)() for s in self.stoppers)

    def reason_for(self, trial_id: str) -> str | None:
        return self._reasons.get(trial_id)


class FinalMetricsLogger(tune.Callback):
    """Log final metrics for each trial into ``predator_final.csv``."""

    def __init__(self, reasoned_stop: ReasonedStopper, max_iters: int, grace_period: int, reduction_factor: int):
        super().__init__()
        self._reasoned_stop = reasoned_stop
        self._max_iters = int(max_iters)
        self._grace = int(grace_period)
        self._rf = int(reduction_factor)
        self._rungs = self._build_rungs()

    def _build_rungs(self):
        rungs = []
        cur = self._grace
        while cur < self._max_iters and cur > 0:
            rungs.append(cur)
            next_cur = int(cur * self._rf)
            if next_cur <= cur:
                break
            cur = next_cur
        return rungs

    def _annotate_rung_info(self, final_iter: int, stop_reason: str):
        asha_pruned = int(stop_reason == "asha_early_stop")
        idx = -1
        for i, boundary in enumerate(self._rungs):
            if final_iter >= boundary:
                idx = i
            else:
                break
        rung_pruned_at = ""
        if asha_pruned:
            for boundary in self._rungs:
                if final_iter < boundary:
                    rung_pruned_at = str(boundary)
                    break
        return idx, rung_pruned_at, asha_pruned

    def _write_csv_row(self, trial, result, stop_reason):
        experiment_dir = os.path.dirname(trial.local_path)
        csv_path = os.path.join(experiment_dir, "predator_final.csv")
        score_pred = float(result.get("score_pred", float("nan")))
        final_iter = int(result.get("training_iteration", -1))
        cfg = result.get("config", {}) or {}
        lr = float(cfg.get("lr", float("nan")))
        num_epochs = int(cfg.get("num_epochs", -1))
        progress_ratio = (final_iter / self._max_iters) if self._max_iters > 0 and final_iter >= 0 else float("nan")
        rung_index, rung_pruned_at, asha_pruned = self._annotate_rung_info(final_iter, stop_reason)
        row = {
            "trial_name": os.path.basename(trial.local_path),
            "iteration": final_iter,
            "progress_ratio": f"{progress_ratio:.6f}" if progress_ratio == progress_ratio else "nan",
            "score_pred": score_pred,
            "lr": lr,
            "num_epochs": num_epochs,
            "stop_reason": stop_reason,
            "rung_index": rung_index,
            "rung_pruned_at": rung_pruned_at,
            "asha_pruned": asha_pruned,
        }
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "trial_name",
                    "iteration",
                    "progress_ratio",
                    "score_pred",
                    "lr",
                    "num_epochs",
                    "stop_reason",
                    "rung_index",
                    "rung_pruned_at",
                    "asha_pruned",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def on_trial_complete(self, iteration, trials, trial, result=None, **info):  # noqa: D401
        if result is None:
            result = info.get("result") or getattr(trial, "last_result", {}) or {}
        reason = self._reasoned_stop.reason_for(trial.trial_id) or result.get("__stop_reason")
        final_iter = int(result.get("training_iteration", -1))
        if not reason and 0 <= final_iter < self._max_iters:
            reason = "asha_early_stop"
        if not reason:
            reason = "completed"
        self._write_csv_row(trial, result, reason)

    def on_trial_error(self, iteration, trials, trial, **info):  # noqa: D401
        fake_result = {"score_pred": float("nan"), "training_iteration": -1, "config": {}}
        self._write_csv_row(trial, fake_result, "error")


__all__ = [
    "PredatorScore",
    "DropStopper",
    "ReasonedStopper",
    "FinalMetricsLogger",
]
