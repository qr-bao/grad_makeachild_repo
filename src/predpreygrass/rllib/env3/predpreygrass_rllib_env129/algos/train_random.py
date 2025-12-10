#!/usr/bin/env python3
"""Wrapper to launch pure random baseline (predator=random, prey=random)."""

from __future__ import annotations

from pathlib import Path

from predpreygrass.rllib.env3.predpreygrass_rllib_env129.algos.common import (
    base_parser,
    run_train_simple,
    str_path,
)


def main() -> int:
    parser = base_parser()
    args = parser.parse_args()

    cmd_args = [
        "--num-iterations",
        str(args.num_iterations),
        "--num-workers",
        str(args.num_workers),
        "--num-envs-per-worker",
        str(args.num_envs_per_worker),
        "--checkpoint-freq",
        str(args.checkpoint_freq),
        "--env-config-file",
        str_path(args.env_config_file),
        "--max-env-steps",
        str(args.max_env_steps),
        "--log-level",
        args.log_level,
        "--rollout-fragment-length",
        str(args.rollout_fragment_length),
        "--train-batch-size",
        str(args.train_batch_size),
        "--ppo-minibatch-size",
        str(args.ppo_minibatch_size),
        "--ppo-num-epochs",
        str(args.ppo_num_epochs),
        "--grad-clip",
        str(args.grad_clip),
        "--predator-strategy",
        "random",
        "--prey-strategy",
        "random",
    ]

    if args.keep_all_checkpoints:
        cmd_args.append("--keep-all-checkpoints")
    if args.extra_args:
        cmd_args += args.extra_args

    return run_train_simple(cmd_args)


if __name__ == "__main__":
    raise SystemExit(main())
