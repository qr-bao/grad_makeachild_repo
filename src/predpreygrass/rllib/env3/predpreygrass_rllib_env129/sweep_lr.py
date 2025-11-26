#!/usr/bin/env python3
"""Utility script to sweep different LR schedules by calling train_simple.py."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_SCHEDULES = [
    {"name": "const_3e-4", "schedule": [[0, 3e-4]]},
    {"name": "decay_200", "schedule": [[0, 3e-4], [200, 1.5e-4]]},
    {"name": "decay_500", "schedule": [[0, 3e-4], [500, 1e-4]]},
]


def build_cmd(args, schedule_spec):
    lr_schedule = json.dumps(schedule_spec["schedule"])
    cmd = [
        sys.executable,
        str(args.train_script),
        "--num-iterations",
        str(args.num_iterations),
        "--rollout-fragment-length",
        "auto",
        "--train-batch-size",
        str(args.train_batch_size),
        "--eval-interval",
        str(args.eval_interval),
        "--eval-episodes",
        str(args.eval_episodes),
        "--log-level",
        args.log_level,
        "--checkpoint-freq",
        str(args.checkpoint_freq),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--lr-schedule",
        lr_schedule,
    ]
    if args.env_config_file:
        cmd.extend(["--env-config-file", args.env_config_file])
    if args.log_dir:
        cmd.extend(["--log-dir", args.log_dir])
    if args.disable_env_debug:
        cmd.append("--disable-env-debug")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep different LR schedules via train_simple.py")
    parser.add_argument("--train-script", type=Path, default=Path("train_simple.py"))
    parser.add_argument("--env-config-file", type=str, default="env_config_enable_repro.json")
    parser.add_argument("--num-iterations", type=int, default=300)
    parser.add_argument("--train-batch-size", type=int, default=396)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--log-level", type=str, default="ERROR")
    parser.add_argument("--checkpoint-freq", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument(
        "--disable-env-debug", action="store_true", help="Pass --disable-env-debug to train_simple"
    )
    parser.add_argument(
        "--schedules-json",
        type=str,
        help="Optional JSON list [{\"name\":..,\"schedule\":[[0,lr],...]}, ...]. "
        "Defaults to three preset schedules.",
    )
    args = parser.parse_args()

    if args.schedules_json:
        schedules = json.loads(args.schedules_json)
    else:
        schedules = DEFAULT_SCHEDULES

    for spec in schedules:
        cmd = build_cmd(args, spec)
        print(f"\n=== Running schedule {spec['name']} ===")
        print(" ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[WARN] Schedule {spec['name']} exited with code {result.returncode}")


if __name__ == "__main__":
    main()
