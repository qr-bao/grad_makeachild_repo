"""Shared helpers to launch training runs for different algorithms."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num-iterations", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-envs-per-worker", type=int, default=2)
    parser.add_argument("--checkpoint-freq", type=int, default=20)
    parser.add_argument("--keep-all-checkpoints", action="store_true")
    parser.add_argument(
        "--env-config-file",
        type=Path,
        default=Path("predpreygrass_rllib_env129/config/config_env_base.json"),
    )
    parser.add_argument("--max-env-steps", type=int, default=2000)
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--rollout-fragment-length",
        type=str,
        default="auto",
        help="Value for train_simple.py --rollout-fragment-length",
    )
    parser.add_argument("--train-batch-size", type=int, default=396)
    parser.add_argument("--ppo-minibatch-size", type=int, default=256)
    parser.add_argument("--ppo-num-epochs", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Any extra args passed verbatim to train_simple.py",
    )
    return parser


def run_train_simple(args_list: list[str]) -> int:
    """Invoke train_simple.py with a list of arguments."""
    cmd = [sys.executable, "predpreygrass_rllib_env129/train_simple.py"] + args_list
    proc = subprocess.run(cmd)
    return proc.returncode


def str_path(path: Path) -> str:
    return str(path.expanduser().resolve())
