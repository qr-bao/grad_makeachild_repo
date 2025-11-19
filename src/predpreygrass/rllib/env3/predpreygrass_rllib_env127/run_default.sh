#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH=$SCRIPT_DIR/../../../../ python "$SCRIPT_DIR/train_simple.py" \
  --num-iterations 1000 \
  --rollout-fragment-length auto \
  --train-batch-size 396 \
  --eval-interval 10 \
  --eval-episodes 5 \
  --checkpoint-freq 50 \
  "$@"
