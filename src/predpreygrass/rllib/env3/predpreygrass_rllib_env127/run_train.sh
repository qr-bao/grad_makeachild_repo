#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------
# 可修改参数
# -------------------------------------------------------
NUM_ITERATIONS=10000
NUM_WORKERS=4
NUM_ENVS_PER_WORKER=3
PREDATOR_SURVIVAL_BONUS=0.5
PREY_SURVIVAL_BONUS=0.3
# -------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
LOG_BASE="${PROJECT_ROOT}/src/predpreygrass/rllib/env3/predpreygrass_rllib_env127/logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_BASE}/run_${RUN_ID}"
mkdir -p "${LOG_DIR}"
ENV_CONFIG_FILE="${LOG_DIR}/env_config.json"

# 将当前 prey_test_config 导出为本次运行的环境配置快照
cd "${PROJECT_ROOT}"
python - <<PY
import json
from pathlib import Path
from predpreygrass.rllib.env3.predpreygrass_rllib_env127.prey_test_config import prey_test_config
env_path = Path("${ENV_CONFIG_FILE}")
env_path.write_text(json.dumps(prey_test_config, indent=2), encoding="utf-8")
PY

source /mnt/hdd/miniconda3/etc/profile.d/conda.sh
conda activate predpreygrass

cd "${PROJECT_ROOT}"
python "${PROJECT_ROOT}/src/predpreygrass/rllib/env3/predpreygrass_rllib_env127/train_simple.py" \
  --num-iterations "${NUM_ITERATIONS}" \
  --num-workers "${NUM_WORKERS}" \
  --num-envs-per-worker "${NUM_ENVS_PER_WORKER}" \
  --predator-survival-bonus "${PREDATOR_SURVIVAL_BONUS}" \
  --prey-survival-bonus "${PREY_SURVIVAL_BONUS}" \
  --log-dir "${LOG_DIR}" \
  --env-config-file "${ENV_CONFIG_FILE}"
