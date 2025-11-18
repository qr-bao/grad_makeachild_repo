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
ENV_CONFIG_FILE="${LOG_DIR}/env_config_override.json"

# 将需要修改的环境参数写到 JSON，运行前可直接编辑下方值
cat <<'JSON' > "${ENV_CONFIG_FILE}"
{
  "enable_continuous_space": true,
  "world_width": 800,
  "world_height": 800,
  "agent_radius": 8,
  "reward_predator_catch_prey": 1.5,
  "reward_prey_eat_grass": 1.0,
  "reward_predator_step": -0.01,
  "reward_prey_step": -0.01,
  "penalty_prey_caught": -2.0,
  "reproduction_reward_predator": 1.0,
  "reproduction_reward_prey": 1.0,
  "survival_bonus_predator": 0.5,
  "survival_bonus_prey": 0.5,
  "allow_empty_predator_population": true,
  "n_initial_active_predator": 0,
  "n_initial_active_prey": 10,
  "n_possible_predators": 20,
  "n_possible_prey": 20,
  "n_populations": 1,
  "population_display_info": {
    "predator_0": "Immortal",
    "prey_4": "Random"
  },
  "initial_num_grass": 50,
  "initial_energy_grass": 10,
  "energy_gain_per_step_grass": 0.5,
  "grass_energy_decay_constant": 10.0,
  "enable_grass_reproduction": false,
  "grass_reproduction_age": 80,
  "grass_reproduction_energy_threshold": 10.0,
  "grass_reproduction_cooldown": 60,
  "grass_perception_radius": 400.0,
  "grass_reference_neighbors": 0.1,
  "grass_density_cache_interval": 10,
  "grass_reproduction_range": 400.0,
  "grass_reproduction_cost": 2.0,
  "grass_offspring_energy": 10.0,
  "grass_spawn_max_attempts": 5,
  "grass_respawn_delay": 100,
  "fixed_grass_mode": true,
  "initial_energy_predator": 500.0,
  "energy_loss_per_step_predator": 0.0,
  "predator_creation_energy_threshold": 500.0,
  "initial_energy_prey": 150.0,
  "energy_loss_per_step_prey": 0.01,
  "prey_creation_energy_threshold": 120.0,
  "enable_soft_energy_limit": false,
  "energy_saturation_predator": 200.0,
  "energy_saturation_prey": 180.0,
  "enable_max_energy": false,
  "max_energy_predator": 500.0,
  "max_energy_prey": 200.0,
  "energy_transfer_efficiency": 0.9,
  "metabolism_rate": 0.0,
  "base_metabolism": 0.05,
  "movement_cost_factor": 0.001,
  "thrust_cost_factor": 0.005,
  "turn_penalty_factor": 0.001,
  "enable_paired_reproduction": true,
  "reproduction_mode": "ratio",
  "reproduction_energy_ratio": 0.15,
  "min_reproduction_age": 50,
  "max_reproduction_age": 10000,
  "reproduction_cooldown": 300,
  "mating_distance": 80.0,
  "max_population_size": 150,
  "n_sensors": 20,
  "sensor_range": 200.0,
  "thrust_scale_predator": 2.0,
  "thrust_scale_prey": 1600.0,
  "soft_speed_limit_predator": null,
  "soft_speed_limit_prey": null,
  "drag_coefficient": 0.0,
  "use_direct_velocity_control": true,
  "direct_velocity_speed_predator": 200.0,
  "direct_velocity_speed_prey": 200.0,
  "direct_velocity_accel_scale_predator": 24.0,
  "direct_velocity_accel_scale_prey": 40.0,
  "enable_hunger": false,
  "max_steps_without_food_predator": 999999,
  "max_steps_without_food_prey": 400,
  "hunger_damage": 0.1,
  "enable_agent_collision": false,
  "collision_damage": 0.3,
  "wall_collision_damage": 0.5,
  "max_steps": 5000,
  "verbose_spawning": false,
  "verbose_engagement": false,
  "verbose_movement": false
}
JSON

source /mnt/hdd/miniconda3/etc/profile.d/conda.sh
conda activate predpreygrass

python "${PROJECT_ROOT}/src/predpreygrass/rllib/env3/predpreygrass_rllib_env127/train_simple.py" \
  --num-iterations "${NUM_ITERATIONS}" \
  --num-workers "${NUM_WORKERS}" \
  --num-envs-per-worker "${NUM_ENVS_PER_WORKER}" \
  --predator-survival-bonus "${PREDATOR_SURVIVAL_BONUS}" \
  --prey-survival-bonus "${PREY_SURVIVAL_BONUS}" \
  --log-dir "${LOG_DIR}" \
  --env-config-file "${ENV_CONFIG_FILE}"
