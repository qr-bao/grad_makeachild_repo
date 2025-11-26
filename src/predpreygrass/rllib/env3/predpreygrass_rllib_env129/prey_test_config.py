"""Prey test config built on top of env129 base config."""

from predpreygrass.rllib.env3.predpreygrass_rllib_env129.config.config_env_base import (
    config_env_base,
)

# prey_test_config = {
#     **config_env_base,
#     "n_initial_active_prey": 10,
#     "population_display_info": {
#         "predator_0": "Immortal",
#         "prey_4": "Random",
#     },
#     "enable_grass_reproduction": False,
#     # ============================================================
#     # 👶 繁殖系统
#     # ============================================================
#     "enable_paired_reproduction": True,
#     "reproduction_mode": "ratio",
#     "reproduction_energy_ratio": 0.12,
#     "reproduction_energy_ratio_predator": 0.25,
#     "reproduction_energy_ratio_prey": 0.1,
#     "min_reproduction_age": 50,
#     "min_reproduction_age_predator": 80,
#     "min_reproduction_age_prey": 40,
#     "max_reproduction_age": 50000,
#     "reproduction_cooldown": 20,
#     "reproduction_cooldown_predator": 200,
#     "reproduction_cooldown_prey": 100,
#     "mating_distance": 80.0,
#     "max_population_size": 20000,
#     "max_population_size_predator": 20000,
#     "max_population_size_prey": 20000,
#     # ============================================================
#     # 🏃 物理系统
#     # ============================================================
#     "n_sensors": 20,
#     "sensor_range": 200.0,
#     "thrust_scale_predator": 1200.0,
#     "thrust_scale_prey": 1600.0,
#     "soft_speed_limit_predator": 420.0,
#     "soft_speed_limit_prey": None,
#     "drag_coefficient": 0.0,
#     # 直接速度控制：让连续动作立即映射为速度向量
#     "use_direct_velocity_control": True,
#     "direct_velocity_speed_predator": 100.0,
#     "direct_velocity_speed_prey": 220.0,
#     # 每步速度增量：0.5 × scale ≈ px/frame
#     "direct_velocity_accel_scale_predator": 60.0,
#     "direct_velocity_accel_scale_prey": 40.0,
#     # ============================================================
#     # 🍽️ 饥饿系统
#     # ============================================================
#     "enable_hunger": False,
#     "max_steps_without_food_predator": 999999,
#     "max_steps_without_food_prey": 400,
#     "hunger_damage": 0.1,
#     # ============================================================
#     # 💥 碰撞系统
#     # ============================================================
#     "enable_agent_collision": False,
#     "collision_damage": 0.3,
#     "wall_collision_damage": 0.5,
#     # ============================================================
#     # 🎮 仿真设置
#     # ============================================================
#     "max_steps": 2000,
#     "verbose_spawning": False,
#     "verbose_engagement": False,
#     "verbose_movement": False,
# }
prey_test_config = config_env_base
