# prey_test_config.py - 添加软限制参数

prey_test_config = {
    # ============================================================
    # 🌍 世界基础设置
    # ============================================================
    'enable_continuous_space': True,
    'world_width': 800,
    'world_height': 800,
    'agent_radius': 8,
    # ============================================================
    # 🎁 Reward系统
    # ============================================================
    'reward_predator_catch_prey': 1.5,
    'reward_prey_eat_grass': 1.0,
    'reward_predator_step': -0.01,
    'reward_prey_step': -0.01,
    'penalty_prey_caught': -2.0,
    'reproduction_reward_predator': 1.0,
    'reproduction_reward_prey': 1.0,
    # 终局奖励：回合结束时仍存活的种群加分
    'survival_bonus_predator': 0.5,
    'survival_bonus_prey': 0.5,
    # 是否允许在测试模式下没有捕食者（训练阶段保持False）
    'allow_empty_predator_population': True,
    # ============================================================
    # 👥 种群初始设置
    # ============================================================
    'n_initial_active_predator': 0,
    'n_initial_active_prey': 10,
    'n_possible_predators': 20,
    'n_possible_prey': 20,

    'n_populations': 1,
    "population_display_info": {
        "predator_0": "Immortal",
        # "predator_1": "Immortal",
        # "predator_2": "Immortal",
        # "predator_3": "Immortal",
        # "predator_4": "Immortal",
        
        # "prey_0": "SmartAgent",
        # "prey_1": "Random",
        # "prey_2": "Random",
        # "prey_3": "Random",
        "prey_4": "Random",
    },

    # ============================================================
    # 🌱 草系统 - 软限制配置
    # ============================================================
    # --- 基础数量 ---
    'initial_num_grass': 50,  # 固定草数量

    # --- 能量系统 ---
    'initial_energy_grass': 10,  # 初始/重生能量
    'energy_gain_per_step_grass': 0.5,  # 每步增长速率
    'grass_energy_decay_constant': 10.0,  # 兼容占位

    # --- 繁殖基本条件 ---
    'enable_grass_reproduction': False,  # 固定草模式禁用繁殖
    'grass_reproduction_age': 80,
    'grass_reproduction_energy_threshold': 10.0,
    'grass_reproduction_cooldown': 60,
    #实用最大能量 ≈ decay_constant × 4 到 5

    # --- 局部密度调节（软限制核心）⭐ ---
    'grass_perception_radius': 400.0,  # 感知半径/px（↑对密度更敏感，↓饱和密度）
    'grass_reference_neighbors': 0.1,  # 参考邻居数（↓对密度更敏感，↓饱和密度）⭐最关键
    'grass_density_cache_interval': 10,  # 密度缓存更新间隔/步（性能优化）
    #N_max ≈ (world_width × world_height) / (π × perception_radius²) × (reference × 9)
    # --- 繁殖机制 ---
    'grass_reproduction_range': 400.0,  # 后代生成半径/px（↑分布更分散）
    'grass_reproduction_cost': 2.0,  # 繁殖能量消耗（↑单株繁殖频率降低）
    'grass_offspring_energy': 10.0,  # 新生草初始能量（↑二代繁殖更快）
    'grass_spawn_max_attempts': 5,  # 找位置最大尝试次数（防卡死）

    # --- 死亡与重生 ---
    'grass_respawn_delay': 100,  # 原地复生延迟
    'fixed_grass_mode': True,
    # ============================================================
    # 🔋 能量系统 - 软限制配置
    # ============================================================
    # === PREDATOR:无敌设置 ===
    'initial_energy_predator': 500.0,
    'energy_loss_per_step_predator': 0.0,
    'predator_creation_energy_threshold': 500.0,

    # === PREY:优化能量系统 ===
    'initial_energy_prey': 150.0,
    'energy_loss_per_step_prey': 0.01,
    'prey_creation_energy_threshold': 120.0,

    # 软限制开关（True=衰减函数，False=硬上限）
    'enable_soft_energy_limit': False,
    'energy_saturation_predator': 200.0,  # 饱和点
    'energy_saturation_prey': 180.0,
    
    # 硬上限（enable_soft_energy_limit=False时使用）
    'enable_max_energy': False,
    'max_energy_predator': 500.0,
    'max_energy_prey': 200.0,
    
    'energy_transfer_efficiency': 0.9,
    'metabolism_rate': 0.0,
    'base_metabolism': 0.05,
    'movement_cost_factor': 0.001,
    'thrust_cost_factor': 0.005,
    'turn_penalty_factor': 0.001,

    # ============================================================
    # 👶 繁殖系统
    # ============================================================
    'enable_paired_reproduction': False,
    'reproduction_mode': 'ratio',
    'reproduction_energy_ratio': 0.15,

    'min_reproduction_age': 50,
    'max_reproduction_age': 10000,
    'reproduction_cooldown': 300,
    'mating_distance': 80.0,
    'max_population_size': 150,

    # ============================================================
    # 🏃 物理系统
    # ============================================================
    'n_sensors': 30,
    'sensor_range': 600.0,
    'thrust_scale_predator': 2.0,
    'thrust_scale_prey': 1600.0,
    'soft_speed_limit_predator': 2.0,
    'soft_speed_limit_prey': 180.0,
    'drag_coefficient': 0.0008,
    
    # ============================================================
    # 🍽️ 饥饿系统
    # ============================================================
    'enable_hunger': False,
    'max_steps_without_food_predator': 999999,
    'max_steps_without_food_prey': 400,
    'hunger_damage': 0.1,

    # ============================================================
    # 💥 碰撞系统
    # ============================================================
    'enable_agent_collision': False,
    'collision_damage': 0.3,
    'wall_collision_damage': 0.5,

    # ============================================================
    # 🎮 仿真设置
    # ============================================================
    'max_steps': 500,
    'verbose_spawning': False,
    'verbose_engagement': False,
    'verbose_movement': False,
}
