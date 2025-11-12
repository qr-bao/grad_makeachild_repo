"""
The base environment for the PredPreyGrass simulation.
Two types of agents: predators and prey. Independently learning policies for each type.
"""
from predpreygrass.rllib.env3.predpreygrass_rllib_env124 import config_env

# external libraries
import numpy as np

import logging
import math
import os
import uuid
from pathlib import Path
from typing import Any, Set

from numpy.typing import NDArray
import gymnasium
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, Dict, List, Tuple
import pymunk
import pymunk.pygame_util  # 用于调试渲染

class PredPreyGrass(MultiAgentEnv):
    def _setup_debug_logger(self, config: Dict[str, Any]) -> None:
        """Configure a per-environment logger for structured debugging."""
        self.env_instance_id = config.get("env_instance_id", f"env-{uuid.uuid4().hex[:8]}")
        self.debug_logging = bool(config.get("debug_logging", False))
        self.debug_log_level = str(config.get("debug_log_level", "INFO")).upper()
        self.debug_log_dir = Path(config.get("debug_log_dir", "logs"))
        self.debug_log_file = config.get("debug_log_file")

        self.debug_log_dir.mkdir(parents=True, exist_ok=True)
        if self.debug_log_file:
            log_path = Path(self.debug_log_file)
        else:
            log_path = self.debug_log_dir / f"predpreygrass_env_{os.getpid()}_{self.env_instance_id}.log"

        self.debug_log_path = log_path
        logger_name = f"PredPreyGrassEnv.{self.env_instance_id}"
        self._logger = logging.getLogger(logger_name)
        self._logger.propagate = False
        self._logger.handlers.clear()

        if self.debug_logging:
            handler = logging.FileHandler(log_path)
            level = getattr(logging, self.debug_log_level, logging.INFO)
            handler.setLevel(level)
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
            self._logger.addHandler(handler)
            self._logger.setLevel(level)
            self._logger.info(
                "event=logger_initialised | env_id=%s | log_path=%s | level=%s",
                self.env_instance_id,
                log_path,
                self.debug_log_level,
            )
        else:
            self._logger.addHandler(logging.NullHandler())
            self._logger.setLevel(logging.WARNING)

    def _short_repr(self, value: Any, max_items: int = 5) -> str:
        try:
            if isinstance(value, dict):
                preview = list(value.items())[:max_items]
                formatted = ", ".join(f"{k}:{self._short_repr(v, max_items)}" for k, v in preview)
                if len(value) > max_items:
                    formatted += ", ..."
                return f"{{{formatted}}}"
            if isinstance(value, (list, tuple, set)):
                seq = list(value)
                preview = seq[:max_items]
                suffix = ", ..." if len(seq) > max_items else ""
                return f"[{', '.join(self._short_repr(v, max_items) for v in preview)}{suffix}]"
            if isinstance(value, np.ndarray):
                preview = value.flatten()[:max_items].tolist()
                suffix = "..." if value.size > max_items else ""
                return (
                    f"ndarray(shape={value.shape}, dtype={value.dtype}, preview={preview}{suffix})"
                )
            return repr(value)
        except Exception as exc:  # pragma: no cover - best effort logging helper
            return f"<unreprable {type(value).__name__}: {exc}>"

    def _format_log_message(self, event: str, fields: Dict[str, Any]) -> str:
        parts = [
            f"event={event}",
            f"env_id={self.env_instance_id}",
            f"step={getattr(self, 'current_step', 'NA')}",
        ]
        for key, value in fields.items():
            parts.append(f"{key}={self._short_repr(value)}")
        return " | ".join(parts)

    def _log_debug(self, event: str, **fields: Any) -> None:
        if self.debug_logging and self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(self._format_log_message(event, fields))

    def _log_info(self, event: str, **fields: Any) -> None:
        if self.debug_logging and self._logger.isEnabledFor(logging.INFO):
            self._logger.info(self._format_log_message(event, fields))

    def _log_warning(self, event: str, **fields: Any) -> None:
        if self._logger.isEnabledFor(logging.WARNING):
            self._logger.warning(self._format_log_message(event, fields))

    def _apply_survival_bonus(self, agents: List[AgentID], bonus_value: float, rewards: Dict[AgentID, float]) -> None:
        """Add survival bonus to the provided agents."""
        if bonus_value == 0.0 or not agents:
            return
        for agent in agents:
            rewards[agent] = rewards.get(agent, 0.0) + bonus_value
            self.cumulative_rewards[agent] = self.cumulative_rewards.get(agent, 0.0) + bonus_value

    def __init__(self, config=None):
        super().__init__()
        config = config or config_env  # Use provided config or default config_env
        self._setup_debug_logger(config)
        
        # === 1. 基础配置（最先读取） ===
        self.verbose_engagement = config.get("verbose_engagement", False)
        self.verbose_movement = config.get("verbose_movement", False)
        self.verbose_spawning = config.get("verbose_spawning", False)
        self.max_steps = config.get("max_steps", 10000)
        
        # === 2. 空间相关配置（必须在观察空间定义之前） ===
        self.enable_continuous_space = config.get("enable_continuous_space", True)
        self.world_width = config.get("world_width", 750)
        self.world_height = config.get("world_height", 750)
        self.agent_radius = config.get("agent_radius", 15)
        
        # === 3. 网格和观察设置 ===
        self.grid_size = config.get("grid_size", 10)
        self.num_obs_channels = config.get("num_obs_channels", 4)
        self.predator_obs_range = config.get("predator_obs_range", 7)
        self.prey_obs_range = config.get("prey_obs_range", 5)
        
        # === 4. 传感器配置（连续空间） ===
        if self.enable_continuous_space:
            self.n_sensors = config.get("n_sensors", 30)
            self.sensor_range = config.get("sensor_range", 150.0)
        
        # === 5. Rewards ===
        self.reward_predator_catch_prey = config.get("reward_predator_catch_prey", 0.0)
        self.reward_prey_eat_grass = config.get("reward_prey_eat_grass", 0.0)
        self.reward_predator_step = config.get("reward_predator_step", 0.0)
        self.reward_prey_step = config.get("reward_prey_step", 0.0)
        self.penalty_prey_caught = config.get("penalty_prey_caught", 0.0)
        self.survival_bonus_predator = config.get("survival_bonus_predator", 0.0)
        self.survival_bonus_prey = config.get("survival_bonus_prey", 0.0)
        self.reproduction_reward_predator = config.get("reproduction_reward_predator", 10.0)
        self.reproduction_reward_prey = config.get("reproduction_reward_prey", 10.0)
        
        # === 6. Energy settings ===
        self.energy_loss_per_step_predator = config.get("energy_loss_per_step_predator", 0.15)
        self.energy_loss_per_step_prey = config.get("energy_loss_per_step_prey", 0.05)
        self.predator_creation_energy_threshold = config.get("predator_creation_energy_threshold", 12.0)
        self.prey_creation_energy_threshold = config.get("prey_creation_energy_threshold", 8.0)

        # === 新增：完整能量系统配置 ===
        # 能量上限
        self.enable_max_energy = config.get("enable_max_energy", False)  # ← 默认True（有上限）
        self.max_energy_predator = config.get("max_energy_predator", 100.0)
        self.max_energy_prey = config.get("max_energy_prey", 100.0)

        # 初始能量（作为最大能量的百分比）
        self.initial_energy_predator = config.get("initial_energy_predator", 50.0)
        self.initial_energy_prey = config.get("initial_energy_prey", 50.0)
        # # 健康系统
        # self.max_health = config.get("max_health", 100.0)
        # self.initial_health = config.get("initial_health", 100.0)
        # self.health_regen_rate = config.get("health_regen_rate", 0.1)  # 每步恢复的健康
        # self.low_energy_health_loss = config.get("low_energy_health_loss", 0.5)  # 能量不足时健康损失
        # self.low_energy_threshold = config.get("low_energy_threshold", 20.0)  # 低能量阈值

        # 饥饿系统
        self.enable_hunger = config.get("enable_hunger", True)
        self.hunger_damage = config.get("hunger_damage", 0.2)  # 饥饿伤害
        self.max_steps_without_food_predator = config.get("max_steps_without_food_predator", 200)
        self.max_steps_without_food_prey = config.get("max_steps_without_food_prey", 150)

        # 能量转换效率
        self.energy_transfer_efficiency = config.get("energy_transfer_efficiency", 0.8)  # 捕食/进食时能量转换效率
        self.energy_transfer_efficiency_predator = config.get(
            "energy_transfer_efficiency_predator", self.energy_transfer_efficiency
        )
        self.energy_transfer_efficiency_prey = config.get(
            "energy_transfer_efficiency_prey", self.energy_transfer_efficiency
        )
        
        # === 7. Learning agents ===
        self.use_monotonic_offspring_ids = config.get("use_monotonic_offspring_ids", True)
        self.n_possible_predators = config.get("n_possible_predators", 50)
        self.n_possible_prey = config.get("n_possible_prey", 50)
        self.allow_empty_predator_population = bool(config.get("allow_empty_predator_population", False))
        self.n_initial_active_predator = config.get("n_initial_active_predator", 6)
        self.n_initial_active_prey = config.get("n_initial_active_prey", 8)
        self.initial_energy_predator = config.get("initial_energy_predator", 5.0)
        self.initial_energy_prey = config.get("initial_energy_prey", 3.0)
        if self.n_initial_active_predator < 0:
            raise ValueError("n_initial_active_predator cannot be negative.")
        if self.n_initial_active_predator == 0 and not self.allow_empty_predator_population:
            raise ValueError(
                "Predator count is zero but allow_empty_predator_population=False. "
                "Enable the flag to run predator-free tests."
            )
        if self.n_initial_active_predator > self.n_possible_predators:
            raise ValueError(
                "n_initial_active_predator cannot exceed n_possible_predators."
            )
        self.max_indices = {
            "predator": self.n_possible_predators,
            "prey": self.n_possible_prey,
        }
        # Start monotonic IDs right after the initially active agents so they never reuse IDs mid-episode.
        self.next_free_idx = {
            "predator": self.n_initial_active_predator,
            "prey": self.n_initial_active_prey,
        }
        self.retired_agents: set[str] = set()
        # === 新增：种群配置 ===
        self.n_populations = config.get("n_populations", 2)        
        # 接收外部传入的算法显示信息（仅用于GUI显示）
        self.population_display_info = config.get("population_display_info", {})
        # 格式示例: {"predator_0": "PPO", "predator_1": "Random", "prey_0": "DQN"}
        
        # 如果没有提供，生成默认显示信息
        if not self.population_display_info:
            for agent_type in ["predator", "prey"]:
                for pop_id in range(self.n_populations):
                    pop_key = f"{agent_type}_{pop_id}"
                    self.population_display_info[pop_key] = "Random"
        # === 新增：草的局部密度系统 ===
        self.grass_perception_radius = config.get("grass_perception_radius", 100.0)
        self.grass_density_reference = config.get("grass_density_reference", 8.0)
        self.grass_spawn_max_attempts = config.get("grass_spawn_max_attempts", 20)
        
        # === 8. Grass settings ===
        self.initial_num_grass = config.get("initial_num_grass", 25)
        # 初始草的数量。决定环境开始时草的密度（默认 25 株）。
        self.grass_min_energy = config.get("grass_min_energy", 0.1)
        # 草的最低能量阈值。当能量低于该值时，草会被视为死亡并被移除（默认 0.1）。
        self.grass_max_energy = config.get("grass_max_energy", 10.0)
        # 草的最大能量上限。生长过程不会让能量超过这个值（默认 10.0）。
        self.grass_base_growth_rate = config.get("grass_base_growth_rate", 1.0)
         # 草的基础生长速率。每步增加的能量以此为基准（会受局部密度抑制）（默认 1.0）。

        self.grass_decay_rate = config.get("grass_decay_rate", 0.05)
         # 草能量接近上限时的衰减速率，用于防止过度积累能量（默认 0.05）。

        self.enable_grass_reproduction = config.get("enable_grass_reproduction", True)
         # 是否启用草的繁殖机制（True = 可以繁殖；False = 不会繁殖）（默认启用）。
        self.grass_base_reproduce_prob = config.get("grass_base_reproduce_prob", 0.1)
        # 草的基础繁殖概率，在能量满足条件时，以此为基准乘以 (1 - 局部密度) 计算实际概率（默认 0.1）。
        self.grass_reproduce_threshold = config.get("grass_reproduce_threshold", 8.0)
        # 草的繁殖能量阈值。只有能量超过此值的草才可能繁殖（默认 8.0）。
        self.grass_reproduce_cost = config.get("grass_reproduce_cost", 2.0)
           # 草在成功繁殖后会消耗的能量（默认 2.0）。
        self.grass_offspring_energy = config.get("grass_offspring_energy", 5.0)
        # 新生成草的初始能量。影响幼草的生长起点与后续生长速度（默认 5.0）。

        self.fixed_grass_mode = bool(config.get("fixed_grass_mode", False))
        self.grass_fixed_growth_rate = float(config.get("energy_gain_per_step_grass", 0.5))
        self.grass_fixed_respawn_delay = int(config.get("grass_respawn_delay", 200))
        self.grass_fixed_initial_energy = float(config.get("initial_energy_grass", 0.5))
        self.grass_home_positions: Dict[str, Tuple[float, float]] = {}
        self.inactive_grass: Set[str] = set()
        self.grass_respawn_timers: Dict[str, int] = {}

        # === 9. 繁殖相关配置 ===
        self.enable_paired_reproduction = config.get("enable_paired_reproduction", False)
        self.mating_distance = config.get("mating_distance", 100.0)
        self.min_reproduction_health = config.get("min_reproduction_health", 70.0)
        self.n_populations = config.get("n_populations", 2)

        # === 繁殖参数（按物种区分） ===
        default_repro_config = {
            "mode": config.get("reproduction_mode", "fixed_ratio"),
            "energy_ratio": config.get("reproduction_energy_ratio", 1 / 3),
            "fixed_cost": config.get("reproduction_fixed_cost", 10.0),
            "transfer_ratio": config.get("reproduction_transfer_ratio", 0.3),
            "offspring_min_energy": config.get("offspring_min_energy", 60.0),
            "min_age": config.get("min_reproduction_age", 50),
            "max_age": config.get("max_reproduction_age", 800),
            "cooldown": config.get("reproduction_cooldown", 100),
            "max_population_size": config.get("max_population_size", 999999),
        }
        self.reproduction_settings = {}
        for agent_type in ("predator", "prey"):
            suffix = f"_{agent_type}"
            self.reproduction_settings[agent_type] = {
                "mode": config.get(f"reproduction_mode{suffix}", default_repro_config["mode"]),
                "energy_ratio": config.get(
                    f"reproduction_energy_ratio{suffix}", default_repro_config["energy_ratio"]
                ),
                "fixed_cost": config.get(
                    f"reproduction_fixed_cost{suffix}", default_repro_config["fixed_cost"]
                ),
                "transfer_ratio": config.get(
                    f"reproduction_transfer_ratio{suffix}", default_repro_config["transfer_ratio"]
                ),
                "offspring_min_energy": config.get(
                    f"offspring_min_energy{suffix}", default_repro_config["offspring_min_energy"]
                ),
                "min_age": config.get(
                    f"min_reproduction_age{suffix}", default_repro_config["min_age"]
                ),
                "max_age": config.get(
                    f"max_reproduction_age{suffix}", default_repro_config["max_age"]
                ),
                "cooldown": config.get(
                    f"reproduction_cooldown{suffix}", default_repro_config["cooldown"]
                ),
                "max_population_size": config.get(
                    f"max_population_size{suffix}", default_repro_config["max_population_size"]
                ),
            }

        # 兼容旧属性（外部可能仍在读取）
        self.reproduction_mode = default_repro_config["mode"]
        self.reproduction_energy_ratio = default_repro_config["energy_ratio"]
        self.reproduction_fixed_cost = default_repro_config["fixed_cost"]
        self.reproduction_transfer_ratio = default_repro_config["transfer_ratio"]
        self.offspring_min_energy = default_repro_config["offspring_min_energy"]
        self.min_reproduction_age = default_repro_config["min_age"]
        self.max_reproduction_age = default_repro_config["max_age"]
        self.reproduction_cooldown = default_repro_config["cooldown"]
        self.max_population_size = default_repro_config["max_population_size"]
        # # === 10. 食物系统配置 ===
        # self.n_food = config.get("n_food", 10)
        # self.n_poison = config.get("n_poison", 20)
        # self.food_reward = config.get("food_reward", 3.0)
        # self.poison_penalty = config.get("poison_penalty", -3.0)
        # self.food_energy = config.get("food_energy", 10.0)
        # self.poison_damage = config.get("poison_damage", 10.0)
        
        # === 11. 能量和奖励配置 ===
        self.metabolism_rate = config.get("metabolism_rate", 0.02)
        self.metabolism_rate_predator = config.get("metabolism_rate_predator", self.metabolism_rate)
        self.metabolism_rate_prey = config.get("metabolism_rate_prey", self.metabolism_rate)

        self.base_metabolism = config.get("base_metabolism", self.metabolism_rate)
        self.base_metabolism_predator = config.get("base_metabolism_predator", self.base_metabolism)
        self.base_metabolism_prey = config.get("base_metabolism_prey", self.base_metabolism)

        self.movement_cost_factor = config.get("movement_cost_factor", 0.01)
        self.movement_cost_factor_predator = config.get(
            "movement_cost_factor_predator", self.movement_cost_factor
        )
        self.movement_cost_factor_prey = config.get(
            "movement_cost_factor_prey", self.movement_cost_factor
        )

        self.thrust_cost_factor = config.get("thrust_cost_factor", 0.5)
        self.thrust_cost_factor_predator = config.get(
            "thrust_cost_factor_predator", self.thrust_cost_factor
        )
        self.thrust_cost_factor_prey = config.get(
            "thrust_cost_factor_prey", self.thrust_cost_factor
        )

        self.turn_penalty_factor = config.get("turn_penalty_factor", 0.0)
        self.turn_penalty_factor_predator = config.get(
            "turn_penalty_factor_predator", self.turn_penalty_factor
        )
        self.turn_penalty_factor_prey = config.get(
            "turn_penalty_factor_prey", self.turn_penalty_factor
        )
        # === 新增：碰撞配置 ===
        self.collision_damage = config.get("collision_damage", 0.5)  # 碰撞能量损失
        self.wall_collision_damage = config.get("wall_collision_damage", 1.0)  # 撞墙能量损失
        self.enable_agent_collision = config.get("enable_agent_collision", False)  # 是否启用智能体碰撞       
        # === 12. Track cumulative rewards ===
        self.cumulative_rewards = {}
        # === 13. Agent lists ===
        self.possible_agents: List[AgentID] = [
            f"predator_{i}" for i in range(self.n_possible_predators)
        ] + [f"prey_{j}" for j in range(self.n_possible_prey)]
        
        self.agents: List[AgentID] = [
            f"predator_{i}" for i in range(self.n_initial_active_predator)
        ] + [f"prey_{j}" for j in range(self.n_initial_active_prey)]
        
        self.grass_agents: List[AgentID] = [f"grass_{k}" for k in range(self.initial_num_grass)]

        
        # === 14. Gymnasium Spaces（现在可以安全使用 enable_continuous_space） ===
        self.self_state_dim = 12
        if self.enable_continuous_space:
            # === 连续空间：传感器观察（n_sensors*12 + self_state_dim）===
            sensor_obs_dim = self.n_sensors * 12 + self.self_state_dim
            self.sensor_obs_dim = sensor_obs_dim
            
            sensor_obs_space = gymnasium.spaces.Box(
                low=-1.0, 
                high=1.0, 
                shape=(sensor_obs_dim,), 
                dtype=np.float32
            )
            
            self.observation_spaces = {
                agent: sensor_obs_space for agent in self.possible_agents
            }
            
            if self.verbose_movement:
                print(f"[INIT] Enhanced observation space: {sensor_obs_dim} dimensions")
                print(f"  - Environment layer: {self.n_sensors * 3} (obstacle/grass_dist/grass_energy)")
                print(f"  - Predator layer: {self.n_sensors * 3} (dist/velocity/energy)")
                print(f"  - Prey layer: {self.n_sensors * 3} (dist/velocity/energy)")
                print(f"  - Mate layer: {self.n_sensors * 3} (dist/velocity/fertility)")
                print(f"  - Self state: {self.self_state_dim}")
        else:
            # === 离散空间：网格观察 ===
            predator_obs_shape = (self.num_obs_channels, self.predator_obs_range, self.predator_obs_range)
            prey_obs_shape = (self.num_obs_channels, self.prey_obs_range, self.prey_obs_range)
            
            predator_obs_space = gymnasium.spaces.Box(low=0.0, high=100.0, shape=predator_obs_shape, dtype=np.float64)
            prey_obs_space = gymnasium.spaces.Box(low=0.0, high=100.0, shape=prey_obs_shape, dtype=np.float64)
            
            self.observation_spaces = {
                agent: predator_obs_space if "predator" in agent else prey_obs_space for agent in self.possible_agents
            }
        
        # === 15. Action spaces ===
        # === 15. Action spaces ===
        # === 15. Action spaces ===
        # === 15. Action spaces ===
        if self.enable_continuous_space:
            # === 连续空间：2D 连续推力 ===
            action_space = gymnasium.spaces.Box(
                low=-0.5, 
                high=0.5, 
                shape=(2,), 
                dtype=np.float32
            )
            self.action_spaces = {agent: action_space for agent in self.possible_agents}
            
            # 推力缩放因子（将 [-0.5, 0.5] 映射到实际物理力）
            default_thrust_scale = config.get("thrust_scale", 200.0)
            self.thrust_scale_predator = config.get("thrust_scale_predator", default_thrust_scale)
            self.thrust_scale_prey = config.get("thrust_scale_prey", default_thrust_scale)
            self.soft_speed_limit_predator = config.get("soft_speed_limit_predator", None)
            self.soft_speed_limit_prey = config.get("soft_speed_limit_prey", None)
            self.drag_coefficient = config.get("drag_coefficient", 0.0)
            
            # 保留 action_to_move_tuple 用于兼容性（虽然不使用）
            self.action_to_move_tuple = None
            self.num_actions = 2  # 2D 推力向量
            
            if self.verbose_movement:
                # print(f"[INIT] Continuous action space: thrust [-0.5, 0.5]^2, scale={self.thrust_scale}")
                pass
        else:
            # === 离散空间：9 个离散动作 ===
            self.action_to_move_tuple: Dict[int, Tuple[int, int]] = {
                0: (-1, -1),
                1: (-1, 0),
                2: (-1, 1),
                3: (0, -1),
                4: (0, 0),
                5: (0, 1),
                6: (1, -1),
                7: (1, 0),
                8: (1, 1),
            }
            action_space = gymnasium.spaces.Discrete(len(self.action_to_move_tuple))
            self.action_spaces = {agent: action_space for agent in self.possible_agents}
            self.num_actions = len(self.action_to_move_tuple)


        
        # === 16. Initialize position and energy dictionaries ===
        self.agent_positions: Dict[AgentID, Tuple[float, float]] = {}
        self.predator_positions: Dict[AgentID, Tuple[float, float]] = {}
        self.prey_positions: Dict[AgentID, Tuple[float, float]] = {}
        self.grass_positions: Dict[AgentID, Tuple[float, float]] = {}
        self.agent_energies: Dict[AgentID, float] = {}
        self.grass_energies: Dict[AgentID, float] = {}
        self._grass_cell_size: float = max(1.0, float(self.grass_perception_radius) / 2.0)
        self._grass_spatial_index: Dict[Tuple[int, int], List[str]] = {}
        self._grass_spatial_index_dirty: bool = True
        self._grass_radius_squared: float = float(self.grass_perception_radius) ** 2
        
        # === 17. Grid world state ===
        self.grid_world_state_shape: Tuple[int, int, int] = (
            self.num_obs_channels,
            self.grid_size,
            self.grid_size,
        )
        self.initial_grid_world_state: NDArray[np.float64] = np.zeros(self.grid_world_state_shape, dtype=np.float64)
        self.grid_world_state: NDArray[np.float64] = self.initial_grid_world_state.copy()
        
        # === 18. Mapping actions to movements ===
        # self.num_actions = len(self.action_to_move_tuple)
        self.agents_just_ate = set()
        
        # === 19. Pymunk 物理空间初始化（仅在启用连续空间时） ===
        # === 19. Pymunk 物理空间初始化（仅在启用连续空间时） ===
        # === 19. Pymunk 物理空间初始化（仅在启用连续空间时） ===
        self.space = None
        self.agent_bodies: Dict[AgentID, pymunk.Body] = {}
        self.agent_shapes: Dict[AgentID, pymunk.Shape] = {}
        self.wall_shapes: List[pymunk.Shape] = []

        self.damping = config.get("damping", 0.9)

        if self.enable_continuous_space:
            # 创建物理空间
            self.space = pymunk.Space()
            self.space.gravity = (0, 0)
            self.space.damping = self.damping
            
            # 设置碰撞类型常量
            # 碰撞类型常量
            self.COLLISION_TYPE_PREDATOR = 1
            self.COLLISION_TYPE_PREY = 2
            self.COLLISION_TYPE_GRASS = 3  # 新增草的碰撞类型
            self.COLLISION_TYPE_WALL = 4
            
            # 初始化推力与转向成本记录
            self.thrust_costs = {}
            self.turn_penalties = {}
            self.agent_last_heading = {}
            
            # 初始化食物和毒物列表
            self.population_history = {
                'steps': [],
                'predators': [],
                'prey': []
            }            
            if self.verbose_movement:
                # print("[INIT] Pymunk space initialized")
                pass

        self._log_info(
            "InitComplete",
            enable_continuous_space=self.enable_continuous_space,
            world_size=(self.world_width, self.world_height),
            grid_size=self.grid_size,
            possible_agents=len(self.possible_agents),
            initial_agents=len(self.agents),
            max_steps=self.max_steps,
            action_space_type="continuous" if self.enable_continuous_space else "discrete",
            log_path=str(self.debug_log_path),
        )


    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        self._log_info(
            "ResetBegin",
            seed=seed,
            options=self._short_repr(options) if options is not None else None,
            enable_continuous_space=self.enable_continuous_space,
            n_possible_predators=self.n_possible_predators,
            n_possible_prey=self.n_possible_prey,
        )

        # === 初始化所有追踪字典 ===
        self.agent_population_id: Dict[AgentID, int] = {}
        self.agent_last_reproduction_step: Dict[AgentID, int] = {}
        self.agent_generation: Dict[AgentID, int] = {}
        self.agent_age: Dict[AgentID, int] = {}
        self.agent_algorithm: Dict[AgentID, str] = {}
        self.agent_wants_to_mate: Dict[AgentID, bool] = {}
        self.agent_steps_since_last_meal: Dict[AgentID, int] = {}
        self.agent_last_heading: Dict[AgentID, float] = {}
        self.turn_penalties = {}
        self.thrust_costs = {}
        
        # === 草的追踪字典 ===
        self.grass_age: Dict[str, int] = {}
        self.grass_generation: Dict[str, int] = {}
        
        super().reset(seed=seed)
        self.current_step = 0
        self.rng = np.random.default_rng(seed)
        
        if self.enable_continuous_space and self.space is None:
            self.space = pymunk.Space()
            self.space.gravity = (0, 0)
            self.space.damping = self.damping
        
        # === 连续空间：创建物理边界墙 ===
        if self.enable_continuous_space and self.space is not None:
            # 清空旧的物理对象
            for shape in self.space.shapes:
                self.space.remove(shape)
            for body in self.space.bodies:
                self.space.remove(body)
            
            self.agent_bodies.clear()
            self.agent_shapes.clear()
            self.wall_shapes.clear()
            
            # 创建静态边界墙
            static_body = self.space.static_body
            wall_thickness = 10
            
            # 上墙
            wall_top = pymunk.Segment(
                static_body,
                (0, 0),
                (self.world_width, 0),
                wall_thickness
            )
            wall_top.collision_type = self.COLLISION_TYPE_WALL
            wall_top.friction = 0.5
            wall_top.elasticity = 0.5
            
            # 下墙
            wall_bottom = pymunk.Segment(
                static_body,
                (0, self.world_height),
                (self.world_width, self.world_height),
                wall_thickness
            )
            wall_bottom.collision_type = self.COLLISION_TYPE_WALL
            wall_bottom.friction = 0.5
            wall_bottom.elasticity = 0.5
            
            # 左墙
            wall_left = pymunk.Segment(
                static_body,
                (0, 0),
                (0, self.world_height),
                wall_thickness
            )
            wall_left.collision_type = self.COLLISION_TYPE_WALL
            wall_left.friction = 0.5
            wall_left.elasticity = 0.5
            
            # 右墙
            wall_right = pymunk.Segment(
                static_body,
                (self.world_width, 0),
                (self.world_width, self.world_height),
                wall_thickness
            )
            wall_right.collision_type = self.COLLISION_TYPE_WALL
            wall_right.friction = 0.5
            wall_right.elasticity = 0.5
            
            # 添加到空间
            self.wall_shapes = [wall_top, wall_bottom, wall_left, wall_right]
            for wall in self.wall_shapes:
                self.space.add(wall)
            
            if self.verbose_movement:
                # print("[RESET] Created boundary walls")
                pass
        
        # === 初始化网格世界状态 ===
        self.grid_world_state = self.initial_grid_world_state.copy()
        
        # === 定义可能的agent列表 ===
        self.possible_agents: List[AgentID] = [
            f"predator_{i}" for i in range(self.n_possible_predators)
        ] + [f"prey_{j}" for j in range(self.n_possible_prey)]
        
        # === 定义初始活跃agent列表 ===
        self.agents = [
            f"predator_{i}" for i in range(self.n_initial_active_predator)
        ] + [f"prey_{j}" for j in range(self.n_initial_active_prey)]
        
        # === 重置草实体ID列表，避免上轮追加的ID残留 ===
        self.grass_agents = [f"grass_{k}" for k in range(self.initial_num_grass)]
        
        self.agent_positions: Dict[AgentID, Tuple[int, int]] = {}
        self.agent_energies: Dict[AgentID, float] = {}
        self.agent_last_energy: Dict[AgentID, float] = {}
        self.agent_recent_energy_delta: Dict[AgentID, float] = {}
        self.retired_agents = set()
        self.next_free_idx = {
            "predator": self.n_initial_active_predator,
            "prey": self.n_initial_active_prey,
        }
        
        # === 重置累积奖励 ===
        self.cumulative_rewards: Dict[AgentID, float] = {agent_id: 0 for agent_id in self.agents}
        
        # === 清空种群历史 ===
        # === 清空种群历史（分种群）===
        self.population_history = {'steps': []}
        for agent_type in ["predator", "prey"]:
            for pop_id in range(self.n_populations):
                pop_key = f"{agent_type}_{pop_id}"
                self.population_history[pop_key] = []

        
        # === 生成随机位置的辅助函数 ===
        def generate_random_positions(grid_size: int, num_positions: int):
            """生成唯一的网格位置"""
            if num_positions > grid_size * grid_size:
                raise ValueError("Cannot place more unique positions than grid cells.")
            rng = np.random.default_rng(seed)
            positions = set()
            while len(positions) < num_positions:
                pos = tuple(rng.integers(0, grid_size, size=2))
                positions.add(pos)
            return list(positions)
        
        def generate_random_positions_continuous(width, height, num_positions, radius):
            """生成唯一的连续空间位置"""
            positions = set()
            max_attempts = num_positions * 100
            attempts = 0
            while len(positions) < num_positions and attempts < max_attempts:
                x = self.rng.uniform(radius, width - radius)
                y = self.rng.uniform(radius, height - radius)
                pos = (float(x), float(y))
                
                # 简单的碰撞检测（确保不重叠）
                too_close = False
                for existing_pos in positions:
                    dist = ((pos[0] - existing_pos[0])**2 + (pos[1] - existing_pos[1])**2)**0.5
                    if dist < radius * 2:
                        too_close = True
                        break
                
                if not too_close:
                    positions.add(pos)
                attempts += 1
            
            if len(positions) < num_positions:
                self._log_warning(
                    "PositionSamplingFallback",
                    requested=num_positions,
                    generated=len(positions),
                    max_attempts=max_attempts,
                )
                while len(positions) < num_positions:
                    x = self.rng.uniform(radius, width - radius)
                    y = self.rng.uniform(radius, height - radius)
                    positions.add((float(x), float(y)))
            
            return list(positions)
        
        # === 生成所有实体的位置 ===
        total_entities = self.n_initial_active_predator + self.n_initial_active_prey + self.initial_num_grass
        
        if self.enable_continuous_space:
            all_positions = generate_random_positions_continuous(
                self.world_width, 
                self.world_height, 
                total_entities, 
                self.agent_radius
            )
        else:
            all_positions = generate_random_positions(self.grid_size, total_entities)
        
        # === 分配位置 ===
        predator_positions = all_positions[:self.n_initial_active_predator]
        prey_positions = all_positions[
            self.n_initial_active_predator : self.n_initial_active_predator + self.n_initial_active_prey
        ]
        grass_positions = all_positions[self.n_initial_active_predator + self.n_initial_active_prey:]

        self._log_debug(
            "ResetPositionsGenerated",
            predator_samples=predator_positions[:3],
            prey_samples=prey_positions[:3],
            grass_samples=grass_positions[:3],
            total_entities=total_entities,
        )
        
        # ============================================================
        # === 新增：根据种群数量分配agent ===
        # ============================================================
        
        # 计算每个种群应该有多少个agent
        predator_distribution = self._distribute_agents(
            self.n_initial_active_predator, 
            self.n_populations
        )
        
        prey_distribution = self._distribute_agents(
            self.n_initial_active_prey, 
            self.n_populations
        )
        
        if self.verbose_spawning:
            # print(f"[INIT] Predator distribution: {predator_distribution}")
            # print(f"[INIT] Prey distribution: {prey_distribution}")
            pass
        
        # === 初始化种群计数 ===
        self.population_counts: Dict[str, int] = {}
        for agent_type in ["predator", "prey"]:
            for pop_id in range(self.n_populations):
                key = f"{agent_type}_{pop_id}"
                self.population_counts[key] = 0
        
        # 根据分配初始化种群计数
        for pop_id, count in enumerate(predator_distribution):
            self.population_counts[f"predator_{pop_id}"] = count
        for pop_id, count in enumerate(prey_distribution):
            self.population_counts[f"prey_{pop_id}"] = count
        
        # ============================================================
        # === 按种群分配初始化 Predator ===
        # ============================================================
        predator_idx = 0
        for pop_id, pop_count in enumerate(predator_distribution):
            for _ in range(pop_count):
                agent = f"predator_{predator_idx}"
                position = predator_positions[predator_idx]
                
                # 基础属性
                self.agent_positions[agent] = position
                self.predator_positions[agent] = position
                self.agent_energies[agent] = self.initial_energy_predator
                self.agent_last_energy[agent] = self.initial_energy_predator
                self.agent_recent_energy_delta[agent] = 0.0
                
                # === 分配种群ID ===
                self.agent_population_id[agent] = pop_id
                self.agent_last_reproduction_step[agent] = -1000
                self.agent_generation[agent] = 0
                self.agent_age[agent] = 0
                self.agent_wants_to_mate[agent] = False
                self.agent_steps_since_last_meal[agent] = 0
                
                # === 从显示信息中获取算法名 ===
                pop_key = f"predator_{pop_id}"
                self.agent_algorithm[agent] = self.population_display_info.get(pop_key, "Random")
                
                # === 连续空间：创建物理 Body ===
                if self.enable_continuous_space and self.space is not None:
                    mass = 1.0
                    moment = pymunk.moment_for_circle(mass, 0, self.agent_radius)
                    body = pymunk.Body(mass, moment)
                    body.position = position
                    
                    shape = pymunk.Circle(body, self.agent_radius)
                    shape.collision_type = self.COLLISION_TYPE_PREDATOR
                    shape.friction = 0.5
                    shape.elasticity = 0.8
                    shape.agent_id = agent
                    shape.filter = pymunk.ShapeFilter()
                    
                    self.agent_bodies[agent] = body
                    self.agent_shapes[agent] = shape
                    
                    self.space.add(body, shape)
                else:
                    # 离散网格：更新网格状态
                    self.grid_world_state[1, *position] = self.initial_energy_predator
                
                predator_idx += 1
                
                if self.verbose_spawning:
                    # print(f"[INIT] {agent} -> pop={pop_id}, algo={self.agent_algorithm[agent]}")
                    pass
        
        # ============================================================
        # === 按种群分配初始化 Prey ===
        # ============================================================
        prey_idx = 0
        for pop_id, pop_count in enumerate(prey_distribution):
            for _ in range(pop_count):
                agent = f"prey_{prey_idx}"
                position = prey_positions[prey_idx]
                
                # 基础属性
                self.agent_positions[agent] = position
                self.prey_positions[agent] = position
                self.agent_energies[agent] = self.initial_energy_prey
                self.agent_last_energy[agent] = self.initial_energy_prey
                self.agent_recent_energy_delta[agent] = 0.0
                
                # === 分配种群ID ===
                self.agent_population_id[agent] = pop_id
                self.agent_last_reproduction_step[agent] = -1000
                self.agent_generation[agent] = 0
                self.agent_age[agent] = 0
                self.agent_wants_to_mate[agent] = False
                self.agent_steps_since_last_meal[agent] = 0
                
                # === 从显示信息中获取算法名 ===
                pop_key = f"prey_{pop_id}"
                self.agent_algorithm[agent] = self.population_display_info.get(pop_key, "Random")
                
                # === 连续空间：创建物理 Body ===
                if self.enable_continuous_space and self.space is not None:
                    mass = 1.0
                    moment = pymunk.moment_for_circle(mass, 0, self.agent_radius)
                    body = pymunk.Body(mass, moment)
                    body.position = position
                    
                    shape = pymunk.Circle(body, self.agent_radius)
                    shape.collision_type = self.COLLISION_TYPE_PREY
                    shape.friction = 0.5
                    shape.elasticity = 0.8
                    shape.agent_id = agent
                    shape.filter = pymunk.ShapeFilter()
                    
                    self.agent_bodies[agent] = body
                    self.agent_shapes[agent] = shape
                    
                    self.space.add(body, shape)
                else:
                    # 离散网格：更新网格状态
                    self.grid_world_state[2, *position] = self.initial_energy_prey
                
                prey_idx += 1
                
                if self.verbose_spawning:
                    # print(f"[INIT] {agent} -> pop={pop_id}, algo={self.agent_algorithm[agent]}")
                    pass
        
        # ============================================================
        # === 初始化草（保持不变）===
        # ============================================================
        self.grass_positions = {}
        self.grass_energies = {}
        self.grass_bodies: Dict[str, pymunk.Body] = {}
        self.grass_shapes: Dict[str, pymunk.Shape] = {}
        self._grass_spatial_index = {}
        self._grass_spatial_index_dirty = True

        self.current_num_grass = 0
        self.grass_home_positions = {}
        self.inactive_grass.clear()
        self.grass_respawn_timers.clear()

        for i, grass in enumerate(self.grass_agents):
            position = grass_positions[i]
            self.grass_home_positions[grass] = position
            initial_energy = self._get_initial_grass_energy()
            self._activate_grass_patch(grass, position, initial_energy, reset_generation=True)

        self._mark_grass_index_dirty()

        # === 初始化计数器 ===
        self.current_num_prey = self.n_initial_active_prey
        self.current_num_predators = self.n_initial_active_predator
        self.current_num_grass = len(self.grass_positions)
        self._grass_radius_squared = float(self.grass_perception_radius) ** 2
        
        # === 生成初始观察 ===
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        
        # ============================================================
        # === 连续空间：注册碰撞回调 ===
        # ============================================================
        if self.enable_continuous_space and self.space is not None:
            # 捕食者-猎物碰撞
            def predator_prey_collision(arbiter, space, data):
                """捕食者抓到猎物"""
                shapes = arbiter.shapes
                
                predator_shape = shapes[0] if shapes[0].collision_type == self.COLLISION_TYPE_PREDATOR else shapes[1]
                prey_shape = shapes[1] if shapes[0].collision_type == self.COLLISION_TYPE_PREDATOR else shapes[0]
                
                predator_agent = getattr(predator_shape, 'agent_id', None)
                prey_agent = getattr(prey_shape, 'agent_id', None)
                
                if predator_agent and prey_agent:
                    if not hasattr(self, 'collisions_this_step'):
                        self.collisions_this_step = []
                    
                    self.collisions_this_step.append(('predator_catch_prey', predator_agent, prey_agent))
                    
                    if self.verbose_engagement:
                        # print(f"[COLLISION_DETECTED] {predator_agent} collided with {prey_agent}")
                        pass
                
                return False
            
            # 猎物-草碰撞
            def prey_grass_collision(arbiter, space, data):
                """猎物吃草"""
                shapes = arbiter.shapes
                prey_shape = shapes[0] if shapes[0].collision_type == self.COLLISION_TYPE_PREY else shapes[1]
                grass_shape = shapes[1] if shapes[0].collision_type == self.COLLISION_TYPE_PREY else shapes[0]
                
                prey_agent = getattr(prey_shape, 'agent_id', None)
                grass_id = getattr(grass_shape, 'grass_id', None)
                
                if prey_agent and grass_id:
                    if grass_id in self.grass_energies and self.grass_energies[grass_id] > 0:
                        if not hasattr(self, 'collisions_this_step'):
                            self.collisions_this_step = []
                        
                        self.collisions_this_step.append(('prey_eat_grass', prey_agent, grass_id))
                        
                        if self.verbose_engagement:
                            # print(f"[COLLISION_DETECTED] {prey_agent} collided with {grass_id}")
                            pass
                
                return True
            
            # 智能体-智能体碰撞
            def agent_agent_collision(arbiter, space, data):
                """智能体间碰撞回调"""
                if not self.enable_agent_collision:
                    return True
                
                shapes = arbiter.shapes
                agent1_id = getattr(shapes[0], 'agent_id', None)
                agent2_id = getattr(shapes[1], 'agent_id', None)
                
                if agent1_id and agent2_id:
                    impulse = arbiter.total_impulse.length
                    
                    if not hasattr(self, 'collision_damages'):
                        self.collision_damages = {}
                    
                    if agent1_id not in self.collision_damages:
                        self.collision_damages[agent1_id] = 0
                    if agent2_id not in self.collision_damages:
                        self.collision_damages[agent2_id] = 0
                    
                    damage = self.collision_damage * (impulse / 1000.0)
                    self.collision_damages[agent1_id] += damage
                    self.collision_damages[agent2_id] += damage
                    
                    if self.verbose_engagement:
                        # print(f"[COLLISION] {agent1_id} <-> {agent2_id}, impulse={impulse:.2f}, damage={damage:.4f}")
                        pass
                
                return True
            
            # 智能体-墙碰撞
            def agent_wall_collision(arbiter, space, data):
                """智能体撞墙回调"""
                shapes = arbiter.shapes
                agent_shape = shapes[0] if hasattr(shapes[0], 'agent_id') else shapes[1]
                agent_id = getattr(agent_shape, 'agent_id', None)
                
                if agent_id:
                    impulse = arbiter.total_impulse.length
                    
                    if not hasattr(self, 'wall_collision_damages'):
                        self.wall_collision_damages = {}
                    
                    if agent_id not in self.wall_collision_damages:
                        self.wall_collision_damages[agent_id] = 0
                    
                    damage = self.wall_collision_damage * (impulse / 1000.0)
                    self.wall_collision_damages[agent_id] += damage
                    
                    if self.verbose_engagement:
                        # print(f"[WALL_COLLISION] {agent_id}, impulse={impulse:.2f}, damage={damage:.4f}")
                        pass
                
                return True
            
            # 注册处理器
            handler_predator_prey = self.space.add_collision_handler(
                self.COLLISION_TYPE_PREDATOR,
                self.COLLISION_TYPE_PREY
            )
            handler_predator_prey.begin = predator_prey_collision
            
            handler_prey_grass = self.space.add_collision_handler(
                self.COLLISION_TYPE_PREY,
                self.COLLISION_TYPE_GRASS
            )
            handler_prey_grass.begin = prey_grass_collision
            
            handler_predator_predator = self.space.add_collision_handler(
                self.COLLISION_TYPE_PREDATOR,
                self.COLLISION_TYPE_PREDATOR
            )
            handler_predator_predator.post_solve = agent_agent_collision
            
            handler_prey_prey = self.space.add_collision_handler(
                self.COLLISION_TYPE_PREY,
                self.COLLISION_TYPE_PREY
            )
            handler_prey_prey.post_solve = agent_agent_collision
            
            handler_predator_wall = self.space.add_collision_handler(
                self.COLLISION_TYPE_PREDATOR,
                self.COLLISION_TYPE_WALL
            )
            handler_predator_wall.post_solve = agent_wall_collision
            
            handler_prey_wall = self.space.add_collision_handler(
                self.COLLISION_TYPE_PREY,
                self.COLLISION_TYPE_WALL
            )
            handler_prey_wall.post_solve = agent_wall_collision
            
            if self.verbose_engagement:
                # print("[RESET] Collision handlers registered (including agent-agent and wall)")
                pass
        
        # === 初始化碰撞记录 ===
        self.collisions_this_step = []
        self.collision_damages = {}
        self.wall_collision_damages = {}

        predator_samples = list(self.predator_positions.items())[:3]
        prey_samples = list(self.prey_positions.items())[:3]
        grass_samples = list(self.grass_positions.items())[:3]

        self._log_info(
            "ResetComplete",
            predator_count=self.current_num_predators,
            prey_count=self.current_num_prey,
            grass_count=self.current_num_grass,
            predator_samples=predator_samples,
            prey_samples=prey_samples,
            grass_samples=grass_samples,
        )

        self._mark_grass_index_dirty()

        infos = {agent: {} for agent in observations}
        return observations, infos




    def step(self, action_dict):
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        self._log_debug(
            "StepBegin",
            incoming_actions=list(action_dict.keys()),
            active_agents=len(self.agents),
            current_step=self.current_step,
        )
        # 初始化推力消耗记录
        if self.enable_continuous_space:
            self.thrust_costs = {}
            self.turn_penalties = {}
            # 确保碰撞伤害字典存在（不清空，留到 Step 1 处理完后清空）
            if not hasattr(self, 'collision_damages'):
                self.collision_damages = {}
            if not hasattr(self, 'wall_collision_damages'):
                self.wall_collision_damages = {}
            # self.collision_damages = {}
            # self.wall_collision_damages = {}
        # step 0: check for truncation

        if self.current_step >= self.max_steps:
            self._log_info(
                "MaxStepsReached",
                current_step=self.current_step,
                max_steps=self.max_steps,
                active_agents=len(self.agents),
            )
            alive_predators = [agent for agent in self.agents if "predator" in agent]
            alive_prey = [agent for agent in self.agents if "prey" in agent]
            self._apply_survival_bonus(alive_predators, self.survival_bonus_predator, rewards)
            self._apply_survival_bonus(alive_prey, self.survival_bonus_prey, rewards)
            for agent in self.possible_agents:
                if agent in self.agents:  # Active agents get a real observation
                    observations[agent] = self._get_observation(agent)
                else:  # Previously removed agents get a zero-filled observation
                    if self.enable_continuous_space:
                        # 连续空间：返回零传感器观察
                        total_dim = getattr(self, "sensor_obs_dim", self.n_sensors * 12 + self.self_state_dim)
                        observations[agent] = np.zeros(total_dim, dtype=np.float32)
                    else:
                        # 离散空间：返回零网格观察
                        observation_range = self.predator_obs_range if "predator" in agent else self.prey_obs_range
                        observations[agent] = np.zeros((self.num_obs_channels, observation_range, observation_range), dtype=np.float64)
                rewards[agent] = 0.0
                truncations[agent] = True
                terminations[agent] = False
            # Mark global truncation and return immediately
            truncations["__all__"] = True
            terminations["__all__"] = False
            return observations, rewards, terminations, truncations, infos
        # === Step 0.5: 过滤 action_dict，只保留活跃智能体的动作 ===
        inactive_agents = [agent for agent in action_dict if agent not in self.agents]
        if inactive_agents:
            self._log_warning(
                "ActionsForInactiveAgents",
                inactive_agents=inactive_agents,
                active_agents_snapshot=list(self.agents),
            )
        action_dict = {agent: action for agent, action in action_dict.items() if agent in self.agents}
        # === 新增：增加所有活跃智能体的年龄 ===
        for agent in self.agents:
            if agent in self.agent_age:
                self.agent_age[agent] += 1
        # For stepwise display eating in grid
        self.agents_just_ate.clear()

        # Step 1: Process energy depletion due to time steps
        # Step 1: Process energy depletion due to time steps
        # Step 1: Process energy depletion due to time steps
        # Step 1: Process energy depletion due to time steps
        # Step 1: Process energy depletion due to time steps
        for agent, action in action_dict.items():
            if "predator" in agent:
                if agent not in self.agent_energies:
                    continue
                
                self.agent_energies[agent] -= self.energy_loss_per_step_predator
                
                if not self.enable_continuous_space:
                    if agent in self.agent_positions:
                        self.grid_world_state[1, *self.agent_positions[agent]] = self.agent_energies[agent]
            
            elif "prey" in agent:
                if agent not in self.agent_energies:
                    continue
                
                self.agent_energies[agent] -= self.energy_loss_per_step_prey
                
                if not self.enable_continuous_space:
                    if agent in self.agent_positions:
                        self.grid_world_state[2, *self.agent_positions[agent]] = self.agent_energies[agent]

        # === 草的生长与繁殖 ===
        if self.fixed_grass_mode:
            self._update_fixed_grass()
        else:
            for grass in list(self.grass_positions.keys()):
                if grass not in self.grass_energies:
                    continue

                density_ratio = self._estimate_grass_density(grass)
                growth = self.grass_base_growth_rate * max(0.0, 1.0 - density_ratio)
                updated_energy = self.grass_energies[grass] + growth
                updated_energy = min(updated_energy, self.grass_max_energy)

                if updated_energy > 0.95 * self.grass_max_energy:
                    updated_energy = max(0.0, updated_energy - self.grass_decay_rate)

                self.grass_energies[grass] = updated_energy

                if not self.enable_continuous_space and grass in self.grass_positions:
                    self.grid_world_state[3, *self.grass_positions[grass]] = updated_energy

                if grass in self.grass_age:
                    self.grass_age[grass] += 1

                if (
                    self.enable_grass_reproduction
                    and updated_energy > self.grass_reproduce_threshold
                    and self.grass_energies[grass] >= self.grass_reproduce_cost
                ):
                    reproduction_prob = self.grass_base_reproduce_prob * max(0.0, 1.0 - density_ratio)
                    reproduction_prob = float(np.clip(reproduction_prob, 0.0, 1.0))
                    if reproduction_prob > 0.0 and self.rng.random() < reproduction_prob:
                        offspring = self._create_grass_offspring(grass)
                        if offspring is not None:
                            self.grass_energies[grass] = max(
                                0.0,
                                self.grass_energies[grass] - self.grass_reproduce_cost,
                            )

                if self.grass_energies[grass] <= self.grass_min_energy:
                    self._remove_grass(grass)
        # === 连续空间：基础代谢消耗和伤害 ===
        if self.enable_continuous_space:
            for agent in list(self.agents):
                if agent not in self.agent_energies:
                    continue
                
                agent_type = "predator" if "predator" in agent else "prey"

                # 基础代谢
                if agent_type == "predator":
                    self.agent_energies[agent] -= self.base_metabolism_predator
                else:
                    self.agent_energies[agent] -= self.base_metabolism_prey
                
                # 移动消耗（基于速度）
                speed = 0.0
                if agent in self.agent_bodies:
                    body = self.agent_bodies[agent]
                    speed = body.velocity.length
                movement_factor = (
                    self.movement_cost_factor_predator
                    if agent_type == "predator"
                    else self.movement_cost_factor_prey
                )
                movement_cost = movement_factor * speed
                self.agent_energies[agent] -= movement_cost
                
                # 推力成本（平方惩罚）
                thrust_magnitude = self.thrust_costs.get(agent, 0.0)
                if thrust_magnitude > 0.0:
                    thrust_factor = (
                        self.thrust_cost_factor_predator
                        if agent_type == "predator"
                        else self.thrust_cost_factor_prey
                    )
                    self.agent_energies[agent] -= thrust_factor * (thrust_magnitude ** 2)
                
                # 转向惩罚
                turn_delta = self.turn_penalties.get(agent, 0.0)
                if turn_delta > 0.0:
                    turn_factor = (
                        self.turn_penalty_factor_predator
                        if agent_type == "predator"
                        else self.turn_penalty_factor_prey
                    )
                    self.agent_energies[agent] -= turn_factor * turn_delta
                
                # === 碰撞伤害（直接扣除能量）===
                if agent in self.collision_damages:
                    collision_damage = self.collision_damages[agent]
                    self.agent_energies[agent] -= collision_damage
                    
                    if self.verbose_engagement:
                        # print(f"[DAMAGE] {agent} collision damage: {collision_damage:.4f}")
                        pass
                
                # === 撞墙伤害（直接扣除能量）===
                if agent in self.wall_collision_damages:
                    wall_damage = self.wall_collision_damages[agent]
                    self.agent_energies[agent] -= wall_damage
                    
                    if self.verbose_engagement:
                        # print(f"[DAMAGE] {agent} wall collision damage: {wall_damage:.4f}")
                        pass
                
                # === 能量上限 ===
                if self.enable_max_energy:  # ← 添加条件判断
                    max_energy = self.max_energy_predator if "predator" in agent else self.max_energy_prey
                    self.agent_energies[agent] = min(self.agent_energies[agent], max_energy)
                
                # === 饥饿系统（直接扣除能量）===
                if self.enable_hunger and agent in self.agent_steps_since_last_meal:
                    self.agent_steps_since_last_meal[agent] += 1
                    
                    max_steps = (self.max_steps_without_food_predator if "predator" in agent 
                                else self.max_steps_without_food_prey)
                    
                    if self.agent_steps_since_last_meal[agent] > max_steps:
                        hunger_steps = self.agent_steps_since_last_meal[agent] - max_steps
                        hunger_damage = self.hunger_damage * hunger_steps
                        
                        self.agent_energies[agent] -= hunger_damage
                        
                        if self.verbose_engagement and hunger_steps % 10 == 0:
                            # print(f"[HUNGER] {agent} starving for {hunger_steps} steps, damage={hunger_damage:.2f}")
                            pass
            
            # === 清空碰撞伤害记录 ===
            self.collision_damages.clear()
            self.wall_collision_damages.clear()
            self.turn_penalties.clear()
        # Step 2: Process movements
        # Step 2: Process movements
        for agent, action in action_dict.items():
            if agent in self.agent_positions:
                old_position = self.agent_positions[agent]
                new_position = self._get_move(agent, action)
                
                if not self.enable_continuous_space:
                    # 离散空间：直接更新位置
                    self.agent_positions[agent] = new_position
                    move_cost = self._get_movement_energy_cost(agent, old_position, new_position)
                    self.agent_energies[agent] -= move_cost
                    
                    if "predator" in agent:
                        self.predator_positions[agent] = new_position
                        self.grid_world_state[1, *old_position] = 0
                        self.grid_world_state[1, *new_position] = self.agent_energies[agent]
                    elif "prey" in agent:
                        self.prey_positions[agent] = new_position
                        self.grid_world_state[2, *old_position] = 0
                        self.grid_world_state[2, *new_position] = self.agent_energies[agent]
                    
                    if self.verbose_movement:
                        # print(f"[MOVE] Agent {agent} moved: {old_position} -> {new_position}.")
                        pass

        # === 连续空间：执行物理模拟 ===
        # === 连续空间：执行物理模拟 ===
        # === 连续空间:执行物理模拟 ===
        if self.enable_continuous_space and self.space is not None:
            # 物理模拟步进(60 FPS)
            dt = 1.0 / 60.0
            self.space.step(dt)

            for agent in list(self.agents):
                if agent in self.agent_bodies:
                    body = self.agent_bodies[agent]
                    speed = body.velocity.length

                    limit = self.soft_speed_limit_predator if "predator" in agent else self.soft_speed_limit_prey
                    if self.drag_coefficient > 0.0 and speed > 0.0:
                        if limit is not None and speed > limit:
                            excess = speed - limit
                            reduction_factor = max(0.0, 1.0 - self.drag_coefficient * excess)
                            body.velocity *= reduction_factor
                        else:
                            # 轻微空气阻力，帮助逐渐减速
                            body.velocity *= max(0.0, 1.0 - self.drag_coefficient * 0.05)

                    # 同步位置
                    new_position = (float(body.position.x), float(body.position.y))
                    self.agent_positions[agent] = new_position

                    if "predator" in agent:
                        self.predator_positions[agent] = new_position
                    elif "prey" in agent:
                        self.prey_positions[agent] = new_position

        # === 连续空间：处理本步的碰撞事件 ===
        # === 连续空间：处理本步的碰撞事件 ===
        if self.enable_continuous_space and hasattr(self, 'collisions_this_step'):
            if self.collisions_this_step:
                self._log_debug(
                    "CollisionBatch",
                    collisions=self.collisions_this_step,
                )
            for collision in self.collisions_this_step:
                collision_type = collision[0]
                
                # === 捕食者抓到猎物 ===
                if collision_type == 'predator_catch_prey':
                    predator_agent, prey_agent = collision[1], collision[2]
                    
                    if predator_agent in self.agents and prey_agent in self.agents:
                        if self.verbose_engagement:
                            # print(f"[ENGAGE] {predator_agent} caught {prey_agent}!")
                            pass
                        
                        self.agents_just_ate.add(predator_agent)
                        
                        # 捕食者获得猎物能量
                        prey_energy = self.agent_energies[prey_agent]
                        efficiency = self.energy_transfer_efficiency_predator
                        energy_gained = prey_energy * efficiency
                        self.agent_energies[predator_agent] += energy_gained
                        # === 条件应用能量上限 ===
                        if self.enable_max_energy:  # ← 添加条件判断
                            self.agent_energies[predator_agent] = min(
                                self.agent_energies[predator_agent],
                                self.max_energy_predator
                            )                        
                        # 重置饥饿计数器
                        if predator_agent in self.agent_steps_since_last_meal:
                            self.agent_steps_since_last_meal[predator_agent] = 0
                        
                        # 猎物死亡
                        self.agent_energies[prey_agent] = 0
                        
                        # 奖励
                        if predator_agent not in rewards:
                            rewards[predator_agent] = 0
                        rewards[predator_agent] += self.reward_predator_catch_prey
                        self.cumulative_rewards[predator_agent] += self.reward_predator_catch_prey
                    # ... 现有的捕食者-猎物碰撞处理 ...
                
                # === 新增：猎物吃食物 ===
                # === 猎物吃食物 ===
                # === 修改：猎物吃草（适配连续空间）===
                elif collision_type == 'prey_eat_grass':
                    prey_agent, grass_id = collision[1], collision[2]
                    
                    if prey_agent in self.agents and grass_id in self.grass_energies:
                        if self.grass_energies[grass_id] > 0:
                            if self.verbose_engagement:
                                # print(f"[ENGAGE] {prey_agent} ate grass {grass_id} at {self.grass_positions[grass_id]}!")
                                pass
                            
                            self.agents_just_ate.add(prey_agent)
                            
                            # 应用能量转换效率
                            grass_energy = self.grass_energies[grass_id]
                            efficiency = self.energy_transfer_efficiency_prey
                            energy_gained = grass_energy * efficiency
                            self.agent_energies[prey_agent] += energy_gained
                            
                            # 限制最大能量
                            # === 条件应用能量上限 ===
                            if self.enable_max_energy:  # ← 添加条件判断
                                self.agent_energies[prey_agent] = min(
                                    self.agent_energies[prey_agent], 
                                    self.max_energy_prey
                                )
                            
                            # 重置饥饿计数器
                            if prey_agent in self.agent_steps_since_last_meal:
                                self.agent_steps_since_last_meal[prey_agent] = 0
                            
                            # # 增加健康
                            # if prey_agent in self.agent_health:
                            #     self.agent_health[prey_agent] = min(
                            #         self.agent_health[prey_agent] + grass_energy * 0.5,
                            #         self.max_health
                            #     )
                            
                            # 移除被吃掉的草
                            self._remove_grass(grass_id)
            # 清空碰撞记录
            self.collisions_this_step = []



        # 初始化移除集合
        if not hasattr(self, 'prey_to_remove'):
            self.prey_to_remove = set()
        # Step 3: Prepare agent removals (Prey caught, Energy depleted)
        # Step 3: Prepare agent removals (Prey caught, Energy depleted)

        # # === 新增：为所有活跃智能体初始化字典条目 ===
        # for agent in self.agents:
        #     if agent not in terminations:
        #         terminations[agent] = False
        #     if agent not in truncations:
        #         truncations[agent] = False
        #     if agent not in rewards:
        #         rewards[agent] = 0.0

        # === 修改：使用 agents 的副本迭代，避免迭代过程中修改 ===
        for agent in self.agents:  # ← 使用 list() 创建副本
            # # === 新增：安全检查 - 智能体可能已被移除 ===
            # if agent not in self.agent_positions:
            #     continue
            
            # # === 新增：安全检查 - 能量字典可能已被清理 ===
            # if agent not in self.agent_energies:
            #     continue
            
            # Agent has no energy left
            if self.agent_energies[agent] <= 0:
                if self.verbose_movement:
                    # print(f"[MOVE] {agent} at {self.agent_positions[agent]} ran out of energy and is removed.")
                    pass
                self._log_debug(
                    "AgentEnergyDepleted",
                    agent=agent,
                    position=self.agent_positions.get(agent),
                    energy=self.agent_energies.get(agent),
                )
                
                observations[agent] = self._get_observation(agent)
                rewards[agent] = 0
                terminations[agent] = True
                truncations[agent] = False
                
                # === 更新种群计数 ===
                if agent in self.agent_population_id:
                    agent_type = "predator" if "predator" in agent else "prey"
                    pop_id = self.agent_population_id[agent]
                    pop_key = f"{agent_type}_{pop_id}"
                    if pop_key in self.population_counts:
                        self.population_counts[pop_key] = max(0, self.population_counts[pop_key] - 1)
                
                if "predator" in agent:
                    self.current_num_predators -= 1
                    if not self.enable_continuous_space:
                        self.grid_world_state[1, *self.agent_positions[agent]] = 0
                    if agent in self.predator_positions:
                        del self.predator_positions[agent]
                elif "prey" in agent:
                    self.current_num_prey -= 1
                    if not self.enable_continuous_space:
                        self.grid_world_state[2, *self.agent_positions[agent]] = 0
                    if agent in self.prey_positions:
                        del self.prey_positions[agent]
                
                # 清理所有字典
                if agent in self.agent_positions:
                    del self.agent_positions[agent]
                if agent in self.agent_energies:
                    del self.agent_energies[agent]
                self.agent_last_energy.pop(agent, None)
                self.agent_recent_energy_delta.pop(agent, None)
                if agent in self.agent_population_id:
                    del self.agent_population_id[agent]
                if agent in self.agent_last_reproduction_step:
                    del self.agent_last_reproduction_step[agent]
                if agent in self.agent_generation:
                    del self.agent_generation[agent]
                if agent in self.agent_age:
                    del self.agent_age[agent]
                if agent in self.agent_algorithm:
                    del self.agent_algorithm[agent]
                if agent in self.agent_wants_to_mate:
                    del self.agent_wants_to_mate[agent]
                if agent in self.agent_steps_since_last_meal:
                    del self.agent_steps_since_last_meal[agent]
                
                # === 新增：清理物理 Body ===
                if self.enable_continuous_space:
                    if agent in self.agent_bodies:
                        body = self.agent_bodies[agent]
                        shape = self.agent_shapes[agent]
                        self.space.remove(shape, body)
                        del self.agent_bodies[agent]
                        del self.agent_shapes[agent]
                
                continue
            
            elif "predator" in agent:
                # === 再次检查：确保智能体仍在字典中 ===
                if agent not in self.agent_positions or agent not in self.agent_energies:
                    continue
                
                predator_position = self.agent_positions[agent]
                
                # === 连续空间：从碰撞记录中查找 ===
                caught_prey = None
                if self.enable_continuous_space:
                    if hasattr(self, 'prey_to_remove'):
                        for predator_a, prey_a in self.prey_to_remove:
                            if predator_a == agent:
                                caught_prey = prey_a
                                break
                else:
                    # === 离散网格：保持原有逻辑 ===
                    caught_prey = next(
                        (
                            prey
                            for prey, prey_position in self.agent_positions.items()
                            if "prey" in prey and np.array_equal(predator_position, prey_position)
                        ),
                        None,
                    )
                
                if caught_prey:
                    # ... 捕食逻辑（保持不变）...
                    # if self.verbose_engagement:
                    #     print(
                    #         f"[ENGAGE] {agent} caught {caught_prey} at {predator_position}! "
                    #         f"Predator Reward: {self.reward_predator_catch_prey}"
                    #     )
                    self.agents_just_ate.add(agent)

                    prey_energy = self.agent_energies.get(caught_prey, 0)
                    efficiency = self.energy_transfer_efficiency_predator
                    energy_gained = prey_energy * efficiency

                    rewards[agent] = self.reward_predator_catch_prey
                    self.cumulative_rewards[agent] += rewards[agent]
                    self.agent_energies[agent] += energy_gained
                    self.agent_energies[agent] = min(self.agent_energies[agent], self.max_energy_predator)

                    self._log_debug(
                        "PredatorCaughtPrey",
                        predator=agent,
                        prey=caught_prey,
                        predator_position=predator_position,
                        prey_energy=prey_energy,
                        energy_gained=energy_gained,
                        predator_energy=self.agent_energies.get(agent),
                    )

                    if agent in self.agent_steps_since_last_meal:
                        self.agent_steps_since_last_meal[agent] = 0
                    
                    if not self.enable_continuous_space:
                        self.grid_world_state[1, *predator_position] = self.agent_energies[agent]
                    
                    observations[caught_prey] = self._get_observation(caught_prey)
                    rewards[caught_prey] = self.penalty_prey_caught
                    self.cumulative_rewards[caught_prey] += rewards[caught_prey]
                    
                    terminations[caught_prey] = True
                    truncations[caught_prey] = False
                    self.current_num_prey -= 1
                    
                    if not self.enable_continuous_space:
                        self.grid_world_state[2, *self.agent_positions[caught_prey]] = 0
                    
                    del self.agent_positions[caught_prey]
                    if caught_prey in self.prey_positions:
                        del self.prey_positions[caught_prey]
                    del self.agent_energies[caught_prey]
                else:
                    # 没有捕到猎物
                    base_reward = self.reward_predator_step
                    
                    if self.enable_continuous_space and hasattr(self, 'thrust_costs'):
                        thrust_penalty = self.thrust_costs.get(agent, 0.0)
                        rewards[agent] = base_reward - thrust_penalty
                    else:
                        rewards[agent] = base_reward
                
                observations[agent] = self._get_observation(agent)
                self.cumulative_rewards[agent] += rewards[agent]
                terminations[agent] = False
                truncations[agent] = False
            
            elif "prey" in agent:
                # === 再次检查：确保智能体仍在字典中 ===
                if agent not in self.agent_positions or agent not in self.agent_energies:
                    continue
                
                if terminations.get(agent) is None or not terminations[agent]:
                    # ... 猎物逻辑（保持不变）...
                    prey_position = self.agent_positions[agent]
                    
                    # 草碰撞检测
                    if self.enable_continuous_space:
                        caught_grass = None
                        eating_distance = self.agent_radius * 2
                        
                        for grass, grass_position in self.grass_positions.items():
                            if self.grass_energies[grass] > 0:
                                dist = ((prey_position[0] - grass_position[0])**2 + 
                                    (prey_position[1] - grass_position[1])**2)**0.5
                                if dist < eating_distance:
                                    caught_grass = grass
                                    break
                    else:
                        caught_grass = next(
                            (
                                grass
                                for grass, grass_position in self.grass_positions.items()
                                if "grass" in grass and np.array_equal(prey_position, grass_position)
                            ),
                            None,
                        )
                    
                    if caught_grass:
                        if self.verbose_engagement:
                            # print(f"[ENGAGE] {agent} caught grass at {prey_position}! Prey Reward: 0.0")
                            pass
                        self.agents_just_ate.add(agent)
                        rewards[agent] = self.reward_prey_eat_grass
                        self.cumulative_rewards[agent] += rewards[agent]

                        grass_energy = self.grass_energies[caught_grass]
                        efficiency = self.energy_transfer_efficiency_prey
                        energy_gained = grass_energy * efficiency
                        self.agent_energies[agent] += energy_gained
                        self.agent_energies[agent] = min(self.agent_energies[agent], self.max_energy_prey)

                        self._log_debug(
                            "PreyAteGrass",
                            prey=agent,
                            grass=caught_grass,
                            prey_position=prey_position,
                            grass_energy=grass_energy,
                            energy_gained=energy_gained,
                            prey_energy=self.agent_energies.get(agent),
                        )
                        
                        if agent in self.agent_steps_since_last_meal:
                            self.agent_steps_since_last_meal[agent] = 0
                        
                        if not self.enable_continuous_space:
                            self.grid_world_state[2, *prey_position] = self.agent_energies[agent]
                        
                        self._remove_grass(caught_grass)
                    else:
                        rewards[agent] = self.reward_prey_step
                    observations[agent] = self._get_observation(agent)
                    self.cumulative_rewards[agent] += rewards[agent]
                    terminations[agent] = False
                    truncations[agent] = False

        # Step 3 结束
        self.prey_to_remove = set()
        # Step 4: Handle agent removals
        for agent in self.agents[:]:
            # === 修改：添加安全检查 ===
            agent_terminated = terminations.get(agent, False)
            if agent_terminated:
                if self.verbose_engagement:
                    print(f"[ENGAGE] Agent {agent} terminated!")
                self._log_debug(
                    "AgentRemovedFromActiveList",
                    agent=agent,
                    remaining_agents=len(self.agents) - 1,
                )
                if hasattr(self, "retired_agents"):
                    self.retired_agents.add(agent)
                self.agents.remove(agent)

        # Step 5: Spawning of new agents
        if self.enable_paired_reproduction:
            # Step 5: 配对繁殖系统
            # === 5.1: 检测繁殖意愿 ===
            for agent in list(self.agents):
                if agent not in self.agent_energies:
                    continue
                
                # 检查基本条件
                can_reproduce = self._check_reproduction_eligibility(agent)
                
                if can_reproduce:
                    self.agent_wants_to_mate[agent] = True
                    # if self.verbose_spawning:
                    #     # print(f"[MATE] {agent} wants to mate (E={self.agent_energies[agent]:.1f}, "
                    #         f"age={self.agent_age[agent]})")
                else:
                    self.agent_wants_to_mate[agent] = False

            # === 5.2: 寻找配对 ===
            potential_pairs = []

            for agent1 in list(self.agents):
                if not self.agent_wants_to_mate.get(agent1, False):
                    continue
                
                agent1_type = "predator" if "predator" in agent1 else "prey"
                agent1_pop = self.agent_population_id.get(agent1, 0)
                
                for agent2 in list(self.agents):
                    if agent1 >= agent2:  # 避免重复配对
                        continue
                    
                    if not self.agent_wants_to_mate.get(agent2, False):
                        continue
                    
                    agent2_type = "predator" if "predator" in agent2 else "prey"
                    agent2_pop = self.agent_population_id.get(agent2, 0)
                    
                    # 必须是同类型、同种群
                    if agent1_type != agent2_type or agent1_pop != agent2_pop:
                        continue
                    
                    # 检查距离
                    if self.enable_continuous_space:
                        pos1 = self.agent_positions[agent1]
                        pos2 = self.agent_positions[agent2]
                        distance = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
                        
                        if distance <= self.mating_distance:
                            potential_pairs.append((agent1, agent2, distance))
                    else:
                        # 离散空间：必须相邻
                        pos1 = np.array(self.agent_positions[agent1])
                        pos2 = np.array(self.agent_positions[agent2])
                        distance = np.abs(pos1 - pos2).sum()
                        
                        if distance <= 1:
                            potential_pairs.append((agent1, agent2, distance))

            # === 5.3: 执行繁殖 ===
            successfully_reproduced = set()

            for agent1, agent2, distance in potential_pairs:
                # 跳过已经繁殖过的智能体
                if agent1 in successfully_reproduced or agent2 in successfully_reproduced:
                    continue
                
                # 执行繁殖
                offspring = self._create_offspring(agent1, agent2)
                
                if offspring:
                    self._log_debug(
                        "OffspringCreated",
                        parent_a=agent1,
                        parent_b=agent2,
                        offspring=offspring,
                        distance=distance,
                    )
                    successfully_reproduced.add(agent1)
                    successfully_reproduced.add(agent2)
                    
                    # 重置繁殖意愿
                    self.agent_wants_to_mate[agent1] = False
                    self.agent_wants_to_mate[agent2] = False
                    
                    # 更新最后繁殖时间
                    self.agent_last_reproduction_step[agent1] = self.current_step
                    self.agent_last_reproduction_step[agent2] = self.current_step
                    
                    # 给予繁殖奖励
                    agent_type = "predator" if "predator" in agent1 else "prey"
                    reproduction_reward = (self.reproduction_reward_predator if agent_type == "predator" 
                                        else self.reproduction_reward_prey)
                    
                    if agent1 in rewards:
                        rewards[agent1] += reproduction_reward
                    else:
                        rewards[agent1] = reproduction_reward
                    self.cumulative_rewards[agent1] += reproduction_reward
                    
                    if agent2 in rewards:
                        rewards[agent2] += reproduction_reward
                    else:
                        rewards[agent2] = reproduction_reward
                    self.cumulative_rewards[agent2] += reproduction_reward
                    
                    if self.verbose_spawning:
                        # print(f"[REPRODUCE] {agent1} + {agent2} → {offspring}")
                        pass
        else:
            # Flag disabled: ensure mating state does not linger between steps
            self.agent_wants_to_mate.clear()
        # 6: Generate observations for all agents AFTER all engagements in the step
        # 6: Generate observations for all agents AFTER all engagements in the step
        for agent in self.agents:
            # ✅ 跳过已经有观察的智能体（被吃掉的、能量耗尽的等）
            if agent in observations:
                continue
            
            if agent in self.agent_positions:
                observations[agent] = self._get_observation(agent)

        # Global termination and truncation
        terminations["__all__"] = (self.current_num_prey <= 0) and (self.current_num_predators <= 0)

        # output only observations, rewards for active agents
        observations = {agent: observations[agent] for agent in self.agents if agent in observations}
        rewards = {agent: rewards[agent] for agent in self.agents if agent in rewards}
        terminations = {agent: terminations[agent] for agent in self.agents if agent in terminations}
        truncations = {agent: truncations[agent] for agent in self.agents if agent in truncations}
        truncations["__all__"] = False  # already handled at the beginning of the step

        # Global termination and truncation
        terminations["__all__"] = (self.current_num_prey <= 0) and (self.current_num_predators <= 0)

        for agent in self.agents:
            current_energy = self.agent_energies.get(agent, 0.0)
            last_energy = self.agent_last_energy.get(agent, current_energy)
            self.agent_recent_energy_delta[agent] = current_energy - last_energy
            self.agent_last_energy[agent] = current_energy
        # === 在这里添加！👇 ===
        # === 记录分种群历史 ===
        # 确保字典已初始化
        if 'steps' not in self.population_history:
            self.population_history['steps'] = []

        for agent_type in ["predator", "prey"]:
            for pop_id in range(self.n_populations):
                pop_key = f"{agent_type}_{pop_id}"
                if pop_key not in self.population_history:
                    self.population_history[pop_key] = []

        # 记录当前步
        self.population_history['steps'].append(self.current_step)

        # 记录每个种群的数量
        for pop_key, count in self.population_counts.items():
            if pop_key in self.population_history:  # ← 安全检查
                self.population_history[pop_key].append(count)
        # === 添加结束 ===
        terminated_agents = [agent for agent, flag in terminations.items() if agent != "__all__" and flag]
        truncated_agents = [agent for agent, flag in truncations.items() if agent != "__all__" and flag]
        reward_preview = {
            agent: rewards.get(agent)
            for agent in list(rewards.keys())[:5]
        }
        info_preview = {
            agent: list(infos.get(agent, {}).keys())
            for agent in list(infos.keys())[:5]
        }

        self._log_debug(
            "StepSummary",
            current_step=self.current_step,
            terminated_agents=terminated_agents,
            truncated_agents=truncated_agents,
            remaining_agents=len(self.agents),
            reward_preview=reward_preview,
            info_preview=info_preview,
        )

        if __debug__ and hasattr(self, "retired_agents"):
            assert all(agent not in self.retired_agents for agent in self.agents), (
                "Active agents should not include retired IDs: "
                f"{self.retired_agents.intersection(self.agents)}"
            )

        self.agents.sort()  # Sort agents

        # Increment step counter
        self.current_step += 1

        return observations, rewards, terminations, truncations, infos  # ← 在return之前添加

    def _get_movement_energy_cost(self, agent, current_position, new_position, distance_factor=0.1):
        """
        Calculate the energy cost for moving an agent.

        Args:
            current_position (np.array): Current position of the agent [x, y].
            new_position (np.array): New position of the agent [x, y].
            current_energy (float): Current energy level of the agent.
            distance_factor (float): Scaling factor for the movement energy cost based on distance.

        Returns:
            float: Energy cost of the move.
        """
        # current_energy = self.agent_energies[agent]
        # distance = math.sqrt((new_position[0] - current_position[0]) ** 2 + (new_position[1] - current_position[1]) ** 2)

        # Calculate the energy cost
        # energy_cost = distance * distance_factor * current_energy
        return 0  # energy_cost
    def _get_move(self, agent: AgentID, action) -> Tuple[float, float]:
        """
        Get the new position of the agent based on the action.
        在连续空间中应用推力，在离散空间中保持原有逻辑。
        
        Args:
            agent: 智能体 ID
            action: 连续空间为 [fx, fy]，离散空间为 int
            
        Returns:
            新位置 (x, y)
        """
        if self.enable_continuous_space and agent in self.agent_bodies:
            # === 连续空间：应用连续推力 ===
            if isinstance(action, np.ndarray):
                # 确保动作在 [-0.5, 0.5] 范围内
                action = np.clip(action, -0.5, 0.5)
                fx, fy = float(action[0]), float(action[1])
            else:
                # 兼容性：如果是其他类型
                fx, fy = 0.0, 0.0
            
            # 缩放推力到物理单位
            scale = self.thrust_scale_predator if "predator" in agent else self.thrust_scale_prey
            force_x = fx * scale
            force_y = fy * scale
            force = (force_x, force_y)
            
            body = self.agent_bodies[agent]
            
            # 应用推力
            body.apply_force_at_local_point(force, (0, 0))
            
            # 计算推力能量消耗
            thrust_magnitude = float(math.hypot(fx, fy))
            # 扣除推力能量（在 step 中统一处理）
            if not hasattr(self, 'thrust_costs'):
                self.thrust_costs = {}
            self.thrust_costs[agent] = thrust_magnitude
            
            if not hasattr(self, 'turn_penalties'):
                self.turn_penalties = {}
            if not hasattr(self, 'agent_last_heading'):
                self.agent_last_heading = {}
            
            if thrust_magnitude > 1e-6:
                current_heading = math.atan2(fy, fx)
                previous_heading = self.agent_last_heading.get(agent)
                if previous_heading is not None:
                    delta = math.atan2(
                        math.sin(current_heading - previous_heading),
                        math.cos(current_heading - previous_heading)
                    )
                    self.turn_penalties[agent] = abs(delta)
                else:
                    self.turn_penalties[agent] = 0.0
                self.agent_last_heading[agent] = current_heading
            else:
                self.turn_penalties[agent] = 0.0
            
            # if self.verbose_movement:
            #     print(f"[MOVE] {agent} thrust: ({fx:.3f}, {fy:.3f}), cost: {thrust_cost:.4f}")
            
            # 返回当前位置（实际位置会在物理模拟后更新）
            return (body.position.x, body.position.y)
        
        else:
            # === 离散网格：保持原有逻辑 ===
            if self.action_to_move_tuple is None:
                # 安全检查
                return self.agent_positions[agent]
            
            agent_type_nr = 1 if "predator" in agent else 2
            current_position = self.agent_positions[agent]
            move_vector = self.action_to_move_tuple[action]
            new_position = (current_position[0] + move_vector[0], current_position[1] + move_vector[1])
            new_position = tuple(np.clip(new_position, 0, self.grid_size - 1))
            
            if self.grid_world_state[agent_type_nr, *new_position] > 0:
                new_position = current_position
            
            return new_position

    def _get_observation(self, agent):
        """
        Generate an observation for the agent.
        连续空间返回传感器观察，离散空间返回网格观察。
        """
        if self.enable_continuous_space:
            return self._get_sensor_observation(agent)
        else:
            return self._get_grid_observation(agent)

    def _get_grid_observation(self, agent):
        """
        生成网格观察（离散空间）。
        """
        observation_range = self.predator_obs_range if "predator" in agent else self.prey_obs_range
        xp, yp = self.agent_positions[agent]
        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self._obs_clip(int(xp), int(yp), observation_range)
        
        observation = np.zeros((self.num_obs_channels, observation_range, observation_range), dtype=np.float64)
        observation[0].fill(1)
        observation[0, xolo:xohi, yolo:yohi] = 0
        observation[1:, xolo:xohi, yolo:yohi] = self.grid_world_state[1:, xlo:xhi, ylo:yhi]
        
        return observation

    def _get_sensor_observation(self, agent):
        """生成增强的传感器观察（n_sensors*12 + self_state_dim）"""
        if agent not in self.agent_positions:
            total_dim = getattr(self, "sensor_obs_dim", self.n_sensors * 12 + self.self_state_dim)
            return np.zeros(total_dim, dtype=np.float32)
        
        agent_pos = self.agent_positions[agent]
        agent_type = "predator" if "predator" in agent else "prey"
        agent_pop_id = self.agent_population_id.get(agent, 0)
        
        # 计算传感器角度
        sensor_angles = np.linspace(0, 2 * np.pi, self.n_sensors, endpoint=False)
        
        # 初始化观察数组
        total_dim = getattr(self, "sensor_obs_dim", self.n_sensors * 12 + self.self_state_dim)
        observation = np.zeros(total_dim, dtype=np.float32)
        
        # 一次性收集所有射线信息
        ray_results = []
        for angle in sensor_angles:
            result = self._cast_sensor_ray(
                agent_pos, 
                angle, 
                self.sensor_range,
                exclude_agent_id=agent
            )
            ray_results.append(result)
        
        # 生成各层观察
        env_layer = self._get_environment_layer(ray_results)
        predator_layer = self._get_predator_layer(ray_results, agent, agent_type)
        prey_layer = self._get_prey_layer(ray_results, agent, agent_type)
        mate_layer = self._get_mate_layer(ray_results, agent_type, agent_pop_id, agent)
        self_state = self._get_self_state(agent, agent_type)
        
        # 组装观察
        layer_span = self.n_sensors * 3
        offset = 0
        for layer in (env_layer, predator_layer, prey_layer, mate_layer):
            observation[offset:offset + layer_span] = layer
            offset += layer_span
        observation[offset:offset + self.self_state_dim] = self_state
        
        return observation
    
    def _obs_clip(self, x, y, observation_range):
        """
        Clip the observation window to the boundaries of the grid_world_state.
        """
        observation_offset = (observation_range - 1) // 2
        xld, xhd = x - observation_offset, x + observation_offset
        yld, yhd = y - observation_offset, y + observation_offset
        xlo, xhi = np.clip(xld, 0, self.grid_size - 1), np.clip(xhd, 0, self.grid_size - 1)
        ylo, yhi = np.clip(yld, 0, self.grid_size - 1), np.clip(yhd, 0, self.grid_size - 1)
        xolo, yolo = abs(np.clip(xld, -observation_offset, 0)), abs(np.clip(yld, -observation_offset, 0))
        xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
        return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1

    def _get_agent_by_position(self) -> dict:
        """
        Reverse the agent_positions dictionary to map positions to agents.

        Returns:
            dict: A dictionary where keys are positions (tuples) and values are agent IDs.
        """
        return {position: agent for agent, position in self.agent_positions.items()}

    def _remove_agent(self, agent: AgentID):
        """
        Removes an agent from all tracking dictionaries.
        """
        position = self.agent_positions[agent]
        del self.agent_positions[agent]
        del self.agent_energies[agent]
        self.agent_last_energy.pop(agent, None)
        self.agent_recent_energy_delta.pop(agent, None)

        if "predator" in agent:
            del self.predator_positions[position]
            self.current_num_predators -= 1
        elif "prey" in agent:
            del self.prey_positions[position]
            self.current_num_prey -= 1

    def _print_grid_from_positions(self):
        print(f"\nCurrent Grid State (IDs):  predators: {self.current_num_predators} prey: {self.current_num_prey}  \n")

        # Initialize empty grids (not transposed yet)
        predator_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        prey_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grass_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Populate Predator Grid
        for agent, pos in self.predator_positions.items():
            x, y = pos
            agent_num = int(agent.split("_")[1])
            predator_grid[y][x] = f"P{agent_num:02d}".center(5)

        # Populate Prey Grid
        for agent, pos in self.prey_positions.items():
            x, y = pos
            agent_num = int(agent.split("_")[1])
            prey_grid[y][x] = f"p{agent_num:02d}".center(5)

        # Populate Grass Grid
        for agent, pos in self.grass_positions.items():
            x, y = pos
            agent_num = int(agent.split("_")[1])
            grass_grid[y][x] = f"G{agent_num:02d}".center(5)

        # Transpose the grids (rows become columns)
        predator_grid = list(map(list, zip(*predator_grid)))
        prey_grid = list(map(list, zip(*prey_grid)))
        grass_grid = list(map(list, zip(*grass_grid)))

        # Print Headers
        print(
            f"{'Predators'.center(self.grid_size * 6)}   "
            f"{'Prey'.center(self.grid_size * 6)}   "
            f"{'Grass'.center(self.grid_size * 6)}"
        )
        print("=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6)

        # Print Transposed Grids
        for x in range(self.grid_size):  # Now iterating over transposed rows (original columns)
            predator_row = " ".join(predator_grid[x])
            prey_row = " ".join(prey_grid[x])
            grass_row = " ".join(grass_grid[x])
            print(f"{predator_row}     {prey_row}     {grass_row}")

        print("=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6)

    def _print_grid_from_state(self):
        print(f"\nCurrent Grid State (Energy Levels):  predators: {self.current_num_predators} prey: {self.current_num_prey} \n")

        # Initialize empty grids
        predator_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        prey_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grass_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Fill the grid (storing values in original order)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                predator_energy = self.grid_world_state[1, x, y]
                prey_energy = self.grid_world_state[2, x, y]
                grass_energy = self.grid_world_state[3, x, y]

                if predator_energy > 0:
                    predator_grid[y][x] = f"{predator_energy:4.2f}".center(5)
                if prey_energy > 0:
                    prey_grid[y][x] = f"{prey_energy:4.2f}".center(5)
                if grass_energy > 0:
                    grass_grid[y][x] = f"{grass_energy:4.2f}".center(5)

        # Transpose the grids (swap rows and columns)
        predator_grid = [[predator_grid[x][y] for x in range(self.grid_size)] for y in range(self.grid_size)]
        prey_grid = [[prey_grid[x][y] for x in range(self.grid_size)] for y in range(self.grid_size)]
        grass_grid = [[grass_grid[x][y] for x in range(self.grid_size)] for y in range(self.grid_size)]

        # Print Headers
        print(
            f"{'Predator '.center(self.grid_size * 6)}   "
            f"{'Prey'.center(self.grid_size * 6)}   "
            f"{'Grass'.center(self.grid_size * 6)}"
        )
        print("=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6)

        # Print Transposed Grids (rows become columns)
        for x in range(self.grid_size):  # Now iterating over transposed rows (original columns)
            predator_row = " ".join(predator_grid[x])
            prey_row = " ".join(prey_grid[x])
            grass_row = " ".join(grass_grid[x])
            print(f"{predator_row}     {prey_row}     {grass_row}")

        print("=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6)

    def _print_movement_table(
        self,
        action_dict,
        predator_position_after_action,
        prey_new_unresolved_positions,
        resolved_positions,
        colliding_predator_agents,
        colliding_prey_agents,
    ):
        """
        Prints the movement table for predators and prey, including actions, positions, and energy levels.
        """

        print("\nPredator Position Table:")
        print(
            "{:<12} {:<15} {:<15} {:<10} {:<15} {:<15} {:<15} {:<20}".format(
                "Agent", "Tuple", "Energy", "Array", "Action", "Action Array", "New", "Resolved"
            )
        )
        print("-" * 120)

        for i, (agent, position) in enumerate(self.predator_positions.items()):
            array_position = np.array(position)
            action_number = action_dict[agent]
            action_array = np.array(self.action_to_move_tuple[action_number])
            new_position = predator_position_after_action[i]
            resolved_position = resolved_positions[agent]  # Position after collision resolution
            energy = self.agent_energies[agent]

            print(
                "{:<12} {:<15} {:<15} {:<10} {:<15} {:<15} {:<15} {:<20}".format(
                    agent,
                    str(position),
                    f"{energy:.2f}",
                    str(array_position),
                    action_number,
                    str(action_array),
                    str(new_position),
                    str(resolved_position),
                )
            )

        print("-" * 120)
        print()
        print("Colliding Predators:", colliding_predator_agents)
        print()

        print("\nPrey Position Table:")
        print(
            "{:<12} {:<15} {:<15} {:<10} {:<15} {:<15} {:<15} {:<20}".format(
                "Agent", "Tuple", "Energy", "Array", "Action", "Action Array", "New", "Resolved"
            )
        )
        print("-" * 120)

        for i, (agent, position) in enumerate(self.prey_positions.items()):
            array_position = np.array(position)
            action_number = action_dict[agent]
            action_array = np.array(self.action_to_move_tuple[action_number])
            new_position = prey_new_unresolved_positions[i]
            resolved_position = resolved_positions[agent]  # Position after collision resolution
            energy = self.agent_energies[agent]

            print(
                "{:<12} {:<15} {:<15} {:<10} {:<15} {:<15} {:<15} {:<20}".format(
                    agent,
                    str(position),
                    f"{energy:.2f}",
                    str(array_position),
                    action_number,
                    str(action_array),
                    str(new_position),
                    str(resolved_position),
                )
            )

        print("-" * 120)
        print()
        print("Colliding Prey:", colliding_prey_agents)
        print()

    def _find_available_spawn_position(self, reference_position, occupied_positions):
        """
        Finds an available position for spawning a new agent.
        Tries to spawn near the parent agent first before selecting a random free position.
        """
        # Get all occupied positions
        # occupied_positions = set(self.agent_positions.values()) | set(self.grass_positions.values())

        x, y = reference_position  # Parent agent's position
        potential_positions = [
            (x + dx, y + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
            if 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size  # Stay in bounds
        ]

        # Filter for unoccupied positions
        valid_positions = [pos for pos in potential_positions if pos not in occupied_positions]

        if valid_positions:
            return valid_positions[0]  # Prefer adjacent position if available

        # Fallback: Find any random unoccupied position
        all_positions = {(i, j) for i in range(self.grid_size) for j in range(self.grid_size)}
        free_positions = list(all_positions - occupied_positions)

        if free_positions:
            return free_positions[np.random.randint(len(free_positions))]

        return None  # No available position found

    def get_state_snapshot(self):
        return {
            "current_step": self.current_step,
            "agent_positions": self.agent_positions.copy(),
            "agent_energies": self.agent_energies.copy(),
            "predator_positions": self.predator_positions.copy(),
            "prey_positions": self.prey_positions.copy(),
            "grass_positions": self.grass_positions.copy(),
            "grass_energies": self.grass_energies.copy(),
            "grid_world_state": self.grid_world_state.copy(),
            "agents": self.agents.copy(),
            "cumulative_rewards": self.cumulative_rewards.copy(),
            "current_num_predators": self.current_num_predators,
            "current_num_prey": self.current_num_prey,
            "agents_just_ate": self.agents_just_ate.copy(),
                # === 新增：草的物理对象快照 ===
            "grass_bodies": {k: (v.position.x, v.position.y) for k, v in self.grass_bodies.items()} if self.enable_continuous_space else {},
            "grass_shapes": list(self.grass_shapes.keys()) if self.enable_continuous_space else [],        
            # === 新增属性 ===
            "agent_population_id": self.agent_population_id.copy(),
            "agent_last_reproduction_step": self.agent_last_reproduction_step.copy(),
            "agent_generation": self.agent_generation.copy(),
            "agent_age": self.agent_age.copy(),
            "agent_algorithm": self.agent_algorithm.copy(),
            "agent_wants_to_mate": self.agent_wants_to_mate.copy(),
            "population_counts": self.population_counts.copy(),
            # === 新增：能量系统快照 ===
            "agent_steps_since_last_meal": self.agent_steps_since_last_meal.copy(),
            "agent_last_energy": self.agent_last_energy.copy(),
            "agent_recent_energy_delta": self.agent_recent_energy_delta.copy(),
            # === 新增：草的追踪 ===
            "grass_age": self.grass_age.copy() if hasattr(self, 'grass_age') else {},
            "grass_generation": self.grass_generation.copy() if hasattr(self, 'grass_generation') else {},
            "retired_agents": self.retired_agents.copy(),
            "next_free_idx": self.next_free_idx.copy(),
            "grass_home_positions": self.grass_home_positions.copy(),
            "inactive_grass": list(self.inactive_grass),
            "grass_respawn_timers": self.grass_respawn_timers.copy(),
        }

    def restore_state_snapshot(self, snapshot):
        self.current_step = snapshot["current_step"]
        self.agent_positions = snapshot["agent_positions"].copy()
        self.agent_energies = snapshot["agent_energies"].copy()
        self.predator_positions = snapshot["predator_positions"].copy()
        self.prey_positions = snapshot["prey_positions"].copy()
        self.grass_positions = snapshot["grass_positions"].copy()
        self.grass_energies = snapshot["grass_energies"].copy()
        self.grid_world_state = snapshot["grid_world_state"].copy()
        self.agents = snapshot["agents"].copy()
        self.cumulative_rewards = snapshot["cumulative_rewards"].copy()
        self.current_num_predators = snapshot["current_num_predators"]
        self.current_num_prey = snapshot["current_num_prey"]
        self.agents_just_ate = snapshot["agents_just_ate"].copy()
        self.retired_agents = set(snapshot.get("retired_agents", []))
        next_free_idx = snapshot.get("next_free_idx")
        if next_free_idx is not None:
            self.next_free_idx = dict(next_free_idx)
        else:
            self.next_free_idx = {
                "predator": self.n_initial_active_predator,
                "prey": self.n_initial_active_prey,
            }
        # === 新增：恢复能量系统 ===
        self.agent_steps_since_last_meal = snapshot.get("agent_steps_since_last_meal", {}).copy()
        self.agent_last_energy = snapshot.get("agent_last_energy", {}).copy()
        self.agent_recent_energy_delta = snapshot.get("agent_recent_energy_delta", {}).copy()
        self.grass_home_positions = snapshot.get("grass_home_positions", {}).copy()
        self.inactive_grass = set(snapshot.get("inactive_grass", []))
        self.grass_respawn_timers = snapshot.get("grass_respawn_timers", {}).copy()
        # === 恢复草的追踪 ===
        self.grass_age = snapshot.get("grass_age", {}).copy()
        self.grass_generation = snapshot.get("grass_generation", {}).copy()       
        # === 恢复新增属性 ===
        self.agent_population_id = snapshot["agent_population_id"].copy()
        self.agent_last_reproduction_step = snapshot["agent_last_reproduction_step"].copy()
        self.agent_generation = snapshot["agent_generation"].copy()
        self.agent_age = snapshot["agent_age"].copy()
        self.agent_algorithm = snapshot["agent_algorithm"].copy()
        self.agent_wants_to_mate = snapshot["agent_wants_to_mate"].copy()
        self.population_counts = snapshot["population_counts"].copy()
        # === 恢复草的物理对象 ===
        if self.enable_continuous_space and "grass_bodies" in snapshot:
            for grass_id, pos in snapshot["grass_bodies"].items():
                if grass_id in self.grass_bodies:
                    self.grass_bodies[grass_id].position = pos
    def _cast_sensor_ray(self, start_pos, angle, max_distance, exclude_agent_id=None):
        """发射传感器射线（增强版）"""
        if not self.enable_continuous_space or self.space is None:
            return {
                'hit': False,
                'distance': 1.0,
                'object_type': None,
                'object_id': None,
                'velocity': (0, 0),
                'energy_ratio': 0.0,
                'start_point': start_pos,
                'hit_point': None
            }
        
        # 计算射线终点
        end_x = start_pos[0] + np.cos(angle) * max_distance
        end_y = start_pos[1] + np.sin(angle) * max_distance
        end_pos = (end_x, end_y)
        
        # 获取要排除的shape
        exclude_shape = None
        if exclude_agent_id and exclude_agent_id in self.agent_shapes:
            exclude_shape = self.agent_shapes[exclude_agent_id]
        
        # 执行射线查询
        query_results = self.space.segment_query(start_pos, end_pos, 0, pymunk.ShapeFilter())
        
        if not query_results:
            return {
                'hit': False,
                'distance': 1.0,
                'object_type': None,
                'object_id': None,
                'velocity': (0, 0),
                'energy_ratio': 0.0,
                'start_point': start_pos,
                'hit_point': None
            }
        
        # 过滤掉自己
        valid_results = [r for r in query_results if r.shape != exclude_shape]
        
        if not valid_results:
            return {
                'hit': False,
                'distance': 1.0,
                'object_type': None,
                'object_id': None,
                'velocity': (0, 0),
                'energy_ratio': 0.0,
                'start_point': start_pos,
                'hit_point': None
            }
        
        # 找到最近的碰撞
        closest_result = min(valid_results, key=lambda r: r.alpha)
        
        hit_shape = closest_result.shape
        distance = closest_result.alpha
        hit_point = closest_result.point
        
        object_type = None
        object_id = None
        velocity = (0, 0)
        energy_ratio = 0.0
        
        # 检查碰撞类型
        if hit_shape.collision_type == self.COLLISION_TYPE_WALL:
            object_type = 'wall'
        
        elif hit_shape.collision_type == self.COLLISION_TYPE_GRASS:
            object_type = 'grass'
            object_id = getattr(hit_shape, 'grass_id', None)
            
            if (
                object_id
                and object_id in self.grass_energies
                and self.grass_max_energy > 0
            ):
                energy_ratio = self.grass_energies[object_id] / self.grass_max_energy
        
        elif hit_shape.collision_type == self.COLLISION_TYPE_PREDATOR:
            object_type = 'predator'
            object_id = getattr(hit_shape, 'agent_id', None)
            
            if object_id and object_id in self.agent_bodies:
                body = self.agent_bodies[object_id]
                velocity = (body.velocity.x, body.velocity.y)
                
                if object_id in self.agent_energies:
                    energy_ratio = self.agent_energies[object_id] / self.max_energy_predator
        
        elif hit_shape.collision_type == self.COLLISION_TYPE_PREY:
            object_type = 'prey'
            object_id = getattr(hit_shape, 'agent_id', None)
            
            if object_id and object_id in self.agent_bodies:
                body = self.agent_bodies[object_id]
                velocity = (body.velocity.x, body.velocity.y)
                
                if object_id in self.agent_energies:
                    energy_ratio = self.agent_energies[object_id] / self.max_energy_prey
        
        return {
            'hit': True,
            'distance': distance,
            'object_type': object_type,
            'object_id': object_id,
            'velocity': velocity,
            'energy_ratio': energy_ratio,
            'start_point': start_pos,
            'hit_point': hit_point
        }
    
    def _check_reproduction_eligibility(self, agent):
        """
        检查智能体是否满足繁殖条件。
        
        条件：
        1. 年龄在繁殖范围内
        2. 能量充足
        3. 冷却时间已过
        4. 种群未达上限
        """
        if not self.enable_paired_reproduction:
            return False
        # 检查年龄
        agent_type = "predator" if "predator" in agent else "prey"
        reproduction_cfg = self.reproduction_settings.get(agent_type, self.reproduction_settings["prey"])
        age = self.agent_age.get(agent, 0)
        if age < reproduction_cfg["min_age"] or age > reproduction_cfg["max_age"]:
            return False
        
        # 检查能量
        energy = self.agent_energies.get(agent, 0)
        min_energy = self.predator_creation_energy_threshold if agent_type == "predator" else self.prey_creation_energy_threshold
        
        if energy < min_energy:
            return False
        
        # 检查冷却时间
        last_reproduction = self.agent_last_reproduction_step.get(agent, -1000)
        if self.current_step - last_reproduction < reproduction_cfg["cooldown"]:
            return False
        
        # 检查种群数量
        pop_id = self.agent_population_id.get(agent, 0)
        pop_key = f"{agent_type}_{pop_id}"
        current_count = self.population_counts.get(pop_key, 0)
        
        if current_count >= reproduction_cfg["max_population_size"]:
            return False
        
        return True

    def _create_offspring(self, parent1, parent2):
        """
        创建后代智能体。
        
        Args:
            parent1, parent2: 父母智能体 ID
            
        Returns:
            offspring_id: 后代 ID，如果失败返回 None
        """
        agent_type = "predator" if "predator" in parent1 else "prey"
        
        reproduction_cfg = self.reproduction_settings.get(agent_type, self.reproduction_settings["prey"])

        if self.use_monotonic_offspring_ids:
            next_idx = self.next_free_idx.get(agent_type, 0)
            max_idx = self.max_indices[agent_type]

            if next_idx >= max_idx:
                self._log_warning(
                    "OffspringSlotsExhausted",
                    agent_type=agent_type,
                    next_index=next_idx,
                    max_index=max_idx,
                )
                return None

            offspring = f"{agent_type}_{next_idx}"
            self.next_free_idx[agent_type] = next_idx + 1
        else:
            # 查找可用的智能体 ID（过滤已退休的）
            potential_offspring = [
                agent
                for agent in self.possible_agents
                if agent.startswith(f"{agent_type}_")
                and agent not in self.agents
                and agent not in getattr(self, "retired_agents", set())
            ]

            if not potential_offspring:
                if self.verbose_spawning:
                    print(f"[REPRODUCE] No available {agent_type} slots")
                return None

            offspring = potential_offspring[0]

        if offspring in getattr(self, "retired_agents", set()):
            # 理论上不会发生（新 ID），但为安全起见直接拒绝
            self._log_warning(
                "OffspringReusePrevented",
                offspring=offspring,
                agent_type=agent_type,
            )
            return None
        if offspring in self.agents:
            self._log_warning(
                "OffspringIdAlreadyActive",
                offspring=offspring,
                agent_type=agent_type,
            )
            return None
        
        # === 根据繁殖模式计算能量成本 ===
        parent1_energy = self.agent_energies[parent1]
        parent2_energy = self.agent_energies[parent2]
        
        if reproduction_cfg["mode"] == "ratio":
            # === 模式1：父母各给固定比例（目标设计）===
            energy_ratio = reproduction_cfg["energy_ratio"]
            cost1 = parent1_energy * energy_ratio
            cost2 = parent2_energy * energy_ratio
            offspring_energy = cost1 + cost2
            
            # 检查是否有足够能量
            if parent1_energy < cost1 or parent2_energy < cost2:
                return None
            
            if self.verbose_spawning:
                print(
                    f"[REPRODUCE] Mode=ratio ({agent_type}), cost1={cost1:.1f}, "
                    f"cost2={cost2:.1f}, offspring={offspring_energy:.1f}"
                )
        
        else:  # "fixed_ratio" 模式
            # === 模式2：固定+比例（当前设计）===
            fixed_cost = reproduction_cfg["fixed_cost"]
            transfer_ratio = reproduction_cfg["transfer_ratio"]
            cost1 = fixed_cost + parent1_energy * transfer_ratio
            cost2 = fixed_cost + parent2_energy * transfer_ratio
            
            # 检查是否有足够能量
            if parent1_energy < cost1 or parent2_energy < cost2:
                return None
            
            # 后代初始能量（从父母转移 + 最低保障）
            offspring_energy = max(cost1 + cost2, reproduction_cfg["offspring_min_energy"])
            
            if self.verbose_spawning:
                print(
                    f"[REPRODUCE] Mode=fixed_ratio ({agent_type}), cost1={cost1:.1f}, "
                    f"cost2={cost2:.1f}, offspring={offspring_energy:.1f}"
                )
        
        # 扣除父母能量
        self.agent_energies[parent1] -= cost1
        self.agent_energies[parent2] -= cost2
        
        # === 后面代码保持不变 ===
        # 确定出生位置
        if self.enable_continuous_space:
            pos1 = self.agent_positions[parent1]
            pos2 = self.agent_positions[parent2]
            offspring_pos = (
                (pos1[0] + pos2[0]) / 2,
                (pos1[1] + pos2[1]) / 2
            )
            
            # 添加随机偏移，避免重叠
            offset_x = self.rng.uniform(-self.agent_radius, self.agent_radius)
            offset_y = self.rng.uniform(-self.agent_radius, self.agent_radius)
            offspring_pos = (
                np.clip(offspring_pos[0] + offset_x, self.agent_radius, self.world_width - self.agent_radius),
                np.clip(offspring_pos[1] + offset_y, self.agent_radius, self.world_height - self.agent_radius)
            )
        else:
            occupied_positions = set(self.agent_positions.values())
            offspring_pos = self._find_available_spawn_position(
                self.agent_positions[parent1],
                occupied_positions
            )
            
            if offspring_pos is None:
                # 恢复父母能量
                self.agent_energies[parent1] += cost1
                self.agent_energies[parent2] += cost2
                return None
        
        # 添加到环境
        self.agents.append(offspring)
        self.agent_positions[offspring] = offspring_pos
        self.agent_energies[offspring] = offspring_energy
        self.agent_last_energy[offspring] = offspring_energy
        self.agent_recent_energy_delta[offspring] = 0.0
        
        if agent_type == "predator":
            self.predator_positions[offspring] = offspring_pos
            self.current_num_predators += 1
        else:
            self.prey_positions[offspring] = offspring_pos
            self.current_num_prey += 1
        
        # 初始化属性
        pop_id = self.agent_population_id[parent1]
        self.agent_population_id[offspring] = pop_id
        self.agent_last_reproduction_step[offspring] = -1000
        self.agent_generation[offspring] = max(
            self.agent_generation[parent1],
            self.agent_generation[parent2]
        ) + 1
        self.agent_age[offspring] = 0
        self.agent_algorithm[offspring] = agent_type
        self.agent_wants_to_mate[offspring] = False
        self.agent_steps_since_last_meal[offspring] = 0
        
        # 更新种群计数
        pop_key = f"{agent_type}_{pop_id}"
        if pop_key in self.population_counts:
            self.population_counts[pop_key] += 1
        
        # 初始化奖励
        self.cumulative_rewards[offspring] = 0
        
        # 连续空间：创建物理 Body
        if self.enable_continuous_space and self.space is not None:
            mass = 1.0
            moment = pymunk.moment_for_circle(mass, 0, self.agent_radius)
            body = pymunk.Body(mass, moment)
            body.position = offspring_pos
            
            shape = pymunk.Circle(body, self.agent_radius)
            shape.collision_type = (self.COLLISION_TYPE_PREDATOR if agent_type == "predator" 
                                else self.COLLISION_TYPE_PREY)
            shape.friction = 0.5
            shape.elasticity = 0.8
            shape.agent_id = offspring
            
            self.agent_bodies[offspring] = body
            self.agent_shapes[offspring] = shape
            
            self.space.add(body, shape)
        
        if self.verbose_spawning:
            print(f"[OFFSPRING] {offspring} born from {parent1} + {parent2}, "
                f"E={offspring_energy:.1f}, gen={self.agent_generation[offspring]}, "
                f"pop={pop_id}")
        
        return offspring
    def debug_ray_cast(self, agent_id):
        """调试射线检测"""
        if agent_id not in self.agent_positions:
            print(f"[DEBUG] Agent {agent_id} not found!")
            return
        
        agent_pos = self.agent_positions[agent_id]
        print(f"\n[DEBUG] Testing ray cast from {agent_id} at {agent_pos}")
        
        # 测试一条射线（向右）
        angle = 0  # 0度，向右
        max_distance = self.sensor_range
        
        end_x = agent_pos[0] + np.cos(angle) * max_distance
        end_y = agent_pos[1] + np.sin(angle) * max_distance
        end_pos = (end_x, end_y)
        
        print(f"[DEBUG] Ray from {agent_pos} to {end_pos}")
        print(f"[DEBUG] Space has {len(list(self.space.shapes))} shapes")
        
        # 列出所有物理对象
        for shape in self.space.shapes:
            print(f"  - {shape.collision_type}: sensor={shape.sensor}, body={shape.body.position}")
        
        # 测试射线查询（默认）
        result1 = self.space.segment_query_first(agent_pos, end_pos, 0, pymunk.ShapeFilter())
        print(f"[DEBUG] Default query result: {result1}")
        
        # 测试射线查询（包括sensor）
        result2 = self.space.segment_query(agent_pos, end_pos, 0, pymunk.ShapeFilter())
        print(f"[DEBUG] Full query result: {len(result2)} hits")
        for hit in result2:
            print(f"  - Hit: {hit.shape.collision_type}, distance={hit.alpha * max_distance:.2f}")

    def _estimate_grass_density(self, grass_id: str) -> float:
        """估算指定草周围的密度（0-1）。"""
        neighbor_count = self._count_local_neighbors(grass_id)
        if self.grass_density_reference <= 0:
            return 0.0
        return float(np.clip(neighbor_count / self.grass_density_reference, 0.0, 1.0))

    def _sample_grass_energy(self) -> float:
        """采样草的初始能量，围绕设定目标值做小幅波动。"""
        target = float(np.clip(self.grass_offspring_energy, self.grass_min_energy, self.grass_max_energy))
        span = max(self.grass_max_energy * 0.2, self.grass_min_energy)
        low = max(self.grass_min_energy, target - span)
        high = min(self.grass_max_energy, target + span)
        if high <= low:
            return float(low)
        return float(self.rng.uniform(low, high))

    def _create_grass_offspring(self, parent_grass_id):
        """创建草的后代（尝试多次找位置）"""
        # 查找可用的草 ID
        used_ids = set(self.grass_positions.keys())
        potential_grass_ids = [
            f"grass_{i}" for i in range(9999)  # 支持大量草
            if f"grass_{i}" not in used_ids
        ]
        
        if not potential_grass_ids:
            return None
        
        offspring_id = potential_grass_ids[0]
        
        # 确定后代位置（多次尝试）
        parent_pos = self.grass_positions[parent_grass_id]
        
        if self.enable_continuous_space:
            attempts = 0
            offspring_pos = None
            grass_radius = self.agent_radius * 0.6
            min_spacing = max(self.agent_radius * 2, grass_radius * 2)
            max_distance = max(min_spacing, self.grass_perception_radius)
            
            while attempts < self.grass_spawn_max_attempts:
                # 随机角度和距离
                angle = self.rng.uniform(0, 2 * np.pi)
                distance = self.rng.uniform(min_spacing, max_distance)
                
                # 计算候选位置
                x = parent_pos[0] + np.cos(angle) * distance
                y = parent_pos[1] + np.sin(angle) * distance
                
                # 检查边界
                if not (grass_radius < x < self.world_width - grass_radius and
                        grass_radius < y < self.world_height - grass_radius):
                    attempts += 1
                    continue
                
                # 检查与其他草的距离
                too_close = False
                for existing_pos in self.grass_positions.values():
                    dist = np.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2)
                    if dist < min_spacing:
                        too_close = True
                        break
                
                if not too_close:
                    offspring_pos = (float(x), float(y))
                    break
                
                attempts += 1
            
            if offspring_pos is None:
                if self.verbose_spawning:
                    print(f"[GRASS] {parent_grass_id} 繁殖失败（找不到位置，尝试{attempts}次）")
                return None
        
        else:
            # 离散空间
            occupied_positions = set(self.grass_positions.values()) | set(self.agent_positions.values())
            offspring_pos = self._find_available_spawn_position(parent_pos, occupied_positions)
            
            if offspring_pos is None:
                return None
        
        # 添加新草
        self.grass_positions[offspring_id] = offspring_pos
        offspring_energy = self._sample_grass_energy()
        self.grass_energies[offspring_id] = offspring_energy
        self.grass_age[offspring_id] = 0
        self.grass_generation[offspring_id] = self.grass_generation.get(parent_grass_id, 0) + 1
        if offspring_id not in self.grass_agents:
            self.grass_agents.append(offspring_id)
        self.current_num_grass += 1
        self._mark_grass_index_dirty()
        
        # 连续空间：创建物理对象
        if self.enable_continuous_space and self.space is not None:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = offspring_pos
            
            grass_radius = self.agent_radius * 0.6
            shape = pymunk.Circle(body, grass_radius)
            shape.collision_type = self.COLLISION_TYPE_GRASS
            shape.sensor = True
            shape.grass_id = offspring_id
            
            self.grass_bodies[offspring_id] = body
            self.grass_shapes[offspring_id] = shape
            
            self.space.add(body, shape)
        
        # 离散空间：更新网格
        if not self.enable_continuous_space:
            self.grid_world_state[3, *offspring_pos] = offspring_energy
        
        return offspring_id

    def _remove_grass(self, grass_id: str) -> None:
        """彻底移除一株草及其物理表现。"""
        if self.fixed_grass_mode:
            self._deactivate_grass_patch(grass_id)
            return
        position = self.grass_positions.pop(grass_id, None)
        self.grass_energies.pop(grass_id, None)
        if grass_id in self.grass_age:
            self.grass_age[grass_id] = 0

        if self.enable_continuous_space and self.space is not None:
            body = self.grass_bodies.pop(grass_id, None)
            shape = self.grass_shapes.pop(grass_id, None)
            to_remove = []
            if shape is not None and shape in self.space.shapes:
                to_remove.append(shape)
            if body is not None and body in self.space.bodies:
                to_remove.append(body)
            if to_remove:
                try:
                    self.space.remove(*to_remove)
                except Exception as exc:
                    if self.verbose_engagement:
                        print(f"[WARNING] Failed to remove grass body {grass_id}: {exc}")
        else:
            if position is not None:
                self.grid_world_state[3, *position] = 0.0

        if self.current_num_grass > 0:
            self.current_num_grass -= 1
        self._mark_grass_index_dirty()
    @staticmethod
    def _distribute_agents(total, n_populations):
        """
        将agent数量平均分配到各个种群，余数分配给前面的种群。
        
        Args:
            total: 总agent数量
            n_populations: 种群数量
        
        Returns:
            list: 每个种群的agent数量，如 [4, 4, 3]
        
        Examples:
            >>> _distribute_agents(9, 3)
            [3, 3, 3]
            >>> _distribute_agents(11, 3)
            [4, 4, 3]
            >>> _distribute_agents(10, 3)
            [4, 3, 3]
        """
        base = total // n_populations
        remainder = total % n_populations
        
        distribution = [base] * n_populations
        for i in range(remainder):
            distribution[i] += 1
        
        return distribution
    
    def get_agent_policy_mapping(self):
        """
        返回建议的policy映射（供RLlib使用）
        
        Returns:
            dict: {agent_id: policy_name}
            
        Example:
            {
                "predator_0": "predator_pop0",
                "predator_1": "predator_pop0", 
                "predator_2": "predator_pop1",
                ...
            }
        """
        mapping = {}
        for agent in self.agents:
            agent_type = "predator" if "predator" in agent else "prey"
            pop_id = self.agent_population_id.get(agent, 0)
            mapping[agent] = f"{agent_type}_pop{pop_id}"
        return mapping

    def get_population_distribution(self):
        """
        获取当前种群分布统计
        
        Returns:
            dict: 每个种群的详细信息
        """
        stats = {}
        
        for agent_type in ["predator", "prey"]:
            for pop_id in range(self.n_populations):
                pop_key = f"{agent_type}_{pop_id}"
                
                agents_in_pop = [
                    a for a in self.agents 
                    if agent_type in a and self.agent_population_id.get(a) == pop_id
                ]
                
                stats[pop_key] = {
                    "count": len(agents_in_pop),
                    "algorithm": self.population_display_info.get(pop_key, "Unknown"),
                    "agents": agents_in_pop,
                    "avg_energy": np.mean([self.agent_energies[a] for a in agents_in_pop]) if agents_in_pop else 0,
                    "avg_age": np.mean([self.agent_age[a] for a in agents_in_pop]) if agents_in_pop else 0,
                }
        
        return stats
    def _get_environment_layer(self, ray_results):
        """环境层：障碍物、草距离、草能量（90维）"""
        obstacle_distances = np.ones(self.n_sensors, dtype=np.float32)
        grass_distances = np.ones(self.n_sensors, dtype=np.float32)
        grass_energies = np.zeros(self.n_sensors, dtype=np.float32)
        
        for i, result in enumerate(ray_results):
            if not result['hit']:
                continue
            
            distance = result['distance']
            object_type = result['object_type']
            
            # 障碍物（墙+所有实体）
            obstacle_distances[i] = distance
            
            # 草信息
            if object_type == 'grass':
                grass_distances[i] = distance
                grass_id = result.get('object_id')
                if (
                    grass_id
                    and grass_id in self.grass_energies
                    and self.grass_max_energy > 0
                ):
                    grass_energies[i] = self.grass_energies[grass_id] / self.grass_max_energy
        
        return np.concatenate([obstacle_distances, grass_distances, grass_energies])


    def _get_predator_layer(self, ray_results, self_id, self_type):
        """Predator层：距离、速度、能量（90维）"""
        distances = np.ones(self.n_sensors, dtype=np.float32)
        velocities = np.zeros(self.n_sensors, dtype=np.float32)
        energies = np.zeros(self.n_sensors, dtype=np.float32)
        
        for i, result in enumerate(ray_results):
            if not result['hit'] or result['object_type'] != 'predator':
                continue
            
            distances[i] = result['distance']
            
            # 相对速度投影
            velocity = result.get('velocity', (0, 0))
            if velocity != (0, 0):
                my_velocity = self._get_agent_velocity(self_id)
                rel_vx = velocity[0] - my_velocity[0]
                rel_vy = velocity[1] - my_velocity[1]
                
                # 计算射线方向
                angle = np.arctan2(
                    result['hit_point'][1] - result['start_point'][1],
                    result['hit_point'][0] - result['start_point'][0]
                ) if 'hit_point' in result else 0
                
                ray_dir_x = np.cos(angle)
                ray_dir_y = np.sin(angle)
                projection = (rel_vx * ray_dir_x + rel_vy * ray_dir_y) / 100.0
                velocities[i] = np.clip(projection, -1.0, 1.0)
            
            # 能量估计
            object_id = result.get('object_id')
            if object_id and object_id in self.agent_energies:
                max_energy = self.max_energy_predator
                energies[i] = self.agent_energies[object_id] / max_energy
        
        return np.concatenate([distances, velocities, energies])


    def _get_prey_layer(self, ray_results, self_id, self_type):
        """Prey层：距离、速度、能量（90维）"""
        distances = np.ones(self.n_sensors, dtype=np.float32)
        velocities = np.zeros(self.n_sensors, dtype=np.float32)
        energies = np.zeros(self.n_sensors, dtype=np.float32)
        
        for i, result in enumerate(ray_results):
            if not result['hit'] or result['object_type'] != 'prey':
                continue
            
            distances[i] = result['distance']
            
            # 相对速度投影
            velocity = result.get('velocity', (0, 0))
            if velocity != (0, 0):
                my_velocity = self._get_agent_velocity(self_id)
                rel_vx = velocity[0] - my_velocity[0]
                rel_vy = velocity[1] - my_velocity[1]
                
                angle = np.arctan2(
                    result['hit_point'][1] - result['start_point'][1],
                    result['hit_point'][0] - result['start_point'][0]
                ) if 'hit_point' in result else 0
                
                ray_dir_x = np.cos(angle)
                ray_dir_y = np.sin(angle)
                projection = (rel_vx * ray_dir_x + rel_vy * ray_dir_y) / 100.0
                velocities[i] = np.clip(projection, -1.0, 1.0)
            
            # 能量估计
            object_id = result.get('object_id')
            if object_id and object_id in self.agent_energies:
                max_energy = self.max_energy_prey
                energies[i] = self.agent_energies[object_id] / max_energy
        
        return np.concatenate([distances, velocities, energies])


    def _get_mate_layer(self, ray_results, self_type, self_pop_id, self_id=None):
        """同类同群层：距离、速度、可交配性（90维）"""
        distances = np.ones(self.n_sensors, dtype=np.float32)
        velocities = np.zeros(self.n_sensors, dtype=np.float32)
        fertility = np.zeros(self.n_sensors, dtype=np.float32)
        
        for i, result in enumerate(ray_results):
            if not result['hit']:
                continue
            
            object_type = result['object_type']
            object_id = result.get('object_id')
            
            # 必须是同类
            if (self_type == 'predator' and object_type != 'predator') or \
            (self_type == 'prey' and object_type != 'prey'):
                continue
            
            # 必须是同种群
            if not object_id or object_id not in self.agent_population_id:
                continue
            
            target_pop_id = self.agent_population_id[object_id]
            if target_pop_id != self_pop_id:
                continue
            
            # 是潜在配偶
            distances[i] = result['distance']
            
            # 速度
            velocity = result.get('velocity', (0, 0))
            if velocity != (0, 0):
                ref_agent = self_id or f"{self_type}_0"
                my_velocity = self._get_agent_velocity(ref_agent)
                rel_vx = velocity[0] - my_velocity[0]
                rel_vy = velocity[1] - my_velocity[1]
                
                angle = np.arctan2(
                    result['hit_point'][1] - result['start_point'][1],
                    result['hit_point'][0] - result['start_point'][0]
                ) if 'hit_point' in result else 0
                
                ray_dir_x = np.cos(angle)
                ray_dir_y = np.sin(angle)
                projection = (rel_vx * ray_dir_x + rel_vy * ray_dir_y) / 100.0
                velocities[i] = np.clip(projection, -1.0, 1.0)
            
            # 可交配性
            if object_id in self.agent_wants_to_mate:
                fertility[i] = 1.0 if self.agent_wants_to_mate[object_id] else 0.0
        
        return np.concatenate([distances, velocities, fertility])


    def _get_self_state(self, agent, agent_type):
        """自身状态（self_state_dim维）"""
        state = np.zeros(self.self_state_dim, dtype=np.float32)
        
        # [0] 类型
        state[0] = 1.0 if agent_type == "predator" else 0.0
        
        # [1] 种群ID
        pop_id = self.agent_population_id.get(agent, 0)
        if self.n_populations > 1:
            state[1] = pop_id / (self.n_populations - 1)
        else:
            state[1] = 0.0
        
        # [2] 能量比例
        energy = self.agent_energies.get(agent, 0)
        max_energy = self.max_energy_predator if agent_type == "predator" else self.max_energy_prey
        state[2] = energy / max_energy
        
        # [3] 饥饿比例
        if self.enable_hunger and agent in self.agent_steps_since_last_meal:
            steps_hungry = self.agent_steps_since_last_meal[agent]
            max_steps = (self.max_steps_without_food_predator if agent_type == "predator" 
                        else self.max_steps_without_food_prey)
            state[3] = steps_hungry / max_steps
        else:
            state[3] = 0.0
        
        # [4] 刚吃标志
        state[4] = 1.0 if agent in self.agents_just_ate else 0.0
        
        # [5] 可繁殖标志
        if self.enable_paired_reproduction:
            state[5] = 1.0 if self.agent_wants_to_mate.get(agent, False) else 0.0
        else:
            state[5] = 0.0
        
        # [6-7] 速度（大小+角度）
        if agent in self.agent_bodies:
            body = self.agent_bodies[agent]
            velocity = body.velocity
            speed = velocity.length
            state[6] = speed / 200.0  # 归一化（假设最大速度200）
            
            if speed > 0.1:
                angle = np.arctan2(velocity.y, velocity.x)
                state[7] = angle / np.pi  # 归一化到[-1, 1]
            else:
                state[7] = 0.0
        else:
            state[6] = 0.0
            state[7] = 0.0
        
        # [8] 年龄比例
        age = self.agent_age.get(agent, 0)
        reproduction_cfg = self.reproduction_settings.get(agent_type, self.reproduction_settings["prey"])
        min_age = max(reproduction_cfg["min_age"], 1)
        state[8] = min(age / min_age, 1.0)
        
        # [9] 冷却比例
        if self.enable_paired_reproduction:
            last_repro = self.agent_last_reproduction_step.get(agent, -1000)
            cooldown_target = max(reproduction_cfg["cooldown"], 1)
            cooldown_remaining = max(
                0, reproduction_cfg["cooldown"] - (self.current_step - last_repro)
            )
            state[9] = cooldown_remaining / cooldown_target
        else:
            state[9] = 0.0

        # [10] 能量变化比例
        energy_delta = self.agent_recent_energy_delta.get(agent, 0.0)
        state[10] = np.clip(energy_delta / max_energy, -1.0, 1.0) if max_energy > 0 else 0.0

        # [11] 附近草密度
        if agent in self.agent_positions:
            _, grass_density = self._get_local_grass_density(self.agent_positions[agent])
            state[11] = grass_density
        else:
            state[11] = 0.0

        return state


    def _get_agent_velocity(self, agent_id):
        """获取指定智能体的速度向量（连续空间）"""
        if not self.enable_continuous_space:
            return (0.0, 0.0)
        if agent_id in self.agent_bodies:
            body = self.agent_bodies[agent_id]
            return (float(body.velocity.x), float(body.velocity.y))
        return (0.0, 0.0)
# predpreygrass_rllib_env.py - 添加辅助函数（在类中添加，约1900行附近）

    def _mark_grass_index_dirty(self) -> None:
        """Mark the grass spatial index as needing rebuild."""
        self._grass_spatial_index_dirty = True

    def _ensure_grass_spatial_index(self) -> None:
        """Build a simple spatial hash for grass positions if needed."""
        if not hasattr(self, "_grass_spatial_index"):
            self._grass_spatial_index = {}
        if not hasattr(self, "_grass_cell_size") or self._grass_cell_size <= 0:
            self._grass_cell_size = max(1.0, float(self.grass_perception_radius) / 2.0)
        if not hasattr(self, "_grass_spatial_index_dirty"):
            self._grass_spatial_index_dirty = True

        if not self._grass_spatial_index_dirty:
            return

        cell_size = max(1.0, self._grass_cell_size)
        new_index: Dict[Tuple[int, int], List[str]] = {}
        for grass_id, position in self.grass_positions.items():
            x, y = position
            cell_x = int(float(x) // cell_size)
            cell_y = int(float(y) // cell_size)
            new_index.setdefault((cell_x, cell_y), []).append(grass_id)

        self._grass_spatial_index = new_index
        self._grass_spatial_index_dirty = False

    def _count_local_neighbors(self, grass_id):
        """统计草周围的邻居数量"""
        if grass_id not in self.grass_positions:
            return 0

        self._ensure_grass_spatial_index()

        grass_pos = self.grass_positions[grass_id]
        radius_sq = float(self.grass_perception_radius) ** 2
        cell_x = int(grass_pos[0] // self._grass_cell_size)
        cell_y = int(grass_pos[1] // self._grass_cell_size)

        neighbor_count = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell_key = (cell_x + dx, cell_y + dy)
                for other_id in self._grass_spatial_index.get(cell_key, ()):
                    if other_id == grass_id:
                        continue
                    other_pos = self.grass_positions.get(other_id)
                    if other_pos is None:
                        continue
                    diff_x = grass_pos[0] - other_pos[0]
                    diff_y = grass_pos[1] - other_pos[1]
                    if diff_x * diff_x + diff_y * diff_y <= radius_sq:
                        neighbor_count += 1

        return neighbor_count

    def _get_initial_grass_energy(self) -> float:
        if self.fixed_grass_mode:
            return float(self.grass_fixed_initial_energy)
        return self._sample_grass_energy()

    def _activate_grass_patch(self, grass_id: str, position: Tuple[float, float], energy: float, reset_generation: bool = False) -> None:
        self.grass_positions[grass_id] = position
        self.grass_energies[grass_id] = float(np.clip(energy, 0.0, self.grass_max_energy))
        self.grass_age[grass_id] = 0
        if reset_generation or grass_id not in self.grass_generation:
            self.grass_generation[grass_id] = 0
        else:
            self.grass_generation[grass_id] = self.grass_generation.get(grass_id, 0) + 1

        if self.enable_continuous_space and self.space is not None:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = position
            grass_radius = self.agent_radius * 0.6
            shape = pymunk.Circle(body, grass_radius)
            shape.collision_type = self.COLLISION_TYPE_GRASS
            shape.sensor = True
            shape.grass_id = grass_id

            self.grass_bodies[grass_id] = body
            self.grass_shapes[grass_id] = shape
            self.space.add(body, shape)
        else:
            grid_pos = tuple(int(coord) for coord in position)
            self.grid_world_state[3, *grid_pos] = self.grass_energies[grass_id]

        self._mark_grass_index_dirty()
        self.current_num_grass = min(self.initial_num_grass, self.current_num_grass + 1)
        self.inactive_grass.discard(grass_id)

    def _deactivate_grass_patch(self, grass_id: str) -> None:
        if grass_id in self.inactive_grass:
            return
        position = self.grass_positions.pop(grass_id, None)
        self.grass_energies.pop(grass_id, None)
        self.grass_age.pop(grass_id, None)
        self.grass_generation.pop(grass_id, None)

        if self.enable_continuous_space and self.space is not None:
            body = self.grass_bodies.pop(grass_id, None)
            shape = self.grass_shapes.pop(grass_id, None)
            to_remove = []
            if shape is not None:
                to_remove.append(shape)
            if body is not None:
                to_remove.append(body)
            if to_remove:
                try:
                    self.space.remove(*to_remove)
                except Exception:
                    pass
        else:
            if position is not None:
                grid_pos = tuple(int(coord) for coord in position)
                self.grid_world_state[3, *grid_pos] = 0.0

        self._mark_grass_index_dirty()
        self.current_num_grass = max(0, self.current_num_grass - 1)
        self.inactive_grass.add(grass_id)
        self.grass_respawn_timers[grass_id] = self.grass_fixed_respawn_delay

    def _update_fixed_grass(self) -> None:
        growth = self.grass_fixed_growth_rate
        for grass in list(self.grass_positions.keys()):
            if grass not in self.grass_energies:
                continue
            updated_energy = min(self.grass_max_energy, self.grass_energies[grass] + growth)
            self.grass_energies[grass] = updated_energy
            if not self.enable_continuous_space and grass in self.grass_positions:
                self.grid_world_state[3, *self.grass_positions[grass]] = updated_energy
            if grass in self.grass_age:
                self.grass_age[grass] += 1

        expired = []
        for grass in list(self.inactive_grass):
            timer = self.grass_respawn_timers.get(grass, 0) - 1
            if timer <= 0:
                expired.append(grass)
            else:
                self.grass_respawn_timers[grass] = timer

        for grass in expired:
            self.grass_respawn_timers.pop(grass, None)
            position = self.grass_home_positions.get(grass)
            if position is None:
                continue
            self._activate_grass_patch(grass, position, self.grass_fixed_initial_energy, reset_generation=False)

    def _get_local_grass_density(self, position: Tuple[float, float], radius: float | None = None) -> Tuple[float, float]:
        """返回指定位置附近的草密度和最近草距离比例"""
        if not self.grass_positions:
            return 1.0, 0.0

        self._ensure_grass_spatial_index()
        search_radius = float(radius or self.grass_perception_radius)
        if search_radius <= 0:
            return 1.0, 0.0

        radius_sq = search_radius**2
        cell_size = max(1.0, getattr(self, "_grass_cell_size", search_radius))
        cell_x = int(position[0] // cell_size)
        cell_y = int(position[1] // cell_size)

        neighbor_ids: List[str] = []
        min_dist_sq = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell_key = (cell_x + dx, cell_y + dy)
                for grass_id in self._grass_spatial_index.get(cell_key, ()):
                    grass_pos = self.grass_positions.get(grass_id)
                    if grass_pos is None:
                        continue
                    dist_sq = (grass_pos[0] - position[0]) ** 2 + (grass_pos[1] - position[1]) ** 2
                    if dist_sq <= radius_sq:
                        neighbor_ids.append(grass_id)
                        if min_dist_sq is None or dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq

        min_distance_ratio = 1.0
        if min_dist_sq is not None and search_radius > 0:
            min_distance_ratio = min((min_dist_sq ** 0.5) / search_radius, 1.0)

        density_ratio = 0.0
        if neighbor_ids:
            reference = max(1.0, float(self.grass_density_reference))
            density_ratio = float(min(len(neighbor_ids) / reference, 1.0))

        return min_distance_ratio, density_ratio
    
