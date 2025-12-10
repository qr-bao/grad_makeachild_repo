"""
Prey策略测试 - 数学策略 vs Rampage 对照
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.rampage_agent import (
    RampageAgent,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.predpreygrass_rllib_env import (
    PredPreyGrass,
)
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.visualizer import (
    PredPreyVisualizer,
)
# from predpreygrass.rllib.env3.predpreygrass_rllib_env129.prey_test_config import (
#     prey_test_config,
# )
from predpreygrass.rllib.env3.predpreygrass_rllib_env129.config.config_env_base import (
    config_env_base,
)
# from predpreygrass.rllib.env3.predpreygrass_rllib_env129.config.config_env_base_nopredator import (
#     config_env_base,
# )
prey_test_config = config_env_base


def load_env_config(path: Path) -> Dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Env config file not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    # For Python config files, load as a module to honor relative imports.
    import importlib
    import importlib.util
    import sys

    # 1) Try absolute import via package path (works for in-repo configs with relative imports).
    try:
        module_name = "predpreygrass.rllib.env3.predpreygrass_rllib_env129.config." + path.stem
        module = importlib.import_module(module_name)
        return getattr(module, "config_env", None) or getattr(module, "config_env_base", None) or getattr(module, "config", {})
    except Exception:
        # 2) Fallback: load by file path without package context.
        spec = importlib.util.spec_from_file_location("env_cfg", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load config module from {path}")
        module = importlib.util.module_from_spec(spec)
        # Allow relative imports to find sibling files.
        sys.path.insert(0, str(path.parent))
        try:
            spec.loader.exec_module(module)  # type: ignore[call-arg]
        finally:
            if sys.path and sys.path[0] == str(path.parent):
                sys.path.pop(0)
        return getattr(module, "config_env", None) or getattr(module, "config_env_base", None) or getattr(module, "config", {})


class SensorPolicyBase:
    """帮助基于传感器观测做决策的基础类。"""

    def __init__(self, env: PredPreyGrass):
        self.env = env
        self.rng = np.random.default_rng()
        self._agent_heading_bias: Dict[str, float] = {}
        self.debug_info: Dict[str, Dict[str, object]] = {}
        self.n_sensors = int(getattr(env, "n_sensors", 20))
        self.self_state_dim = int(getattr(env, "self_state_dim", 12))
        self.layer_span = self.n_sensors * 3
        self.total_dim = self.layer_span * 4 + self.self_state_dim
        self.blank_observation = np.zeros(self.total_dim, dtype=np.float32)
        angles = np.linspace(0.0, 2.0 * np.pi, self.n_sensors, endpoint=False, dtype=np.float32)
        self._base_sensor_dirs = np.stack((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)
        self.sensor_dirs = self._base_sensor_dirs

    def _split_layers(self, observation: Optional[np.ndarray]):
        if observation is None:
            obs = self.blank_observation
        else:
            obs = np.asarray(observation, dtype=np.float32)
            if obs.shape[0] < self.total_dim:
                padded = self.blank_observation.copy()
                padded[: obs.shape[0]] = obs
                obs = padded
        layers = {}
        offset = 0
        for name in ("env", "predator", "prey", "mate"):
            raw = obs[offset : offset + self.layer_span]
            layer = raw.reshape(3, self.n_sensors).T
            layers[name] = layer
            offset += self.layer_span
        self_state = obs[offset : offset + self.self_state_dim]
        return layers, self_state

    def _prepare_layers(self, agent_id: str, observation: Optional[np.ndarray]):
        layers, self_state = self._split_layers(observation)
        dirs = self._base_sensor_dirs
        if self.n_sensors > 0:
            heading_angle = float(self_state[7]) * np.pi
            speed = float(self_state[6])
            if speed <= 0.02:
                heading_angle = self._agent_heading_bias.setdefault(
                    agent_id, self.rng.uniform(-np.pi, np.pi)
                )
            norm_angle = (heading_angle + 2 * np.pi) % (2 * np.pi)
            offset = int(round(norm_angle / (2 * np.pi) * self.n_sensors)) % self.n_sensors
            if offset:
                for key in layers:
                    layers[key] = np.roll(layers[key], -offset, axis=0)
                dirs = np.roll(self._base_sensor_dirs, -offset, axis=0)
            else:
                dirs = self._base_sensor_dirs
        return layers, self_state, dirs

    def _wall_repulsion(self, env_layer: np.ndarray, dirs: np.ndarray, threshold: float = 0.03):
        if env_layer.size == 0:
            return None
        obstacle_distances = env_layer[: self.n_sensors, 0]
        close_idx = np.where(obstacle_distances < threshold)[0]
        if close_idx.size == 0:
            return None
        weights = threshold - obstacle_distances[close_idx]
        push = np.sum(dirs[close_idx] * weights[:, None], axis=0)
        norm = float(np.linalg.norm(push))
        if norm < 1e-6:
            return None
        direction = (push / norm).astype(np.float32)
        urgency = float(np.clip(np.max(weights) / threshold, 0.0, 1.0))
        action = np.clip(direction * (0.1 + 0.05 * urgency), -0.5, 0.5).astype(np.float32)
        return {"action": action, "urgency": urgency}


class PredatorChasePolicy(SensorPolicyBase):
    """使用传感器观测选择最近猎物方向的简易策略。"""

    def __init__(
        self,
        env: PredPreyGrass,
        max_gain: float = 0.5,
        close_gain: float = 0.2,
        visibility_threshold: float = 0.99,
    ):
        super().__init__(env)
        self.max_gain = max_gain
        self.close_gain = close_gain
        self.visibility_threshold = visibility_threshold
        self.close_distance_norm = 0.12
        self.wanderer = RampageAgent(env)

    def _pick_target_index(self, distances: np.ndarray, energies: np.ndarray) -> Optional[int]:
        visible = np.where(distances < self.visibility_threshold)[0]
        if visible.size == 0:
            return None
        scores = distances[visible].copy()
        if energies is not None:
            scores -= 0.05 * energies[visible]
        best = scores.min()
        tol = max(0.02, scores.std() * 0.5)
        candidates = visible[scores <= best + tol]
        choice = int(self.rng.choice(candidates))
        return choice

    def get_action(self, agent_id: str, observation: Optional[np.ndarray]) -> np.ndarray:
        layers, _, dirs = self._prepare_layers(agent_id, observation)
        wall_push = self._wall_repulsion(layers["env"], dirs)
        prey_layer = layers["prey"]
        best_idx = self._pick_target_index(prey_layer[:, 0], prey_layer[:, 2])
        if best_idx is None:
            # 略微随机游走，否则所有捕食者会因为同一 fallback 行为扎堆
            noise = self.rng.uniform(-0.2, 0.2, size=2).astype(np.float32)
            action = np.clip(self.wanderer.get_action(agent_id) + noise, -0.5, 0.5)
            self.debug_info[agent_id] = {"mode": "wander", "wall": wall_push is not None}
            return action

        distance_norm = float(np.clip(prey_layer[best_idx, 0], 0.0, 1.0))
        direction = dirs[best_idx]
        gain = self.close_gain if distance_norm < self.close_distance_norm else self.max_gain
        action = np.clip(direction * gain, -0.5, 0.5).astype(np.float32)
        if wall_push and wall_push["urgency"] >= 0.6:
            weight = min(0.4, 0.4 * wall_push["urgency"])
            action = np.clip(action * (1.0 - weight) + wall_push["action"] * weight, -0.5, 0.5)
        self.debug_info[agent_id] = {"mode": "attack", "index": best_idx, "wall": wall_push is not None}
        return action


class PreyTestMonitor:
    """统计并打印测试指标。"""

    def __init__(self, env: PredPreyGrass):
        self.env = env
        self.history: Dict[str, list] = {
            "step": [],
            "prey_0": [],
            "prey_1": [],
            "total_grass": [],
            "prey_0_avg_energy": [],
            "prey_1_avg_energy": [],
        }

    def update(self, step: int) -> None:
        pop_stats = self.env.get_population_distribution()
        prey_0_stats = pop_stats.get("prey_0", {})
        prey_1_stats = pop_stats.get("prey_1", {})

        self.history["step"].append(step)
        self.history["prey_0"].append(prey_0_stats.get("count", 0))
        self.history["prey_1"].append(prey_1_stats.get("count", 0))
        self.history["total_grass"].append(len(self.env.grass_positions))
        self.history["prey_0_avg_energy"].append(prey_0_stats.get("avg_energy", 0.0))
        self.history["prey_1_avg_energy"].append(prey_1_stats.get("avg_energy", 0.0))

    def print_status(self, step: int) -> None:
        pop_stats = self.env.get_population_distribution()
        predator_count = sum(1 for agent in self.env.agents if "predator" in agent)
        prey_0_stats = pop_stats.get("prey_0", {})
        prey_1_stats = pop_stats.get("prey_1", {})

        print(f"\n{'=' * 70}")
        print(f"📊 PREY TEST - Step {step}")
        print(f"{'=' * 70}")
        print(f"🛡️  Predators: {predator_count}")
        print(
            "🟢 Prey Pop 0 (Math):  "
            f"Count={prey_0_stats.get('count', 0):2d} | "
            f"Avg E={prey_0_stats.get('avg_energy', 0.0):5.1f} | "
            f"Avg Age={prey_0_stats.get('avg_age', 0.0):4.0f}"
        )
        print(
            "🔵 Prey Pop 1 (Rampage): "
            f"Count={prey_1_stats.get('count', 0):2d} | "
            f"Avg E={prey_1_stats.get('avg_energy', 0.0):5.1f} | "
            f"Avg Age={prey_1_stats.get('avg_age', 0.0):4.0f}"
        )
        print(f"🌱 Grass: {len(self.env.grass_positions)}")
        print(f"{'=' * 70}\n")

    def summarise(self) -> str:
        if not self.history["step"]:
            return "No data collected"

        prey_0_final = self.history["prey_0"][-1]
        prey_1_final = self.history["prey_1"][-1]
        prey_0_max = max(self.history["prey_0"])
        prey_1_max = max(self.history["prey_1"])
        prey_0_avg = float(np.mean(self.history["prey_0"]))
        prey_1_avg = float(np.mean(self.history["prey_1"]))

        winner = "TIE"
        advantage = 0
        if prey_0_final > prey_1_final:
            winner = "Prey Pop 0"
            advantage = prey_0_final - prey_1_final
        elif prey_1_final > prey_0_final:
            winner = "Prey Pop 1"
            advantage = prey_1_final - prey_0_final

        summary = f"""
{'=' * 70}
🏁 PREY TEST RESULTS
{'=' * 70}
📈 Final Population:
   Pop 0 (Math):    {prey_0_final} (Peak: {prey_0_max}, Avg: {prey_0_avg:.1f})
   Pop 1 (Rampage): {prey_1_final} (Peak: {prey_1_max}, Avg: {prey_1_avg:.1f})

🌱 Final Grass Count: {self.history['total_grass'][-1]}
🕒 Duration: {self.history['step'][-1]} steps

🏆 Winner: {winner}
🔺 Advantage: {advantage} agents
{'=' * 70}
"""
        return summary


class SimplePreyMathPolicy(SensorPolicyBase):
    """朴素数学策略：依赖观测层寻找草地、配偶并做规避。"""

    def __init__(self, env: PredPreyGrass, forage_gain: float = 0.35, mate_gain: float = 0.45):
        super().__init__(env)
        self.forage_gain = forage_gain
        self.mate_gain = mate_gain
        self.energy_threshold = 0.6
        self.age_threshold = 0.99
        self.cooldown_threshold = 0.1
        self.idle_speed = 0.25
        self.idle_steps = 40
        self.rng = np.random.default_rng()
        self.idle_vectors: Dict[str, Dict[str, np.ndarray | int]] = {}
        self.scout_vectors: Dict[str, Dict[str, np.ndarray | int]] = {}
        self.enable_pairing = bool(getattr(env, "enable_paired_reproduction", False))
        self.visibility_threshold = 0.99
        self.fertility_threshold = 0.4
        self.escape_gain = 0.36
        self.panic_gain = 0.5
        self.max_drive_gain = 0.45
        self.close_range_limit = 0.15
        self.mating_threat_threshold = 0.35

    def reset(self) -> None:
        self.idle_vectors.clear()

    def forget(self, agent_id: str) -> None:
        self.idle_vectors.pop(agent_id, None)

    def _select_sensor(self, distances: np.ndarray, qualities: Optional[np.ndarray] = None, min_quality: float = 0.0):
        visible = np.where(distances < self.visibility_threshold)[0]
        if visible.size == 0:
            return None
        if qualities is not None:
            mask = qualities[visible] > min_quality
            visible = visible[mask]
            if visible.size == 0:
                return None
            scores = distances[visible] - 0.05 * qualities[visible]
        else:
            scores = distances[visible]
        best = scores.min()
        tol = max(0.02, scores.std() * 0.5)
        candidates = visible[scores <= best + tol]
        idx = int(self.rng.choice(candidates))
        return idx, float(distances[idx])

    def _can_seek_partner(self, self_state: np.ndarray) -> bool:
        if not self.enable_pairing or self_state.size < max(10, self.self_state_dim):
            return False
        energy_ratio = float(self_state[2])
        age_ratio = float(self_state[8])
        cooldown_ratio = float(self_state[9])
        wants_flag = float(self_state[5])
        return (
            energy_ratio >= self.energy_threshold
            and age_ratio >= self.age_threshold
            and cooldown_ratio <= self.cooldown_threshold
            and wants_flag > 0.5
        )

    def _avoid_predators(self, predator_layer: np.ndarray, dirs: np.ndarray) -> Optional[Dict[str, np.ndarray | float]]:
        distances = predator_layer[:, 0]
        danger_norm = 0.35
        panic_norm = 0.15
        visible = np.where(distances < danger_norm)[0]
        if visible.size == 0:
            return None
        weights = 1.0 - distances[visible]
        if np.sum(weights) <= 1e-6:
            return None
        direction = -np.sum(dirs[visible] * weights[:, None], axis=0)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return None
        direction = (direction / norm).astype(np.float32)
        min_distance = float(np.min(distances[visible]))
        if min_distance >= danger_norm:
            return None
        urgency = 1.0 if min_distance <= panic_norm else (danger_norm - min_distance) / max(danger_norm - panic_norm, 1e-6)
        gain = self.escape_gain + (self.panic_gain - self.escape_gain) * urgency
        action = np.clip(direction * gain, -0.5, 0.5).astype(np.float32)
        return {"action": action, "direction": direction, "urgency": urgency}

    def _idle_action(self, agent_id: str) -> np.ndarray:
        entry = self.idle_vectors.get(agent_id)
        if entry is None or entry["steps"] <= 0:
            vec = self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
            vec *= self.idle_speed
            entry = {"vec": vec.astype(np.float32), "steps": self.idle_steps}
        else:
            entry["steps"] -= 1
        self.idle_vectors[agent_id] = entry
        return np.clip(entry["vec"], -0.5, 0.5).astype(np.float32)

    def _scout_action(self, agent_id: str) -> np.ndarray:
        entry = self.scout_vectors.get(agent_id)
        if entry is None or entry["steps"] <= 0:
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            vec = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * 0.3
            entry = {"vec": vec, "steps": self.idle_steps}
        else:
            entry["steps"] -= 1
        self.scout_vectors[agent_id] = entry
        return entry["vec"].copy()

    def _drive(self, sensor_idx: int, base_gain: float, dirs: np.ndarray, avoidance: Optional[Dict[str, np.ndarray | float]]):
        direction = dirs[sensor_idx]
        gain = min(self.max_drive_gain, base_gain)
        thrust = direction * gain
        if avoidance and avoidance["urgency"] > 0.0:
            weight = min(0.4, 0.4 * float(avoidance["urgency"]))
            thrust = thrust * (1.0 - weight) + avoidance["action"] * weight
        return np.clip(thrust, -0.5, 0.5).astype(np.float32)

    def get_action(self, agent_id: str, observation: Optional[np.ndarray]) -> np.ndarray:
        layers, self_state, dirs = self._prepare_layers(agent_id, observation)
        predator_layer = layers["predator"]
        env_layer = layers["env"]
        mate_layer = layers["mate"]

        avoidance = self._avoid_predators(predator_layer, dirs)
        if avoidance and avoidance["urgency"] >= 1.0:
            return avoidance["action"]

        target_sensor = None
        mode = "forage"

        allow_mating = not (avoidance and avoidance["urgency"] >= self.mating_threat_threshold)
        if allow_mating and self._can_seek_partner(self_state):
            mate = self._select_sensor(mate_layer[:, 0], mate_layer[:, 2], min_quality=self.fertility_threshold)
            if mate is not None:
                target_sensor, _ = mate
                mode = "mate"

        if target_sensor is None:
            grass = self._select_sensor(env_layer[:, 1], env_layer[:, 2], min_quality=0.05)
            if grass is not None:
                target_sensor, _ = grass
                mode = "forage"

        wall_push = self._wall_repulsion(env_layer, dirs)

        decision = {"mode": "idle", "avoid": avoidance is not None, "wall": wall_push is not None}
        if target_sensor is None:
            if avoidance:
                decision["mode"] = "avoid"
                self.debug_info[agent_id] = decision
                return avoidance["action"]
            if wall_push and wall_push["urgency"] >= 0.6:
                decision["mode"] = "wall"
                self.debug_info[agent_id] = decision
                return wall_push["action"]
            action = self._scout_action(agent_id)
            self.debug_info[agent_id] = decision
            return action

        base_gain = self.mate_gain if mode == "mate" else self.forage_gain
        combined_avoidance = avoidance
        if wall_push is not None and wall_push["urgency"] >= 0.6:
            if combined_avoidance is None or wall_push["urgency"] > combined_avoidance["urgency"]:
                combined_avoidance = wall_push
        decision.update({"mode": mode, "sensor": target_sensor})
        self.debug_info[agent_id] = decision
        return self._drive(target_sensor, base_gain, dirs, combined_avoidance)


def run_prey_test_v3(
    max_steps: int = 200000,
    target_fps: int = 30,
    verbose: bool = False,
    predator_count: Optional[int] = None,
    record_video: Optional[Path] = None,
    base_config: Optional[Dict] = None,
):
    print("\n" + "=" * 70)
    print("🧪 PREY STRATEGY TEST (Math vs Rampage)")
    print("=" * 70)
    print(f"Max Steps: {max_steps}")
    print(f"Target FPS: {target_fps}")
    print(f"Verbose: {verbose}\n")

    config_overrides: Dict[str, float | int | bool] = {
        # "n_possible_predators": 999999,
        # "n_possible_prey": 999999,
        # "n_initial_active_predator": 5,
        # "n_initial_active_prey": 10,
        # "initial_num_grass": 25,
        # "grass_min_energy": 0.1,
        # "grass_max_energy": 30.0,
        # "grass_base_growth_rate": 1.0,
        # "grass_decay_rate": 0.05,
        # "enable_grass_reproduction": True,
        # "grass_base_reproduce_prob": 0.2,
        # "grass_reproduce_threshold": 8.0,
        # "grass_reproduce_cost": 4.0,
        # "grass_offspring_energy": 5.0,
        # "grass_perception_radius": 160.0,
        # "grass_density_reference": 8.0,
        # "grass_spawn_max_attempts": 10,
        # # 'initial_energy_predator': 60.0,
        # # 'energy_loss_per_step_predator': 0.0003,
        # # 'predator_creation_energy_threshold': 60.0,
        # # 'soft_speed_limit_predator': 50.0,
        # # 'thrust_scale_predator': 100.0,

    }
    cfg_source = base_config or prey_test_config
    config = {**cfg_source, **config_overrides}
    if predator_count is not None:
        predator_count = max(0, int(predator_count))
        config["n_initial_active_predator"] = predator_count
        if predator_count == 0:
            config["allow_empty_predator_population"] = True
            config["n_possible_predators"] = 0
        else:
            config["n_possible_predators"] = max(config.get("n_possible_predators", predator_count), predator_count)

    config["max_steps"] = max_steps
    env = PredPreyGrass(config=config)

    print("🔄 Resetting environment...")
    observations, info = env.reset(seed=42)
    latest_obs: Dict[str, np.ndarray] = {agent_id: obs for agent_id, obs in observations.items()}

    math_policy = SimplePreyMathPolicy(env)
    math_policy.reset()
    rampage_agent = RampageAgent(env)
    predator_policy = PredatorChasePolicy(env)

    def normalise_label(label: str | None) -> str:
        return (label or "").lower()

    predator_plan = getattr(env, "population_plan", {}).get("predator") or ["default"]
    prey_plan = getattr(env, "population_plan", {}).get("prey") or ["default"]
    # 保证长度至少覆盖 n_populations
    if len(predator_plan) < env.n_populations:
        predator_plan = (predator_plan * (env.n_populations // len(predator_plan) + 1))[: env.n_populations]
    if len(prey_plan) < env.n_populations:
        prey_plan = (prey_plan * (env.n_populations // len(prey_plan) + 1))[: env.n_populations]

    predator_policy_by_pop: Dict[int, str] = {}
    prey_policy_by_pop: Dict[int, str] = {}

    for pop_id, label in enumerate(predator_plan[: env.n_populations]):
        key = normalise_label(label)
        if "random" in key or "rampage" in key:
            predator_policy_by_pop[pop_id] = "random"
        else:
            predator_policy_by_pop[pop_id] = "chase"

    for pop_id, label in enumerate(prey_plan[: env.n_populations]):
        key = normalise_label(label)
        if "random" in key or "rampage" in key:
            prey_policy_by_pop[pop_id] = "random"
        else:
            # 默认视为 math（代表学习策略）
            prey_policy_by_pop[pop_id] = "math"

    math_policy_populations = {
        pop_id for pop_id, mode in prey_policy_by_pop.items() if mode == "math"
    }

    monitor = PreyTestMonitor(env)
    visualizer = PredPreyVisualizer(env, width=1650, height=800, fps=target_fps)
    video_writer = None
    if record_video:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError(
                "imageio is required for video recording. Install with `pip install imageio`."
            ) from exc
        record_path = record_video.expanduser().resolve()
        record_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(str(record_path), fps=target_fps)

    print("\nControls: SPACE pause, ↑/↓ speed, R reset, ESC quit\n")

    step_count = 0
    last_print_step = 0
    print_interval = 100
    frame_time = 1.0 / target_fps
    last_frame_time = time.time()

    try:
        while visualizer.render():
            if video_writer:
                frame = visualizer.capture_frame()
                if frame is not None:
                    video_writer.append_data(frame)

            if visualizer.paused:
                time.sleep(0.01)
                last_frame_time = time.time()
                continue

            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            last_frame_time = time.time()

            actions: Dict[str, np.ndarray] = {}
            try:
                for agent_id in env.agents:
                    agent_obs = latest_obs.get(agent_id)
                    is_predator = "predator" in agent_id
                    pop_id = env.agent_population_id.get(agent_id, 0)
                    if is_predator:
                        mode = predator_policy_by_pop.get(pop_id, "chase")
                        if mode == "random":
                            actions[agent_id] = rampage_agent.get_action(agent_id)
                        else:
                            actions[agent_id] = predator_policy.get_action(agent_id, agent_obs)
                    else:
                        mode = prey_policy_by_pop.get(pop_id, "math")
                        if mode == "random":
                            actions[agent_id] = rampage_agent.get_action(agent_id)
                        else:
                            actions[agent_id] = math_policy.get_action(agent_id, agent_obs)
            except Exception as exc:
                print(f"\n❌ ERROR collecting actions: {exc}")
                import traceback

                traceback.print_exc()
                break

            try:
                observations, rewards, terminations, truncations, infos = env.step(actions)
            except Exception as exc:
                print(f"\n❌ ERROR in env.step() at step {step_count}: {exc}")
                import traceback

                traceback.print_exc()
                break

            for agent_id, obs in observations.items():
                latest_obs[agent_id] = obs
            for agent_id in list(latest_obs.keys()):
                if agent_id not in env.agents:
                    latest_obs.pop(agent_id, None)
                    math_policy.forget(agent_id)

            step_count += 1

            try:
                monitor.update(step_count)
            except Exception as exc:
                print(f"\n❌ ERROR updating monitor: {exc}")

            if step_count % 100 == 0:
                predator_count = sum(1 for a in env.agents if "predator" in a)
                prey_count = sum(1 for a in env.agents if "prey" in a)
                grass_count = len(env.grass_positions)
                print(
                    f"[Step {step_count:5d}] Predators: {predator_count:2d} | "
                    f"Prey: {prey_count:3d} | Grass: {grass_count:3d}"
                )

            if step_count - last_print_step >= print_interval:
                monitor.print_status(step_count)
                last_print_step = step_count

            should_terminate = False
            termination_reason = ""

            if truncations.get("__all__", False):
                should_terminate = True
                termination_reason = "Reached max steps"
            elif terminations.get("__all__", False):
                should_terminate = True
                predator_count = sum(1 for a in env.agents if "predator" in a)
                prey_count = sum(1 for a in env.agents if "prey" in a)
                if predator_count == 0:
                    termination_reason = "⚠️ All predators extinct"
                elif prey_count == 0:
                    termination_reason = "☠️ All prey extinct"
                else:
                    termination_reason = f"Unknown (P:{predator_count} Pr:{prey_count})"

            if should_terminate:
                print(f"\n🏁 Test ended at step {step_count}")
                print(f"   Reason: {termination_reason}")
                predator_count = sum(1 for a in env.agents if "predator" in a)
                prey_count = sum(1 for a in env.agents if "prey" in a)
                print(f"   Predators: {predator_count}")
                print(f"   Prey: {prey_count}")
                print(f"   Grass: {len(env.grass_positions)}")
                print("\n   Pausing for review... Press R to restart or ESC to quit")

                while visualizer.render():
                    if video_writer:
                        frame = visualizer.capture_frame()
                        if frame is not None:
                            video_writer.append_data(frame)
                    time.sleep(0.1)
                    if visualizer.paused and visualizer.running:
                        print("\n🔄 Resetting environment...")
                        observations, info = env.reset(seed=None)
                        step_count = 0
                        last_print_step = 0
                        monitor = PreyTestMonitor(env)
                        latest_obs = {agent_id: obs for agent_id, obs in observations.items()}
                        math_policy.reset()
                        break
                    elif not visualizer.running:
                        break

                if not visualizer.running:
                    break

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")

    finally:
        print(monitor.summarise())
        print("👋 Closing visualizer...")
        if video_writer:
            video_writer.close()
        visualizer.close()
        print("✅ Test complete!\n")
        return monitor.history


def main() -> Dict[str, list]:
    import argparse

    parser = argparse.ArgumentParser(description="Prey Strategy Test (Math vs Rampage)")
    parser.add_argument("--steps", type=int, default=20000, help="Maximum test steps (default: 20000)")
    parser.add_argument("--fps", type=int, default=30, help="Target frames per second (default: 30)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--predator-count",
        type=int,
        default=None,
        help="Override the number of initial predators (use 0 with --prey-only).",
    )
    parser.add_argument(
        "--prey-only",
        action="store_true",
        help="Shortcut for running without any predators (sets predator-count=0).",
    )
    parser.add_argument(
        "--record-video",
        type=Path,
        default=None,
        help="Optional path to save an MP4 recording of the session.",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=None,
        help="Optional path to env config (json/py). Defaults to built-in config_env_base.",
    )
    args = parser.parse_args()

    predator_count = 0 if args.prey_only else args.predator_count
    base_config = None
    if args.env_config:
        base_config = load_env_config(args.env_config)

    return run_prey_test_v3(
        max_steps=args.steps,
        target_fps=args.fps,
        verbose=args.verbose,
        predator_count=predator_count,
        record_video=args.record_video,
        base_config=base_config,
    )


if __name__ == "__main__":
    main()
