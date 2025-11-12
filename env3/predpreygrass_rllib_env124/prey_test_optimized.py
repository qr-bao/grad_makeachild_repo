"""
Prey策略测试 - 数学策略 vs Rampage 对照
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from predpreygrass.rllib.env3.predpreygrass_rllib_env124 import config_env
from predpreygrass.rllib.env3.predpreygrass_rllib_env124.rampage_agent import RampageAgent
from predpreygrass.rllib.env3.predpreygrass_rllib_env124.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.env3.predpreygrass_rllib_env124.visualizer import PredPreyVisualizer
from predpreygrass.rllib.env3.predpreygrass_rllib_env124.prey_test_config import prey_test_config



class ImmortalPredatorPolicy:
    """让捕食者和 Rampage prey 一样移动。"""
    def __init__(self, env: PredPreyGrass):
        self.rampage = RampageAgent(env)

    def get_action(self, agent_id: str) -> np.ndarray:
        return self.rampage.get_action(agent_id)


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


class SimplePreyMathPolicy:
    """朴素数学策略：能量不足时抢草，满足条件时主动找配偶。"""

    def __init__(self, env: PredPreyGrass, forage_gain: float = 0.35, mate_gain: float = 0.45):
        self.env = env
        self.forage_gain = forage_gain
        self.mate_gain = mate_gain
        self.forage_stop_radius = 12.0
        self.mate_stop_radius = 10.0
        self.energy_threshold = 0.6
        self.age_threshold = 0.99
        self.cooldown_threshold = 0.1
        self.idle_speed = 0.12
        self.idle_steps = 30
        self.rng = np.random.default_rng()
        self.idle_vectors: Dict[str, Dict[str, np.ndarray | int]] = {}
        self.brake_distance = 25.0      # 离草 ≤ 25 像素开始刹车
        self.max_approach_speed = 45.0  # 贴近时允许的最高速度
        self.brake_gain = 0.5
        sensor_range = float(getattr(env, "sensor_range", 200.0))
        self.danger_radius = float(np.clip(sensor_range * 0.45, 120.0, 280.0))
        self.panic_radius = float(np.clip(self.danger_radius * 0.45, 40.0, self.danger_radius * 0.75))
        self.prediction_horizon = 0.35
        self.avoidance_activation_threshold = 0.08
        self.max_avoidance_bias = 0.65
        self.escape_gain = 0.36
        self.panic_gain = 0.48
        self.threat_speed_bonus = 0.08
        self.brake_override_threshold = 0.55
        self.max_drive_gain = 0.45
        self.close_range_gain_limit = 0.18
        self.mating_threat_threshold = 0.35
        sample_space = next(iter(env.observation_spaces.values()))
        self._blank_observation = np.zeros(sample_space.shape[0], dtype=np.float32)
        self.self_state_dim = getattr(env, "self_state_dim", 10)

    def reset(self) -> None:
        self.idle_vectors.clear()

    def forget(self, agent_id: str) -> None:
        self.idle_vectors.pop(agent_id, None)

    def get_action(self, agent_id: str, observation: Optional[np.ndarray]) -> np.ndarray:
        if agent_id not in self.env.agent_positions:
            return np.zeros(2, dtype=np.float32)

        if observation is not None:
            obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        else:
            obs = self._blank_observation

        self_state = obs[-self.self_state_dim :] if obs.size >= self.self_state_dim else self._blank_observation[-self.self_state_dim :]

        avoidance_data = self._compute_predator_avoidance(agent_id)
        if avoidance_data and avoidance_data["urgency"] >= 1.0:
            return avoidance_data["action"]

        target = None
        mode = "forage"

        allow_mating = not (
            avoidance_data and avoidance_data["urgency"] >= self.mating_threat_threshold
        )

        if allow_mating and self._can_seek_partner(agent_id, self_state):
            partner = self._find_partner(agent_id)
            if partner is not None:
                target = partner
                mode = "mate"

        if target is None:
            grass = self._find_grass(agent_id)
            if grass is not None:
                target = grass
                mode = "forage"

        if target is None:
            if avoidance_data:
                return avoidance_data["action"]
            return self._idle_action(agent_id)

        return self._drive_towards(agent_id, target, mode, avoidance=avoidance_data)

    def _can_seek_partner(self, agent_id: str, self_state: np.ndarray) -> bool:
        if not getattr(self.env, "enable_paired_reproduction", False):
            return False
        if not self.env.agent_wants_to_mate.get(agent_id, False):
            return False

        energy_ratio = float(self_state[2]) if self_state.size > 2 else 0.0
        age_ratio = float(self_state[8]) if self_state.size > 8 else 0.0
        cooldown_ratio = float(self_state[9]) if self_state.size > 9 else 1.0
        wants_flag = float(self_state[5]) if self_state.size > 5 else 0.0

        return (
            energy_ratio >= self.energy_threshold
            and age_ratio >= self.age_threshold
            and cooldown_ratio <= self.cooldown_threshold
            and wants_flag > 0.5
        )

    def _find_partner(self, agent_id: str) -> Optional[Dict[str, np.ndarray | float]]:
        origin = np.array(self.env.agent_positions.get(agent_id, (0.0, 0.0)), dtype=np.float32)
        pop_id = self.env.agent_population_id.get(agent_id, 0)
        best_id = None
        best_pos = None
        best_dist = float("inf")
        mating_distance = getattr(self.env, "mating_distance", 100.0)

        for other_id, pos in self.env.agent_positions.items():
            if other_id == agent_id or "prey" not in other_id:
                continue
            if other_id not in self.env.agents:
                continue
            if self.env.agent_population_id.get(other_id, -1) != pop_id:
                continue
            if not self.env.agent_wants_to_mate.get(other_id, False):
                continue

            other_pos = np.array(pos, dtype=np.float32)
            distance = float(np.linalg.norm(other_pos - origin))
            if distance < best_dist and distance <= mating_distance:
                best_id = other_id
                best_pos = other_pos
                best_dist = distance

        if best_id is None:
            return None

        return {"id": best_id, "position": best_pos, "distance": best_dist}

    def _find_grass(self, agent_id: str) -> Optional[Dict[str, np.ndarray | float]]:
        origin = np.array(self.env.agent_positions.get(agent_id, (0.0, 0.0)), dtype=np.float32)
        best_id = None
        best_pos = None
        best_dist = float("inf")
        max_distance = getattr(self.env, "sensor_range", 200.0) * 1.2
        min_energy = getattr(self.env, "grass_min_energy", 0.0)

        for grass_id, pos in self.env.grass_positions.items():
            energy = self.env.grass_energies.get(grass_id, 0.0)
            if energy <= min_energy:
                continue
            grass_pos = np.array(pos, dtype=np.float32)
            distance = float(np.linalg.norm(grass_pos - origin))
            if distance < best_dist and distance <= max_distance:
                best_id = grass_id
                best_pos = grass_pos
                best_dist = distance

        if best_id is None:
            return None

        return {"id": best_id, "position": best_pos, "distance": best_dist}

    def _compute_predator_avoidance(self, agent_id: str) -> Optional[Dict[str, np.ndarray | float]]:
        origin = np.array(self.env.agent_positions.get(agent_id, (0.0, 0.0)), dtype=np.float32)
        aggregated = np.zeros(2, dtype=np.float32)
        min_distance = float("inf")
        closest_vector = None

        for other_id in self.env.agents:
            if "predator" not in other_id:
                continue
            pos = self.env.agent_positions.get(other_id)
            if pos is None:
                continue
            pos_vec = np.array(pos, dtype=np.float32)
            body = self.env.agent_bodies.get(other_id)
            if body is not None:
                velocity = np.array([body.velocity.x, body.velocity.y], dtype=np.float32)
                pos_vec = pos_vec + velocity * self.prediction_horizon

            diff = origin - pos_vec
            distance = float(np.linalg.norm(diff))
            if distance < 1e-6 or distance > self.danger_radius:
                continue

            weight = (self.danger_radius - distance) / self.danger_radius
            aggregated += (diff / distance) * weight
            if distance < min_distance:
                min_distance = distance
                closest_vector = diff

        if min_distance == float("inf"):
            return None

        if min_distance <= self.panic_radius:
            urgency = 1.0
        else:
            denom = max(self.danger_radius - self.panic_radius, 1e-6)
            urgency = (self.danger_radius - min_distance) / denom
            urgency = float(np.clip(urgency, 0.0, 1.0))

        if urgency < self.avoidance_activation_threshold:
            return None

        magnitude = float(np.linalg.norm(aggregated))
        if magnitude < 1e-6 and closest_vector is not None:
            direction = closest_vector / (np.linalg.norm(closest_vector) + 1e-6)
        else:
            direction = aggregated / (magnitude + 1e-6)

        direction = direction.astype(np.float32)
        gain = self.escape_gain + (self.panic_gain - self.escape_gain) * urgency
        action = np.clip(direction * gain, -0.5, 0.5).astype(np.float32)

        return {
            "action": action,
            "direction": direction,
            "urgency": urgency,
        }

    def _blend_directions(self, primary: np.ndarray, secondary: np.ndarray, weight: float) -> np.ndarray:
        weight = float(np.clip(weight, 0.0, 1.0))
        if weight <= 1e-6:
            return primary.astype(np.float32)
        combined = primary * (1.0 - weight) + secondary * weight
        norm = float(np.linalg.norm(combined))
        if norm < 1e-6:
            return primary.astype(np.float32)
        return (combined / norm).astype(np.float32)

    def _drive_towards(
        self,
        agent_id: str,
        target: Dict[str, np.ndarray | float],
        mode: str,
        avoidance: Optional[Dict[str, np.ndarray | float]] = None,
    ) -> np.ndarray:
        origin = np.array(self.env.agent_positions.get(agent_id, (0.0, 0.0)), dtype=np.float32)
        vector = target["position"] - origin
        distance = max(float(np.linalg.norm(vector)), 1e-6)

        stop_radius = self.mate_stop_radius if mode == "mate" else self.forage_stop_radius
        if distance <= stop_radius:
            if avoidance and avoidance["urgency"] >= self.avoidance_activation_threshold:
                return avoidance["action"]
            return np.zeros(2, dtype=np.float32)
        body = self.env.agent_bodies.get(agent_id)
        strong_threat = avoidance and avoidance["urgency"] >= self.brake_override_threshold
        if body is not None:
            vel = np.array([body.velocity.x, body.velocity.y], dtype=np.float32)
            speed = float(np.linalg.norm(vel))
            if (
                not strong_threat
                and distance < self.brake_distance
                and speed > self.max_approach_speed
            ):
                if speed > 1e-6:
                    brake_dir = -vel / speed
                else:
                    brake_dir = np.zeros(2, dtype=np.float32)
                thrust = np.clip(brake_dir * self.brake_gain, -0.5, 0.5).astype(np.float32)
                return thrust
        # 重新计算方向
        direction = vector / distance
        if avoidance and avoidance["urgency"] > 0.0 and np.linalg.norm(avoidance["direction"]) > 1e-6:
            blend_weight = min(
                self.max_avoidance_bias,
                self.max_avoidance_bias * float(avoidance["urgency"]),
            )
            direction = self._blend_directions(direction, avoidance["direction"], blend_weight)

        base_gain = self.mate_gain if mode == "mate" else self.forage_gain
        if distance < self.brake_distance:
            gain = min(base_gain, self.close_range_gain_limit)  # 近距离慢推
        else:
            gain = base_gain  # 远距离沿用原增益

        if avoidance:
            gain = min(
                self.max_drive_gain,
                gain + self.threat_speed_bonus * float(avoidance["urgency"]),
            )

        thrust = np.clip(direction * gain, -0.5, 0.5).astype(np.float32)
        return thrust

    def _idle_action(self, agent_id: str) -> np.ndarray:
        entry = self.idle_vectors.get(agent_id)
        if entry is None or entry["steps"] <= 0:
            vec = self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vec *= self.idle_speed
            entry = {"vec": vec.astype(np.float32), "steps": self.idle_steps}
        else:
            entry["steps"] -= 1

        self.idle_vectors[agent_id] = entry
        return np.clip(entry["vec"], -0.5, 0.5).astype(np.float32)


def run_prey_test_v3(
    max_steps: int = 200000,
    target_fps: int = 30,
    verbose: bool = False,
    predator_count: Optional[int] = None,
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
    config = {**prey_test_config, **config_overrides}
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
    math_policy_populations = {0}
    rampage_agent = RampageAgent(env)
    predator_policy = ImmortalPredatorPolicy(env)

    monitor = PreyTestMonitor(env)
    visualizer = PredPreyVisualizer(env, width=1650, height=800, fps=target_fps)

    print("\nControls: SPACE pause, ↑/↓ speed, R reset, ESC quit\n")

    step_count = 0
    last_print_step = 0
    print_interval = 100
    frame_time = 1.0 / target_fps
    last_frame_time = time.time()

    try:
        while visualizer.render():
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
                    if "predator" in agent_id:
                        actions[agent_id] = predator_policy.get_action(agent_id)
                    else:
                        pop_id = env.agent_population_id.get(agent_id, 0)
                        if pop_id in math_policy_populations:
                            actions[agent_id] = math_policy.get_action(agent_id, agent_obs)
                        else:
                            actions[agent_id] = rampage_agent.get_action(agent_id)
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
    args = parser.parse_args()

    predator_count = 0 if args.prey_only else args.predator_count

    return run_prey_test_v3(
        max_steps=args.steps,
        target_fps=args.fps,
        verbose=args.verbose,
        predator_count=predator_count,
    )


if __name__ == "__main__":
    main()
