"""
PredPreyGrass 环境可视化模块
使用纯 Pygame 实现实时渲染和交互功能
"""
import pygame
import numpy as np
import os
import csv
from datetime import datetime
from typing import Optional, Tuple, Dict, List
from predpreygrass.rllib.env3.predpreygrass_rllib_env124.observation_visualizer import ObservationSpaceVisualizer

class PredPreyVisualizer:
    """
    可视化渲染器
    
    窗口布局：
    ┌─────────────────┬─────────────┐
    │                 │             │
    │  环境渲染区      │  种群图表    │
    │  (750×750)      │  (450×350)  │
    │                 │             │
    │                 ├─────────────┤
    │                 │             │
    │                 │  Agent详情   │
    │                 │  (450×450)  │
    └─────────────────┴─────────────┘
    """
    
    # 颜色定义
    COLOR_BG = (240, 240, 240)           # 背景色
    COLOR_ENV_BG = (250, 250, 250)       # 环境背景
    COLOR_WALL = (100, 100, 100)         # 墙壁
    COLOR_PREDATOR = (220, 50, 50)       # 捕食者
    COLOR_PREY = (50, 120, 220)          # 猎物
    COLOR_GRASS = (100, 200, 80)         # 草
    COLOR_SELECTED = (255, 215, 0)       # 选中高亮
    COLOR_ATE = (100, 255, 100)          # 刚吃东西
    COLOR_TEXT = (40, 40, 40)            # 文本
    COLOR_PANEL_BG = (255, 255, 255)     # 面板背景
    COLOR_BORDER = (200, 200, 200)       # 边框
    
    # 图表颜色
    COLOR_CHART_PRED = (220, 50, 50)     # 捕食者曲线
    COLOR_CHART_PREY = (50, 120, 220)    # 猎物曲线
    COLOR_CHART_GRID = (220, 220, 220)   # 网格线
    
    def __init__(self, env, width=1650, height=800, fps=30):
        """
        初始化可视化器
        
        Args:
            env: PredPreyGrass 环境实例
            width: 窗口宽度（默认 1650，支持三栏布局）
            height: 窗口高度
            fps: 目标帧率
        """
        self.env = env
        self.width = width
        self.height = height
        self.fps = fps
        
        # 初始化 Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("PredPreyGrass Simulation")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)
        
        # === 布局参数（三栏布局）===
        self.env_width = 750           # 左：环境渲染
        self.env_height = 750          # 环境渲染高度
        self.panel_width = 450         # 中：种群图表 + 详情
        self.obs_width = 450           # 右：观察空间
        self.chart_height = 350        # 中上：图表高度
        self.info_height = height - self.chart_height  # 中下：详情高度
        
        # 环境渲染缩放比例
        self.scale = min(
            self.env_width / env.world_width,
            self.env_height / env.world_height
        )
        
        # === 创建观察空间可视化器 ===
        self.obs_visualizer = ObservationSpaceVisualizer(
            width=self.obs_width,
            height=height
        )
        
        # 状态变量
        self.selected_agent = None
        self.paused = False
        self.running = True
        self.speed = 1  # 播放速度倍数
        
        # === 修改：移除滚动窗口限制，显示全部历史 ===
        # self.history_window = 500  # ← 删除这行
        
        # === 新增：数据保存功能 ===
        self._setup_data_folder()
        self.data_save_interval = 100  # 每100步保存一次数据
        self.last_save_step = 0
        
    def _setup_data_folder(self):
        """创建数据保存文件夹"""
        # 创建主数据文件夹
        self.data_folder = "simulation_data"
        os.makedirs(self.data_folder, exist_ok=True)
        
        # 为本次运行创建子文件夹（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_folder = os.path.join(self.data_folder, f"run_{timestamp}")
        os.makedirs(self.run_folder, exist_ok=True)
        
        # 创建CSV文件路径
        self.population_csv = os.path.join(self.run_folder, "population_history.csv")
        
        # 初始化CSV文件（写入表头）
        with open(self.population_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            # 动态生成表头
            header = ['step']
            for agent_type in ["predator", "prey"]:
                for pop_id in range(self.env.n_populations):
                    header.append(f"{agent_type}_{pop_id}")
            writer.writerow(header)
        
        print(f"📁 Data will be saved to: {self.run_folder}")
    
    def _save_data(self):
        """保存当前数据到CSV"""
        if not hasattr(self.env, 'population_history'):
            return
        
        history = self.env.population_history
        if len(history['steps']) == 0:
            return
        
        current_step = self.env.current_step
        
        # 只保存新数据（从上次保存点之后）
        if current_step <= self.last_save_step:
            return
        
        # 追加写入CSV
        with open(self.population_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # 找到需要保存的数据范围
            start_idx = None
            for idx, step in enumerate(history['steps']):
                if step > self.last_save_step:
                    start_idx = idx
                    break
            
            if start_idx is None:
                return
            
            # 写入新数据
            for idx in range(start_idx, len(history['steps'])):
                row = [history['steps'][idx]]
                for agent_type in ["predator", "prey"]:
                    for pop_id in range(self.env.n_populations):
                        pop_key = f"{agent_type}_{pop_id}"
                        if pop_key in history:
                            row.append(history[pop_key][idx])
                        else:
                            row.append(0)
                writer.writerow(row)
        
        self.last_save_step = current_step
        print(f"💾 Data saved up to step {current_step}")
    
    def render(self) -> bool:
        """
        渲染一帧
        
        Returns:
            bool: 是否继续运行
        """
        try:
            # 处理事件
            if not self._handle_events():
                return False
            
            # === 定期保存数据 ===
            if self.env.current_step - self.last_save_step >= self.data_save_interval:
                self._save_data()
            
            # 清空屏幕
            self.screen.fill(self.COLOR_BG)
            
            # === 绘制三栏布局 ===
            try:
                self._draw_environment()
            except Exception as e:
                print(f"[ERROR] Failed to draw environment: {e}")
                import traceback
                traceback.print_exc()
            
            try:
                self._draw_population_chart()
            except Exception as e:
                print(f"[ERROR] Failed to draw population chart: {e}")
                import traceback
                traceback.print_exc()
            
            try:
                self._draw_agent_info()
            except Exception as e:
                print(f"[ERROR] Failed to draw agent info: {e}")
                import traceback
                traceback.print_exc()
                self.selected_agent = None
            
            try:
                self._draw_observation_space()
            except Exception as e:
                print(f"[ERROR] Failed to draw observation space: {e}")
                import traceback
                traceback.print_exc()
                self.selected_agent = None
            
            # 绘制控制提示
            try:
                self._draw_controls()
            except Exception as e:
                print(f"[ERROR] Failed to draw controls: {e}")
            
            # 更新显示
            pygame.display.flip()
            self.clock.tick(self.fps * self.speed)
            
            return self.running
        
        except Exception as e:
            print(f"[FATAL ERROR] Render loop crashed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _handle_events(self) -> bool:
        """处理用户输入事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._save_data()  # 退出前保存数据
                self.running = False
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._save_data()  # 退出前保存数据
                    self.running = False
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self._save_data()  # 重置前保存当前数据
                    self.env.reset()
                    self.selected_agent = None
                    # 为新的运行创建新的数据文件夹
                    self._setup_data_folder()
                    self.last_save_step = 0
                elif event.key == pygame.K_UP:
                    self.speed = min(self.speed + 1, 10)
                elif event.key == pygame.K_DOWN:
                    self.speed = max(self.speed - 1, 1)
                elif event.key == pygame.K_s:  # 新增：手动保存
                    self._save_data()
                    print("💾 Manual save triggered")
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键
                    self._handle_click(event.pos)
        
        return True
    
    def _handle_click(self, pos: Tuple[int, int]):
        """处理鼠标点击事件"""
        x, y = pos
        
        # === 计算环境渲染区域的偏移 ===
        env_render_height = min(750, self.height)
        y_offset = (self.height - env_render_height) // 2
        
        # 只处理环境区域的点击（考虑偏移）
        if x > self.env_width:
            return
        if y < y_offset or y > y_offset + env_render_height:
            return
        
        # 转换到环境坐标（减去偏移）
        env_x = x / self.scale
        env_y = (y - y_offset) / self.scale
        
        # 查找点击的 agent
        min_dist = float('inf')
        clicked_agent = None
        
        for agent_id, agent_pos in self.env.agent_positions.items():
            dist = np.sqrt((agent_pos[0] - env_x)**2 + (agent_pos[1] - env_y)**2)
            if dist < self.env.agent_radius and dist < min_dist:
                min_dist = dist
                clicked_agent = agent_id
        
        self.selected_agent = clicked_agent
    
    def _draw_environment(self):
        """绘制环境区域"""
        # 创建环境表面
        env_surface = pygame.Surface((self.env_width, min(750, self.height)))
        env_surface.fill(self.COLOR_ENV_BG)
        
        # 绘制边界墙
        pygame.draw.rect(
            env_surface,
            self.COLOR_WALL,
            (0, 0, self.env_width, self.env_height),
            3
        )
        
        # 绘制草
        for grass_id, grass_pos in self.env.grass_positions.items():
            energy = self.env.grass_energies.get(grass_id, 0)
            if energy > 0:
                screen_x = int(grass_pos[0] * self.scale)
                screen_y = int(grass_pos[1] * self.scale)
                radius = max(3, int(self.env.agent_radius * 0.6 * self.scale))
                
                pygame.draw.circle(env_surface, self.COLOR_GRASS, (screen_x, screen_y), radius)
        
        # 绘制 agents
        for agent_id in self.env.agents:
            if agent_id not in self.env.agent_positions:
                continue
            
            pos = self.env.agent_positions[agent_id]
            screen_x = int(pos[0] * self.scale)
            screen_y = int(pos[1] * self.scale)
            radius = max(3, int(self.env.agent_radius * self.scale))
            
            # 根据种群ID调整颜色
            pop_id = self.env.agent_population_id.get(agent_id, 0)
            
            if "predator" in agent_id:
                if pop_id == 0:
                    color = (220, 50, 50)
                elif pop_id == 1:
                    color = (255, 100, 100)
                else:
                    color = self.COLOR_PREDATOR
            else:
                if pop_id == 0:
                    color = (50, 120, 220)
                elif pop_id == 1:
                    color = (100, 170, 255)
                else:
                    color = self.COLOR_PREY
            
            # 绘制 agent
            pygame.draw.circle(env_surface, color, (screen_x, screen_y), radius)
            
            # 刚吃东西的效果
            if agent_id in self.env.agents_just_ate:
                pygame.draw.circle(
                    env_surface,
                    self.COLOR_ATE,
                    (screen_x, screen_y),
                    radius + 3,
                    2
                )
            
            # 选中高亮
            if agent_id == self.selected_agent:
                pygame.draw.circle(
                    env_surface,
                    self.COLOR_SELECTED,
                    (screen_x, screen_y),
                    radius + 5,
                    3
                )
                
                if self.env.enable_continuous_space:
                    self._draw_sensor_rays(env_surface, agent_id, screen_x, screen_y)
                
                algorithm = self.env.agent_algorithm.get(agent_id, "?")
                id_text = self.font_small.render(
                    f"{agent_id} [P{pop_id}:{algorithm}]",
                    True,
                    self.COLOR_TEXT
                )
                env_surface.blit(id_text, (screen_x + radius + 5, screen_y - 10))
            else:
                label = self.font_small.render(f"P{pop_id}", True, (100, 100, 100))
                env_surface.blit(label, (screen_x - 8, screen_y - radius - 12))
        
        # 绘制暂停提示
        if self.paused:
            pause_text = self.font_large.render("PAUSED", True, (255, 0, 0))
            text_rect = pause_text.get_rect(center=(self.env_width // 2, 30))
            env_surface.blit(pause_text, text_rect)
        
        # 绘制步数
        step_text = self.font_medium.render(
            f"Step: {self.env.current_step}",
            True,
            self.COLOR_TEXT
        )
        env_surface.blit(step_text, (10, 10))
        
        # 将环境表面绘制到主屏幕
        y_offset = (self.height - min(750, self.height)) // 2
        self.screen.blit(env_surface, (0, y_offset))
    
    def _draw_population_chart(self):
        """绘制分种群数量变化图表（累计全部历史数据）"""
        # 创建图表表面
        chart_surface = pygame.Surface((self.panel_width, self.chart_height))
        chart_surface.fill(self.COLOR_PANEL_BG)
        
        # 绘制标题
        title = self.font_medium.render("Population History (Cumulative)", True, self.COLOR_TEXT)
        chart_surface.blit(title, (10, 10))
        
        # 获取历史数据
        if not hasattr(self.env, 'population_history'):
            self.screen.blit(chart_surface, (self.env_width, 0))
            return
        
        history = self.env.population_history
        if len(history['steps']) < 2:
            self.screen.blit(chart_surface, (self.env_width, 0))
            return
        
        # === 修改：显示全部历史数据（不再限制窗口）===
        steps = history['steps']  # 显示全部
        
        # 获取所有种群的数据
        n_populations = self.env.n_populations
        predator_data = {}
        prey_data = {}
        
        for pop_id in range(n_populations):
            pred_key = f"predator_{pop_id}"
            prey_key = f"prey_{pop_id}"
            
            if pred_key in history:
                predator_data[pop_id] = history[pred_key]  # 全部数据
            if prey_key in history:
                prey_data[pop_id] = history[prey_key]  # 全部数据
        
        # 图表区域
        chart_x = 50
        chart_y = 50
        chart_w = self.panel_width - 70
        chart_h = self.chart_height - 100
        
        # 绘制边框
        pygame.draw.rect(
            chart_surface,
            self.COLOR_BORDER,
            (chart_x, chart_y, chart_w, chart_h),
            1
        )
        
        # 计算缩放
        all_values = []
        for data in predator_data.values():
            all_values.extend(data)
        for data in prey_data.values():
            all_values.extend(data)
        
        max_pop = max(all_values) if all_values else 1
        x_scale = chart_w / max(len(steps) - 1, 1)
        y_scale = chart_h / max_pop
        
        # 绘制网格线
        for i in range(5):
            y = chart_y + chart_h - int(i * chart_h / 4)
            pygame.draw.line(
                chart_surface,
                self.COLOR_CHART_GRID,
                (chart_x, y),
                (chart_x + chart_w, y)
            )
            label = self.font_small.render(
                str(int(i * max_pop / 4)),
                True,
                self.COLOR_TEXT
            )
            chart_surface.blit(label, (5, y - 10))
        
        # 颜色生成函数
        def get_population_color(agent_type, pop_id):
            if agent_type == "predator":
                if pop_id == 0:
                    return (220, 50, 50)
                elif pop_id == 1:
                    return (255, 100, 100)
                else:
                    offset = pop_id * 30
                    return (min(255, 220 + offset), min(255, 50 + offset), 50)
            else:
                if pop_id == 0:
                    return (50, 120, 220)
                elif pop_id == 1:
                    return (100, 170, 255)
                else:
                    offset = pop_id * 30
                    return (50, min(255, 120 + offset), min(255, 220 + offset))
        
        # 绘制Predator各种群曲线
        for pop_id, data in predator_data.items():
            if len(data) > 1:
                points = [
                    (
                        chart_x + int(i * x_scale),
                        chart_y + chart_h - int(data[i] * y_scale)
                    )
                    for i in range(len(data))
                ]
                color = get_population_color("predator", pop_id)
                pygame.draw.lines(chart_surface, color, False, points, 2)
        
        # 绘制Prey各种群曲线
        for pop_id, data in prey_data.items():
            if len(data) > 1:
                points = [
                    (
                        chart_x + int(i * x_scale),
                        chart_y + chart_h - int(data[i] * y_scale)
                    )
                    for i in range(len(data))
                ]
                color = get_population_color("prey", pop_id)
                pygame.draw.lines(chart_surface, color, False, points, 2)
        
        # 图例
        legend_y = chart_y + chart_h + 10
        legend_x_start = chart_x
        legend_spacing = 120
        
        col = 0
        
        # Predator 图例
        for pop_id in sorted(predator_data.keys()):
            color = get_population_color("predator", pop_id)
            x = legend_x_start + col * legend_spacing
            
            pygame.draw.circle(chart_surface, color, (x + 10, legend_y), 5)
            
            algo_name = self.env.population_display_info.get(f"predator_{pop_id}", "?")
            current_count = predator_data[pop_id][-1] if predator_data[pop_id] else 0
            text = self.font_small.render(
                f"P{pop_id}:{algo_name} ({current_count})",
                True,
                self.COLOR_TEXT
            )
            chart_surface.blit(text, (x + 20, legend_y - 8))
            
            col += 1
        
        # Prey 图例
        legend_y += 20
        col = 0
        
        for pop_id in sorted(prey_data.keys()):
            color = get_population_color("prey", pop_id)
            x = legend_x_start + col * legend_spacing
            
            pygame.draw.circle(chart_surface, color, (x + 10, legend_y), 5)
            
            algo_name = self.env.population_display_info.get(f"prey_{pop_id}", "?")
            current_count = prey_data[pop_id][-1] if prey_data[pop_id] else 0
            text = self.font_small.render(
                f"p{pop_id}:{algo_name} ({current_count})",
                True,
                self.COLOR_TEXT
            )
            chart_surface.blit(text, (x + 20, legend_y - 8))
            
            col += 1
        
        # 将图表绘制到主屏幕
        self.screen.blit(chart_surface, (self.env_width, 0))
    
    # === 保留其他方法（未修改）===
    def _draw_agent_info(self):
        """绘制 Agent 详情面板"""
        info_surface = pygame.Surface((self.panel_width, self.info_height))
        info_surface.fill(self.COLOR_PANEL_BG)
        
        pygame.draw.rect(
            info_surface,
            self.COLOR_BORDER,
            (0, 0, self.panel_width, self.info_height),
            1
        )
        
        title = self.font_medium.render("Agent Details", True, self.COLOR_TEXT)
        info_surface.blit(title, (10, 10))
        
        if self.selected_agent is None:
            hint = self.font_small.render(
                "Click an agent to view details",
                True,
                (150, 150, 150)
            )
            info_surface.blit(hint, (10, 50))
            self.screen.blit(info_surface, (self.env_width, self.chart_height))
            return
        
        if self.selected_agent not in self.env.agents:
            dead_text = self.font_small.render(
                f"{self.selected_agent} is dead",
                True,
                (255, 0, 0)
            )
            info_surface.blit(dead_text, (10, 50))
            self.selected_agent = None
            self.screen.blit(info_surface, (self.env_width, self.chart_height))
            return
        
        agent = self.selected_agent
        
        try:
            y = 40
            line_height = 25
            
            self._draw_section_title(info_surface, "Basic Info", 10, y)
            y += line_height
            
            self._draw_info_line(info_surface, "ID", agent, 10, y)
            y += line_height
            
            agent_type = "🔴 Predator" if "predator" in agent else "🔵 Prey"
            self._draw_info_line(info_surface, "Type", agent_type, 10, y)
            y += line_height
            
            pos = self.env.agent_positions.get(agent, (0, 0))
            self._draw_info_line(info_surface, "Position", f"({pos[0]:.1f}, {pos[1]:.1f})", 10, y)
            y += line_height + 5
            
            pop_id = self.env.agent_population_id.get(agent, 0)
            self._draw_info_line(info_surface, "Population", f"Pop {pop_id}", 10, y)
            y += line_height
            
            algorithm = self.env.agent_algorithm.get(agent, "Unknown")
            self._draw_info_line(info_surface, "Algorithm", algorithm, 10, y, highlight=True)
            y += line_height + 5
            
            self._draw_section_title(info_surface, "Energy", 10, y)
            y += line_height
            
            energy = self.env.agent_energies.get(agent, 0)
            max_energy = (self.env.max_energy_predator if "predator" in agent 
                        else self.env.max_energy_prey)
            self._draw_progress_bar(info_surface, "Energy", energy, max_energy, 10, y)
            y += line_height + 5
            
            self._draw_section_title(info_surface, "Rewards", 10, y)
            y += line_height
            
            cumulative = self.env.cumulative_rewards.get(agent, 0)
            self._draw_info_line(info_surface, "Cumulative", f"{cumulative:.2f}", 10, y)
            y += line_height + 5
            
            self._draw_section_title(info_surface, "Evolution", 10, y)
            y += line_height
            
            age = self.env.agent_age.get(agent, 0)
            self._draw_info_line(info_surface, "Age", f"{age} steps", 10, y)
            y += line_height
            
            generation = self.env.agent_generation.get(agent, 0)
            self._draw_info_line(info_surface, "Generation", f"Gen {generation}", 10, y)
            y += line_height + 5
            
            self._draw_section_title(info_surface, "Status", 10, y)
            y += line_height
            
            if self.env.enable_hunger:
                steps_hungry = self.env.agent_steps_since_last_meal.get(agent, 0)
                self._draw_info_line(info_surface, "Hungry for", f"{steps_hungry} steps", 10, y)
                y += line_height
            
            if self.env.enable_paired_reproduction:
                last_repro = self.env.agent_last_reproduction_step.get(agent, -1000)
                cooldown = max(0, self.env.reproduction_cooldown - (self.env.current_step - last_repro))
                if cooldown > 0:
                    self._draw_info_line(info_surface, "Repro CD", f"{cooldown} steps", 10, y)
                else:
                    self._draw_info_line(info_surface, "Repro CD", "Ready!", 10, y)
                y += line_height
            
            if self.env.enable_continuous_space and agent in self.env.agent_bodies:
                y += 5
                self._draw_section_title(info_surface, "Physics", 10, y)
                y += line_height
                
                body = self.env.agent_bodies[agent]
                velocity = body.velocity
                self._draw_info_line(
                    info_surface,
                    "Velocity",
                    f"({velocity.x:.1f}, {velocity.y:.1f})",
                    10,
                    y
                )
                y += line_height
                
                speed = velocity.length
                self._draw_info_line(info_surface, "Speed", f"{speed:.2f}", 10, y)
                y += line_height
            
            if self.env.enable_continuous_space and agent in self.env.agents:
                y += 5
                self._draw_section_title(info_surface, "Sensors", 10, y)
                y += line_height
                
                if agent not in self.env.agent_positions:
                    raise KeyError(f"Agent {agent} not in agent_positions")
                
                agent_pos = self.env.agent_positions[agent]
                sensor_angles = np.linspace(0, 2 * np.pi, self.env.n_sensors, endpoint=False)
                
                grass_detected = []
                prey_detected = []
                predator_detected = []
                wall_detected = []
                
                for angle in sensor_angles:
                    ray_result = self.env._cast_sensor_ray(agent_pos, angle, self.env.sensor_range)
                    if ray_result['hit']:
                        obj_type = ray_result['object_type']
                        distance = ray_result['distance'] * self.env.sensor_range
                        
                        if obj_type == 'grass':
                            grass_detected.append(distance)
                        elif obj_type == 'prey':
                            prey_detected.append(distance)
                        elif obj_type == 'predator':
                            predator_detected.append(distance)
                        elif obj_type == 'wall':
                            wall_detected.append(distance)
                
                is_predator = "predator" in agent
                
                if is_predator:
                    if prey_detected:
                        closest = min(prey_detected)
                        self._draw_info_line(
                            info_surface, 
                            "🟡 Prey", 
                            f"{len(prey_detected)} ({closest:.1f}px)", 
                            10, y,
                            highlight=True
                        )
                        y += line_height
                    else:
                        self._draw_info_line(info_surface, "🟡 Prey", "None", 10, y)
                        y += line_height
                    
                    if predator_detected:
                        closest = min(predator_detected)
                        self._draw_info_line(
                            info_surface, 
                            "Predators", 
                            f"{len(predator_detected)} ({closest:.1f}px)", 
                            10, y
                        )
                        y += line_height
                else:
                    if grass_detected:
                        closest = min(grass_detected)
                        self._draw_info_line(
                            info_surface, 
                            "🟢 Grass", 
                            f"{len(grass_detected)} ({closest:.1f}px)", 
                            10, y,
                            highlight=True
                        )
                        y += line_height
                    else:
                        self._draw_info_line(info_surface, "🟢 Grass", "None", 10, y)
                        y += line_height
                    
                    if predator_detected:
                        closest = min(predator_detected)
                        self._draw_info_line(
                            info_surface, 
                            "🔴 Predator", 
                            f"{len(predator_detected)} ({closest:.1f}px)", 
                            10, y,
                            warning=True
                        )
                        y += line_height
                    else:
                        self._draw_info_line(info_surface, "🔴 Predator", "None", 10, y)
                        y += line_height
                
                if wall_detected:
                    closest = min(wall_detected)
                    self._draw_info_line(info_surface, "Walls", f"{closest:.1f}px", 10, y)
                    y += line_height
        
        except Exception as e:
            print(f"[ERROR] Failed to draw info for {self.selected_agent}: {e}")
            import traceback
            traceback.print_exc()
            
            error_text = self.font_small.render(
                f"Error: Agent data corrupted",
                True,
                (255, 0, 0)
            )
            info_surface.blit(error_text, (10, 50))
            
            self.selected_agent = None
        
        self.screen.blit(info_surface, (self.env_width, self.chart_height))
    
    def _draw_section_title(self, surface, title, x, y):
        """绘制分区标题"""
        text = self.font_small.render(f"【{title}】", True, (80, 80, 80))
        surface.blit(text, (x, y))
    
    def _draw_info_line(self, surface, label, value, x, y, highlight=False, warning=False):
        """绘制信息行"""
        label_text = self.font_small.render(f"{label}:", True, self.COLOR_TEXT)
        
        if warning:
            value_color = (255, 50, 50)
        elif highlight:
            value_color = (50, 200, 50)
        else:
            value_color = (50, 50, 200)
        
        value_text = self.font_small.render(str(value), True, value_color)
        
        surface.blit(label_text, (x, y))
        surface.blit(value_text, (x + 120, y))
    
    def _draw_progress_bar(self, surface, label, value, max_value, x, y):
        """绘制进度条"""
        bar_width = 200
        bar_height = 15
        
        label_text = self.font_small.render(f"{label}:", True, self.COLOR_TEXT)
        surface.blit(label_text, (x, y))
        
        pygame.draw.rect(
            surface,
            (220, 220, 220),
            (x + 120, y + 2, bar_width, bar_height)
        )
        
        fill_width = int(bar_width * min(value / max_value, 1.0))
        color = self._get_energy_color(value / max_value)
        pygame.draw.rect(
            surface,
            color,
            (x + 120, y + 2, fill_width, bar_height)
        )
        
        value_text = self.font_small.render(
            f"{value:.1f}/{max_value:.0f}",
            True,
            self.COLOR_TEXT
        )
        surface.blit(value_text, (x + 120 + bar_width + 10, y))
    
    def _get_energy_color(self, ratio):
        """根据能量比例返回颜色"""
        if ratio > 0.6:
            return (100, 200, 100)
        elif ratio > 0.3:
            return (255, 200, 0)
        else:
            return (255, 100, 100)
    
    def _draw_controls(self):
        """绘制控制提示"""
        controls = [
            "Space: Pause",
            "R: Reset",
            "S: Save",  # 新增
            "↑↓: Speed",
            "ESC: Quit",
            f"Speed: {self.speed}x"
        ]
        
        y = self.height - 30
        for i, text in enumerate(controls):
            rendered = self.font_small.render(text, True, (100, 100, 100))
            self.screen.blit(rendered, (10 + i * 120, y))
    
    def close(self):
        """关闭可视化器"""
        self._save_data()  # 最后保存一次数据
        pygame.quit()
        print(f"✅ All data saved to: {self.run_folder}")
    
    def _draw_sensor_rays(self, surface, agent_id, screen_x, screen_y):
        """绘制传感器射线"""
        if not hasattr(self.env, 'n_sensors'):
            return
        
        if agent_id not in self.env.agent_positions:
            return
        
        n_sensors = self.env.n_sensors
        sensor_range = self.env.sensor_range
        agent_pos = self.env.agent_positions[agent_id]
        
        is_predator = "predator" in agent_id
        
        sensor_angles = np.linspace(0, 2 * np.pi, n_sensors, endpoint=False)
        
        ray_surface = pygame.Surface((self.env_width, self.env_height), pygame.SRCALPHA)
        
        current_time = pygame.time.get_ticks()
        
        for i, angle in enumerate(sensor_angles):
            ray_result = self.env._cast_sensor_ray(
                agent_pos, 
                angle, 
                sensor_range
            )
            
            if not ray_result['hit']:
                color = (180, 180, 180, 40)
                distance = 0.3
            else:
                distance = ray_result['distance']
                object_type = ray_result['object_type']
                object_id = ray_result['object_id']
                
                color = self._get_intelligent_ray_color(
                    is_predator, 
                    object_type, 
                    object_id, 
                    distance
                )
            
            ray_length = distance * sensor_range * self.scale
            end_x = screen_x + int(np.cos(angle) * ray_length)
            end_y = screen_y + int(np.sin(angle) * ray_length)
            
            is_important = color[3] > 200
            thickness = 3 if is_important else 1
            
            pygame.draw.line(
                ray_surface,
                color,
                (screen_x, screen_y),
                (end_x, end_y),
                thickness
            )
            
            if is_important and distance < 0.95:
                pygame.draw.circle(ray_surface, color, (end_x, end_y), 5)
                
                pulse = abs(np.sin(current_time * 0.005))
                glow_radius = int(6 + 4 * pulse)
                glow_alpha = int(color[3] * pulse * 0.5)
                glow_color = (*color[:3], glow_alpha)
                pygame.draw.circle(ray_surface, glow_color, (end_x, end_y), glow_radius, 2)
                
                arrow_size = 6
                arrow_angle1 = angle + np.pi * 0.85
                arrow_angle2 = angle - np.pi * 0.85
                
                arrow_p1 = (
                    end_x + int(np.cos(arrow_angle1) * arrow_size),
                    end_y + int(np.sin(arrow_angle1) * arrow_size)
                )
                arrow_p2 = (
                    end_x + int(np.cos(arrow_angle2) * arrow_size),
                    end_y + int(np.sin(arrow_angle2) * arrow_size)
                )
                
                pygame.draw.polygon(ray_surface, color, [(end_x, end_y), arrow_p1, arrow_p2])
        
        surface.blit(ray_surface, (0, 0))
    
    def _get_intelligent_ray_color(self, is_predator, object_type, object_id, distance):
        """智能射线颜色"""
        proximity_factor = 1 - distance
        
        if is_predator:
            if object_type == 'prey':
                brightness = int(200 + 55 * proximity_factor)
                return (255, brightness, 0, 240)
            
            elif object_type == 'predator':
                return (220, 50, 50, 100)
            
            elif object_type == 'grass':
                return (100, 200, 80, 60)
            
            elif object_type == 'wall':
                alpha = int(120 + 80 * proximity_factor)
                return (100, 100, 100, alpha)
        
        else:
            if object_type == 'grass':
                grass_energy_ratio = 1.0
                if object_id and object_id in self.env.grass_energies:
                    grass_energy_ratio = self.env.grass_energies[object_id] / self.env.initial_energy_grass
                
                intensity = int(100 + 155 * grass_energy_ratio)
                alpha = int(180 + 70 * proximity_factor)
                return (50, intensity, 50, alpha)
            
            elif object_type == 'predator':
                intensity = int(200 + 55 * proximity_factor)
                alpha = int(200 + 55 * proximity_factor)
                return (intensity, 50, 50, alpha)
            
            elif object_type == 'prey':
                return (50, 120, 220, 80)
            
            elif object_type == 'wall':
                alpha = int(120 + 80 * proximity_factor)
                return (100, 100, 100, alpha)
        
        return (180, 180, 180, 80)
    
    def _draw_observation_space(self):
        """绘制观察空间"""
        obs_surface = pygame.Surface((self.obs_width, self.height))
        obs_surface.fill(self.COLOR_PANEL_BG)
        
        if self.selected_agent is None:
            hint_text = self.font_medium.render(
                "Select an agent to view observation",
                True,
                (150, 150, 150)
            )
            text_rect = hint_text.get_rect(center=(self.obs_width // 2, self.height // 2))
            obs_surface.blit(hint_text, text_rect)
        
        elif self.selected_agent not in self.env.agents:
            dead_text = self.font_medium.render(
                "Selected agent is dead",
                True,
                (255, 0, 0)
            )
            text_rect = dead_text.get_rect(center=(self.obs_width // 2, self.height // 2))
            obs_surface.blit(dead_text, text_rect)
            self.selected_agent = None
        
        else:
            try:
                # ✅ 调用方式不变，内部会根据实际维度自适应
                self.obs_visualizer.render(
                    obs_surface,
                    self.selected_agent,
                    self.env
                )
            except Exception as e:
                print(f"[ERROR] ObservationSpaceVisualizer failed for {self.selected_agent}: {e}")
                import traceback
                traceback.print_exc()
                
                error_text = self.font_medium.render(
                    "Failed to render observation",
                    True,
                    (255, 0, 0)
                )
                text_rect = error_text.get_rect(center=(self.obs_width // 2, self.height // 2))
                obs_surface.blit(error_text, text_rect)
                
                self.selected_agent = None
        
        x_offset = self.env_width + self.panel_width
        self.screen.blit(obs_surface, (x_offset, 0))
