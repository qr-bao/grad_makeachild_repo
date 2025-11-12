"""
观察空间可视化模块（增强版）
动态适配 n_sensors*12 + self_state_dim 维结构化观测
"""
import pygame
import numpy as np
from typing import Optional

class ObservationSpaceVisualizer:
    """观察空间可视化器（动态维度）"""
    
    # 颜色定义
    COLOR_BG = (255, 255, 255)
    COLOR_TEXT = (40, 40, 40)
    COLOR_HEADER = (80, 80, 150)
    COLOR_BORDER = (200, 200, 200)
    
    # 层级颜色
    COLOR_ENV = (100, 200, 100)      # 环境层-绿色
    COLOR_PREDATOR = (220, 50, 50)   # Predator层-红色
    COLOR_PREY = (50, 120, 220)      # Prey层-蓝色
    COLOR_MATE = (255, 150, 50)      # 配偶层-橙色
    COLOR_SELF = (150, 100, 200)     # 自身-紫色
    
    def __init__(self, width=450, height=800):
        self.width = width
        self.height = height
        self.font_small = pygame.font.Font(None, 18)
        self.font_medium = pygame.font.Font(None, 22)
        self.font_large = pygame.font.Font(None, 26)
    
    def render(self, surface, agent_id, env):
        """渲染观察空间"""
        try:
            # 需要连续空间并具备传感器定义
            n_sensors = getattr(env, "n_sensors", None)
            if n_sensors is None:
                self._draw_error(surface, "Environment missing sensor configuration")
                return
            self_state_dim = getattr(env, "self_state_dim", 10)
            expected_dim = n_sensors * 12 + self_state_dim

            # 获取观察
            observation = env._get_observation(agent_id)
            
            if observation is None or len(observation) != expected_dim:
                dims = len(observation) if observation is not None else 0
                self._draw_error(surface, f"Invalid observation: {dims} dims (expected {expected_dim})")
                return
            
            # 清空背景
            surface.fill(self.COLOR_BG)
            
            # 绘制标题
            title = self.font_large.render(f"Observation Space ({expected_dim}D)", True, self.COLOR_HEADER)
            surface.blit(title, (10, 10))
            
            agent_label = self.font_medium.render(f"Agent: {agent_id}", True, self.COLOR_TEXT)
            surface.blit(agent_label, (10, 40))
            
            y = 70
            
            layer_span = n_sensors * 3
            
            # === 绘制各层 ===
            y = self._draw_layer_radar(surface, "Environment", observation[0:layer_span], 
                                        self.COLOR_ENV, 10, y, n_sensors)
            y += 20
            
            y = self._draw_layer_radar(surface, "Predators", observation[layer_span:layer_span*2], 
                                        self.COLOR_PREDATOR, 10, y, n_sensors)
            y += 20
            
            y = self._draw_layer_radar(surface, "Prey", observation[layer_span*2:layer_span*3], 
                                        self.COLOR_PREY, 10, y, n_sensors)
            y += 20
            
            y = self._draw_layer_radar(surface, "Mates", observation[layer_span*3:layer_span*4], 
                                        self.COLOR_MATE, 10, y, n_sensors)
            y += 20
            
            # === 自身状态（详细显示）===
            self._draw_self_state(surface, observation[layer_span*4:layer_span*4 + self_state_dim], 10, y)
            
        except Exception as e:
            self._draw_error(surface, f"Render error: {str(e)}")
    
    def _draw_layer_radar(self, surface, title, layer_data, color, x, y, n_sensors):
        """绘制一个层的雷达图（简化版）"""
        # 标题
        header = self.font_medium.render(title, True, color)
        surface.blit(header, (x, y))
        y += 25
        
        # 分解层数据
        distances = layer_data[0:n_sensors]
        values1 = layer_data[n_sensors:n_sensors*2]
        values2 = layer_data[n_sensors*2:n_sensors*3]
        
        # 统计信息
        hits = np.sum(distances < 1.0)
        if hits > 0:
            avg_dist = np.mean(distances[distances < 1.0])
            min_dist = np.min(distances[distances < 1.0])
        else:
            avg_dist = 1.0
            min_dist = 1.0
        
        # 显示统计
        stats_text = self.font_small.render(
            f"Detected: {hits}/{n_sensors} | Min: {min_dist:.2f} | Avg: {avg_dist:.2f}",
            True, self.COLOR_TEXT
        )
        surface.blit(stats_text, (x + 20, y))
        y += 20
        
        # 简化雷达图
        radar_size = 80
        radar_center_x = x + 30 + radar_size // 2
        radar_center_y = y + radar_size // 2
        
        # 绘制圆环
        pygame.draw.circle(surface, self.COLOR_BORDER, 
                          (radar_center_x, radar_center_y), radar_size // 2, 1)
        pygame.draw.circle(surface, self.COLOR_BORDER, 
                          (radar_center_x, radar_center_y), radar_size // 4, 1)
        
        # 绘制射线检测结果
        for i in range(n_sensors):
            if distances[i] < 1.0:
                angle = (i / n_sensors) * 2 * np.pi
                radius = (1 - distances[i]) * (radar_size // 2)
                
                end_x = radar_center_x + int(np.cos(angle) * radius)
                end_y = radar_center_y + int(np.sin(angle) * radius)
                
                # 根据第二个值（速度/能量）调整颜色强度
                intensity = int(abs(values1[i]) * 255) if values1[i] != 0 else 100
                ray_color = (*color[:2], intensity)
                
                pygame.draw.line(surface, ray_color, 
                               (radar_center_x, radar_center_y), 
                               (end_x, end_y), 2)
        
        # 右侧详细数值（前5个最近的）
        detail_x = x + 150
        detail_y = y
        
        # 找到最近的5个目标
        close_indices = np.argsort(distances)[:5]
        close_indices = [i for i in close_indices if distances[i] < 1.0]
        
        if close_indices:
            detail_header = self.font_small.render("Closest:", True, color)
            surface.blit(detail_header, (detail_x, detail_y))
            detail_y += 18
            
            for idx in close_indices[:3]:  # 只显示前3个
                angle_deg = int((idx / n_sensors) * 360)
                dist = distances[idx]
                val1 = values1[idx]
                val2 = values2[idx]
                
                text = self.font_small.render(
                    f"#{idx:02d}({angle_deg}°): d={dist:.2f}",
                    True, self.COLOR_TEXT
                )
                surface.blit(text, (detail_x, detail_y))
                detail_y += 16
        
        return y + radar_size + 10
    
    def _draw_self_state(self, surface, state, x, y):
        """绘制自身状态（动态维度）"""
        # 标题
        header = self.font_medium.render("Self State", True, self.COLOR_SELF)
        surface.blit(header, (x, y))
        y += 30
        
        # 解析状态
        energy_pct = state[2] * 100
        hunger_pct = state[3] * 100
        labels = [
            f"Type: {'Predator' if state[0] > 0.5 else 'Prey'}",
            f"Population: {int(state[1] * 10)}",  # 假设最多10个种群
            f"Energy: {energy_pct:.1f}%",
            f"Hunger: {hunger_pct:.1f}%",
            f"Just Ate: {'Yes' if state[4] > 0.5 else 'No'}",
            f"Can Mate: {'Yes' if state[5] > 0.5 else 'No'}",
            f"Speed: {state[6]*200:.1f}px/s",  # 反归一化
            f"Direction: {state[7]*180:.0f}°",
            f"Age: {state[8]*100:.0f}%",
            f"Cooldown: {state[9]*100:.0f}%",
        ]
        if len(state) >= 11:
            delta_pct = state[10] * 100
            labels.append(f"Δ Energy: {delta_pct:.1f}%")
        if len(state) >= 12:
            labels.append(f"Grass Density: {state[11]*100:.0f}%")
        
        # 绘制进度条
        for i, label in enumerate(labels):
            # 标签
            text = self.font_small.render(label, True, self.COLOR_TEXT)
            surface.blit(text, (x + 10, y))
            
            # 对于百分比值，绘制进度条
            if '%' in label and i >= 2:
                value = state[i]
                bar_x = x + 200
                bar_y = y + 5
                bar_w = 150
                bar_h = 10
                
                # 背景
                pygame.draw.rect(surface, self.COLOR_BORDER, 
                               (bar_x, bar_y, bar_w, bar_h))
                
                # 填充
                fill_w = int(bar_w * value)
                bar_color = self._get_bar_color(value, i)
                pygame.draw.rect(surface, bar_color, 
                               (bar_x, bar_y, fill_w, bar_h))
            
            y += 20
    
    def _get_bar_color(self, value, index):
        """根据值和类型返回进度条颜色"""
        if index == 2:  # Energy
            if value > 0.6:
                return (100, 200, 100)
            elif value > 0.3:
                return (255, 200, 0)
            else:
                return (255, 100, 100)
        elif index == 3:  # Hunger
            if value < 0.3:
                return (100, 200, 100)
            elif value < 0.7:
                return (255, 200, 0)
            else:
                return (255, 100, 100)
        else:
            return (100, 150, 200)
    
    def _draw_error(self, surface, message):
        """绘制错误信息"""
        surface.fill(self.COLOR_BG)
        error_text = self.font_medium.render("Error", True, (255, 0, 0))
        surface.blit(error_text, (10, 10))
        
        msg_text = self.font_small.render(message, True, self.COLOR_TEXT)
        surface.blit(msg_text, (10, 40))
