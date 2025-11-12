# rampage_agent.py
"""横冲直撞测试Agent - 全速前进"""
import numpy as np

class RampageAgent:
    """全速横冲直撞，测试环境速度"""
    
    def __init__(self, env):
        self.env = env
        self.agent_directions = {}  # 记录每个agent的方向
        
    def get_action(self, agent_id):
        """全速随机方向前进"""
        # 初始化方向
        if agent_id not in self.agent_directions:
            self.agent_directions[agent_id] = np.random.uniform(0, 2 * np.pi)
        
        # 每50步随机改变方向
        if np.random.random() < 0.02:  # 2%概率改变方向
            self.agent_directions[agent_id] = np.random.uniform(0, 2 * np.pi)
        
        angle = self.agent_directions[agent_id]
        
        # 全速前进！
        thrust_x = np.cos(angle) * 0.5  # 最大推力
        thrust_y = np.sin(angle) * 0.5
        
        return np.array([thrust_x, thrust_y], dtype=np.float32)