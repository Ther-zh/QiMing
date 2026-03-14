import threading
import time
from typing import Dict, Any, List, Optional
from collections import deque

from utils.logger import logger
from utils.data_formatter import DataFormatter
from utils.config_loader import config_loader
from core.risk_evaluator import RiskEvaluator

class RealtimeScheduler:
    def __init__(self):
        """
        初始化实时安全调度引擎
        """
        self.config = config_loader.get_config()
        self.risk_rules = config_loader.get_risk_rules()
        self.alert_queue = deque(maxlen=100)
        self.complex_scene_trigger = threading.Event()
        self.last_alert_time = {}  # 记录每种告警的最后触发时间
        self.consecutive_alerts = {}  # 记录连续告警的帧数
        
        # 初始化风险评价器
        self.risk_evaluator = RiskEvaluator()
        
        # 启动调度线程
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=False  # 最高优先级
        )
        self.running = False
    
    def start(self):
        """
        启动调度引擎
        """
        self.running = True
        self.scheduler_thread.start()
        logger.info("实时安全调度引擎已启动")
    
    def stop(self):
        """
        停止调度引擎
        """
        self.running = False
        if self.scheduler_thread.is_alive():
            self.scheduler_thread.join()
        logger.info("实时安全调度引擎已停止")
    
    def _scheduler_loop(self):
        """
        调度循环
        """
        while self.running:
            # 这里应该从队列中获取环境元数据
            # 暂时模拟处理
            time.sleep(0.2)  # 200ms，对应5FPS
    
    def process_metadata(self, metadata: Dict[str, Any]):
        """
        处理环境元数据，评估危险等级
        
        Args:
            metadata: 环境感知元数据
        """
        timestamp = metadata.get("timestamp")
        
        # 使用新的风险评价器
        risk_result = self.risk_evaluator.evaluate_risk(metadata)
        
        # 检查特殊场景
        special_scene_result = self.risk_evaluator.evaluate_special_scene(metadata)
        if special_scene_result:
            risk_result = special_scene_result
        
        # 评估危险等级
        alert = self._evaluate_risk_level(
            risk_result['risk_score'],
            0,  # 场景复杂度已在新评价器中考虑
            risk_result.get('target_info'),
            timestamp,
            risk_result.get('risk_level'),
            risk_result.get('message')
        )
        
        if alert:
            self.alert_queue.append(alert)
        
        # 如果风险等级较高，触发复杂场景引擎
        if risk_result['risk_score'] >= 60:
            self.complex_scene_trigger.set()
    
    def _calculate_risk_score(self, target: Dict[str, Any]) -> float:
        """
        计算单个目标的危险分
        
        Args:
            target: 目标信息
            
        Returns:
            危险分
        """
        category = target.get("category", "unknown")
        distance = target.get("distance", 10.0)
        speed = target.get("speed", 0.0)
        
        # 获取类别权重
        category_weights = self.risk_rules.get("category_weights", {})
        weight = category_weights.get(category, 5)
        
        # 获取距离系数
        distance_coef = self._get_distance_coefficient(distance)
        
        # 获取速度系数
        speed_coef = self._get_speed_coefficient(speed)
        
        # 计算危险分
        risk_score = weight * distance_coef * speed_coef
        
        return risk_score
    
    def _get_distance_coefficient(self, distance: float) -> float:
        """
        获取距离系数
        """
        distance_coefficients = self.risk_rules.get("distance_coefficients", {})
        if distance < 1:
            return distance_coefficients.get("0-1", 3.0)
        elif distance < 2:
            return distance_coefficients.get("1-2", 2.5)
        elif distance < 3:
            return distance_coefficients.get("2-3", 2.0)
        elif distance < 5:
            return distance_coefficients.get("3-5", 1.5)
        elif distance < 10:
            return distance_coefficients.get("5-10", 1.0)
        elif distance < 20:
            return distance_coefficients.get("10-20", 0.5)
        else:
            return distance_coefficients.get("20+", 0.1)
    
    def _get_speed_coefficient(self, speed: float) -> float:
        """
        获取速度系数
        """
        speed_coefficients = self.risk_rules.get("speed_coefficients", {})
        if speed < 5:
            return speed_coefficients.get("0-5", 0.5)
        elif speed < 10:
            return speed_coefficients.get("5-10", 1.0)
        elif speed < 15:
            return speed_coefficients.get("10-15", 1.5)
        elif speed < 20:
            return speed_coefficients.get("15-20", 2.0)
        elif speed < 30:
            return speed_coefficients.get("20-30", 2.5)
        else:
            return speed_coefficients.get("30+", 3.0)
    
    def _calculate_scene_complexity(self, targets: List[Dict[str, Any]]) -> float:
        """
        计算场景复杂度
        """
        complexity_items = self.risk_rules.get("complexity_items", {})
        scene_score = 0.0
        
        for target in targets:
            category = target.get("category")
            if category in complexity_items:
                scene_score += complexity_items[category]
        
        return scene_score
    
    def _evaluate_risk_level(self, risk_score: float, scene_score: float, target: Dict[str, Any], timestamp: float, level: str = None, message: str = None) -> Optional[Dict[str, Any]]:
        """
        评估危险等级并生成告警
        """
        risk_config = self.config.get("risk", {})
        min_consecutive = risk_config.get("min_consecutive_frames", 2)
        alert_cooldown = risk_config.get("alert_cooldown", 3)
        
        # 如果没有提供级别和消息，根据风险分数确定
        if not level or not message:
            if risk_score >= 80:
                level = "level1"
                message = "危险！请立即避让！"
                # 高危险级别不需要连续帧检查，立即触发
                min_consecutive = 1
            elif risk_score >= 50:
                level = "level2"
                message = "注意！前方有危险！"
            elif risk_score >= 30:
                level = "level3"
                message = "请注意前方情况"
            else:
                level = "level4"
                message = "道路安全"
                return None  # 安全状态不生成告警
        
        # 检查连续告警帧数
        self.consecutive_alerts[level] = self.consecutive_alerts.get(level, 0) + 1
        
        # 检查冷却时间
        last_time = self.last_alert_time.get(level, 0)
        if timestamp - last_time < alert_cooldown:
            return None
        
        # 只有连续达到指定帧数才触发告警
        if self.consecutive_alerts[level] >= min_consecutive:
            self.last_alert_time[level] = timestamp
            self.consecutive_alerts[level] = 0  # 重置连续计数
            
            alert = DataFormatter.format_alert(
                level=level,
                message=message,
                timestamp=timestamp,
                target_info=target
            )
            
            # 对于高危险级别，立即处理
            if level == "level1":
                logger.warning(f"[高危险] {message} - 距离: {target.get('distance', 0):.1f}m - 速度: {target.get('speed', 0):.1f}m/s")
            
            return alert
        
        return None
    
    def get_alert(self) -> Optional[Dict[str, Any]]:
        """
        获取告警信息
        
        Returns:
            告警信息字典
        """
        if self.alert_queue:
            return self.alert_queue.popleft()
        return None
    
    def reset_complex_scene_trigger(self):
        """
        重置复杂场景触发信号
        """
        self.complex_scene_trigger.clear()
    
    def is_complex_scene_triggered(self) -> bool:
        """
        检查是否触发了复杂场景
        
        Returns:
            是否触发
        """
        return self.complex_scene_trigger.is_set()
