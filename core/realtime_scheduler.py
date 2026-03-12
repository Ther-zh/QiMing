import threading
import time
from typing import Dict, Any, List, Optional
from collections import deque

from utils.logger import logger
from utils.data_formatter import DataFormatter
from utils.config_loader import config_loader

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
        targets = metadata.get("targets", [])
        
        # 计算全局危险分
        max_risk_score = 0.0
        high_risk_target = None
        
        for target in targets:
            risk_score = self._calculate_risk_score(target)
            if risk_score > max_risk_score:
                max_risk_score = risk_score
                high_risk_target = target
        
        # 计算场景复杂度分
        scene_score = self._calculate_scene_complexity(targets)
        
        # 评估危险等级
        alert = self._evaluate_risk_level(max_risk_score, scene_score, high_risk_target, timestamp)
        
        if alert:
            self.alert_queue.append(alert)
        
        # 如果场景复杂度超过阈值，触发复杂场景引擎
        if scene_score >= self.config.get("risk", {}).get("complexity_threshold", 60):
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
    
    def _evaluate_risk_level(self, risk_score: float, scene_score: float, target: Dict[str, Any], timestamp: float) -> Optional[Dict[str, Any]]:
        """
        评估危险等级并生成告警
        """
        risk_config = self.config.get("risk", {})
        high_threshold = risk_config.get("high_risk_threshold", 80)
        medium_threshold = risk_config.get("medium_risk_threshold", 50)
        min_consecutive = risk_config.get("min_consecutive_frames", 2)
        alert_cooldown = risk_config.get("alert_cooldown", 3)
        
        # 确定告警级别
        if risk_score >= high_threshold:
            level = "level1"
            message = "危险！请立即避让！"
        elif risk_score >= medium_threshold:
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
            
            return DataFormatter.format_alert(
                level=level,
                message=message,
                timestamp=timestamp,
                target_info=target
            )
        
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
