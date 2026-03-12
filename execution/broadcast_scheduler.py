import threading
import time
from typing import Dict, Any, Optional
import heapq

from utils.logger import logger
from utils.config_loader import config_loader

class BroadcastScheduler:
    def __init__(self):
        """
        初始化语音播报调度器
        """
        self.config = config_loader.get_config()
        self.queue = []  # 优先级队列
        self.lock = threading.Lock()
        self.last_alert_time = {}  # 记录每种告警的最后播报时间
        self.running = False
        
        # 启动调度线程
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True
        )
    
    def start(self):
        """
        启动语音播报调度器
        """
        self.running = True
        self.scheduler_thread.start()
        logger.info("语音播报调度器已启动")
    
    def stop(self):
        """
        停止语音播报调度器
        """
        self.running = False
        if self.scheduler_thread.is_alive():
            self.scheduler_thread.join()
        logger.info("语音播报调度器已停止")
    
    def add_message(self, message: str, priority: int = 3, alert_type: str = "normal"):
        """
        添加播报消息
        
        Args:
            message: 播报消息
            priority: 优先级（1-4，1最高）
            alert_type: 告警类型
        """
        with self.lock:
            # 检查冷却时间
            current_time = time.time()
            if alert_type in self.last_alert_time:
                cooldown = self.config.get("risk", {}).get("alert_cooldown", 3)
                if current_time - self.last_alert_time[alert_type] < cooldown:
                    logger.debug(f"告警类型 {alert_type} 处于冷却期，跳过播报")
                    return
            
            # 添加到优先级队列
            # 使用负数作为优先级，因为heapq是最小堆
            heapq.heappush(self.queue, (-priority, current_time, message, alert_type))
            logger.debug(f"添加播报消息: {message}, 优先级: {priority}")
    
    def _scheduler_loop(self):
        """
        调度循环
        """
        while self.running:
            # 检查队列是否有消息
            with self.lock:
                if self.queue:
                    # 获取最高优先级的消息
                    priority, timestamp, message, alert_type = heapq.heappop(self.queue)
                    
                    # 更新最后播报时间
                    self.last_alert_time[alert_type] = time.time()
                    
                    # 播报消息
                    logger.info(f"播报: {message}")
                    # 这里应该调用TTS引擎
                    # self.tts_engine.speak(message)
            
            time.sleep(0.1)
    
    def clear_queue(self):
        """
        清空队列
        """
        with self.lock:
            self.queue.clear()
            logger.info("播报队列已清空")
    
    def get_queue_size(self) -> int:
        """
        获取队列大小
        
        Returns:
            队列大小
        """
        with self.lock:
            return len(self.queue)
