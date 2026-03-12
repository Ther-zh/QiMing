import threading
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
from collections import deque

from utils.logger import logger
from utils.config_loader import config_loader

class FrameSync:
    def __init__(self):
        """
        初始化帧同步模块
        """
        self.config = config_loader.get_config()
        self.yolo_results = deque(maxlen=5)  # 存储YOLO结果
        self.vda_results = deque(maxlen=5)   # 存储VDA结果
        self.frame_buffer = deque(maxlen=5)   # 存储原始帧
        self.lock = threading.Lock()
        self.running = False
    
    def start(self):
        """
        启动帧同步
        """
        self.running = True
        logger.info("帧同步模块已启动")
    
    def stop(self):
        """
        停止帧同步
        """
        self.running = False
        logger.info("帧同步模块已停止")
    
    def add_frame(self, frame: np.ndarray, timestamp: float, camera_id: int):
        """
        添加新帧
        
        Args:
            frame: 原始图像
            timestamp: 时间戳
            camera_id: 摄像头ID
        """
        with self.lock:
            self.frame_buffer.append((frame, timestamp, camera_id))
    
    def add_yolo_result(self, result: Dict[str, Any], timestamp: float):
        """
        添加YOLO检测结果
        
        Args:
            result: YOLO检测结果
            timestamp: 时间戳
        """
        with self.lock:
            self.yolo_results.append((result, timestamp))
    
    def add_vda_result(self, depth_map: np.ndarray, timestamp: float):
        """
        添加VDA深度估计结果
        
        Args:
            depth_map: 深度图
            timestamp: 时间戳
        """
        with self.lock:
            self.vda_results.append((depth_map, timestamp))
    
    def get_sync_data(self) -> Optional[Tuple[np.ndarray, Dict[str, Any], np.ndarray, float, int]]:
        """
        获取同步的数据
        
        Returns:
            (frame, yolo_result, depth_map, timestamp, camera_id) 或 None
        """
        with self.lock:
            if not self.frame_buffer or not self.yolo_results or not self.vda_results:
                return None
            
            # 寻找时间戳最接近的帧、YOLO结果和VDA结果
            best_match = None
            min_time_diff = float('inf')
            
            for frame, frame_ts, camera_id in self.frame_buffer:
                for yolo_result, yolo_ts in self.yolo_results:
                    for depth_map, vda_ts in self.vda_results:
                        # 计算时间差
                        time_diff = abs(frame_ts - yolo_ts) + abs(frame_ts - vda_ts)
                        
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            best_match = (frame, yolo_result, depth_map, frame_ts, camera_id)
            
            # 如果找到匹配且时间差在可接受范围内
            if best_match and min_time_diff < 0.1:  # 100ms内视为同步
                # 从缓冲区中移除已使用的数据
                # 注意：这里简化处理，实际应该更精确地移除对应的数据
                return best_match
            
            return None
    
    def clear_buffers(self):
        """
        清空缓冲区
        """
        with self.lock:
            self.frame_buffer.clear()
            self.yolo_results.clear()
            self.vda_results.clear()
            logger.info("帧同步缓冲区已清空")
