import threading
import time
from typing import Dict, Any

from utils.logger import logger
from utils.message_queue import message_queue
from utils.config_loader import config_loader

# 导入视觉处理模块
from perception.yolo.yolo_detector import YoloDetector
from perception.yolo.mock_yolo import MockYoloDetector
from perception.vda.vda_depth import VDADepthEstimator
from perception.vda.mock_vda import MockVDADepthEstimator

class VisionThread(threading.Thread):
    """
    视觉处理线程，负责YOLO检测和VDA深度估计
    """
    
    def __init__(self):
        super().__init__(daemon=True)
        self.running = False
        self.yolo = self._init_yolo()
        self.vda = self._init_vda()
        self.frame_count = 0
        self.sample_interval = 25  # 每25帧采样一次
    
    def _init_yolo(self):
        """
        初始化YOLO模块
        """
        config = config_loader.get_config()
        yolo_config = config.get("models", {}).get("yolo", {})
        if yolo_config.get("type", "mock") == "real":
            return YoloDetector(yolo_config)
        else:
            return MockYoloDetector(yolo_config)
    
    def _init_vda(self):
        """
        初始化VDA模块
        """
        config = config_loader.get_config()
        vda_config = config.get("models", {}).get("vda", {})
        if vda_config.get("type", "mock") == "real":
            return VDADepthEstimator(vda_config)
        else:
            return MockVDADepthEstimator(vda_config)
    
    def run(self):
        """
        线程运行方法
        """
        self.running = True
        logger.info("视觉处理线程已启动")
        
        try:
            while self.running:
                # 从视觉队列接收消息
                message = message_queue.receive_message("vision", block=False)
                if message and message.get("type") == "frame":
                    frame = message.get("frame")
                    timestamp = message.get("timestamp")
                    camera_id = message.get("camera_id")
                    
                    if frame is not None:
                        self.frame_count += 1
                        
                        # 帧采样，每25帧处理一次
                        if self.frame_count % self.sample_interval == 0:
                            # 执行YOLO检测
                            yolo_results = self.yolo.inference(frame)
                            
                            # 执行VDA深度估计
                            depth_map = self.vda.inference(frame)
                            
                            # 发送视觉处理结果到推理决策队列
                            message_queue.send_message("inference", {
                                "type": "vision_result",
                                "frame": frame,
                                "yolo_results": yolo_results,
                                "depth_map": depth_map,
                                "timestamp": timestamp,
                                "camera_id": camera_id
                            })
                
                # 短暂休眠，避免占用过多CPU
                time.sleep(0.01)
        finally:
            # 释放资源
            if hasattr(self, 'yolo'):
                self.yolo.release()
            if hasattr(self, 'vda'):
                self.vda.release()
            logger.info("视觉处理线程已停止")
    
    def stop(self):
        """
        停止线程
        """
        self.running = False