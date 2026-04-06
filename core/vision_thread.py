import gc
import threading
import time
from typing import Dict, Any

import cv2
import torch

from utils.logger import logger
from utils.message_queue import message_queue
from utils.config_loader import config_loader
from utils.llm_gate import is_llm_busy

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
        self.sample_interval = int(
            config_loader.get_config().get("system", {}).get("vision_sample_interval", 25)
        )
        cam = config_loader.get_config().get("cameras", {}).get("camera1", {}).get(
            "resolution", [640, 480]
        )
        self._proc_w, self._proc_h = int(cam[0]), int(cam[1])

    def _init_yolo(self):
        """
        初始化YOLO模块
        """
        config = config_loader.get_config()
        yolo_config = config.get("models", {}).get("yolo", {})
        if yolo_config.get("type", "real") != "mock":
            return YoloDetector(yolo_config)
        else:
            return MockYoloDetector(yolo_config)
    
    def _init_vda(self):
        """
        初始化VDA模块
        """
        config = config_loader.get_config()
        vda_config = config.get("models", {}).get("vda", {})
        if vda_config.get("type", "real") != "mock":
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
                            if is_llm_busy():
                                continue
                            # 统一到配置分辨率再推理，避免高分辨率整帧在队列中占用大量统一内存
                            if frame.shape[1] != self._proc_w or frame.shape[0] != self._proc_h:
                                work = cv2.resize(
                                    frame,
                                    (self._proc_w, self._proc_h),
                                    interpolation=cv2.INTER_AREA,
                                )
                            else:
                                work = frame
                            yolo_results = self.yolo.inference(work)
                            depth_map = self.vda.inference(work)
                            message_queue.send_message(
                                "inference",
                                {
                                    "type": "vision_result",
                                    "frame": work,
                                    "yolo_results": yolo_results,
                                    "depth_map": depth_map,
                                    "timestamp": timestamp,
                                    "camera_id": camera_id,
                                },
                            )
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                
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