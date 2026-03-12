import cv2
import numpy as np
import threading
import time
import os
from typing import Dict, Any, List
from collections import deque

from utils.logger import logger
from utils.config_loader import config_loader

class CameraSimulator:
    def __init__(self):
        """
        初始化摄像头模拟器
        """
        self.config = config_loader.get_config()
        self.cameras = {}
        self.frame_buffers = {}
        self.running = False
        self.threads = {}
    
    def start(self):
        """
        启动摄像头模拟
        """
        self.running = True
        
        # 启动每个摄像头的模拟线程
        for camera_id, camera_config in self.config.get("cameras", {}).items():
            thread = threading.Thread(
                target=self._simulate_camera,
                args=(camera_id, camera_config),
                daemon=True
            )
            self.threads[camera_id] = thread
            thread.start()
            logger.info(f"摄像头 {camera_id} 模拟已启动")
    
    def stop(self):
        """
        停止摄像头模拟
        """
        self.running = False
        
        # 等待所有线程结束
        for camera_id, thread in self.threads.items():
            if thread.is_alive():
                thread.join()
            logger.info(f"摄像头 {camera_id} 模拟已停止")
        
        # 释放资源
        for camera_id, cap in self.cameras.items():
            if cap:
                cap.release()
    
    def _simulate_camera(self, camera_id: str, camera_config: Dict[str, Any]):
        """
        模拟单个摄像头
        
        Args:
            camera_id: 摄像头ID
            camera_config: 摄像头配置
        """
        # 初始化帧缓冲区
        buffer_size = self.config.get("system", {}).get("max_frame_buffer", 3)
        self.frame_buffers[camera_id] = deque(maxlen=buffer_size)
        
        # 获取视频路径
        video_path = self.config.get("simulation", {}).get("video_paths", {}).get(camera_id)
        
        if video_path and os.path.exists(video_path):
            # 使用视频文件模拟
            cap = cv2.VideoCapture(video_path)
            self.cameras[camera_id] = cap
            logger.info(f"摄像头 {camera_id} 使用视频文件: {video_path}")
        else:
            # 使用随机生成的图像模拟
            cap = None
            logger.info(f"摄像头 {camera_id} 使用随机图像模拟")
        
        try:
            while self.running:
                timestamp = time.time()
                
                if cap and cap.isOpened():
                    # 从视频文件读取帧
                    ret, frame = cap.read()
                    if not ret:
                        # 视频播放完毕，重新开始
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    
                    # 调整帧大小
                    width, height = camera_config.get("resolution", [640, 480])
                    frame = cv2.resize(frame, (width, height))
                else:
                    # 生成随机图像
                    width, height = camera_config.get("resolution", [640, 480])
                    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                    
                    # 添加模拟内容
                    cv2.putText(frame, f"Camera: {camera_id}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Time: {timestamp:.2f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 添加到帧缓冲区
                self.frame_buffers[camera_id].append((frame, timestamp))
                
                # 控制帧率
                fps = camera_config.get("fps", 5)
                time.sleep(1 / fps)
        finally:
            if cap:
                cap.release()
    
    def get_frame(self, camera_id: str) -> tuple:
        """
        获取摄像头的最新帧
        
        Args:
            camera_id: 摄像头ID
            
        Returns:
            (frame, timestamp) 或 (None, None)
        """
        if camera_id in self.frame_buffers and self.frame_buffers[camera_id]:
            return self.frame_buffers[camera_id][-1]
        return None, None
    
    def get_all_frames(self) -> Dict[str, tuple]:
        """
        获取所有摄像头的最新帧
        
        Returns:
            摄像头ID到(frame, timestamp)的映射
        """
        frames = {}
        for camera_id in self.frame_buffers:
            frame, timestamp = self.get_frame(camera_id)
            frames[camera_id] = (frame, timestamp)
        return frames


