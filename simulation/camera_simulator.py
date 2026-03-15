import cv2
import numpy as np
import threading
import time
import os
import soundfile as sf
import tempfile
from typing import Dict, Any, List, Tuple
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
        self.audio_buffers = {}
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
        # 初始化帧缓冲区和音频缓冲区
        buffer_size = self.config.get("system", {}).get("max_frame_buffer", 3)
        self.frame_buffers[camera_id] = deque(maxlen=buffer_size)
        self.audio_buffers[camera_id] = deque(maxlen=buffer_size)
        
        # 获取视频路径
        video_path = self.config.get("simulation", {}).get("video_paths", {}).get(camera_id)
        
        if video_path and os.path.exists(video_path):
            # 使用视频文件模拟
            cap = cv2.VideoCapture(video_path)
            self.cameras[camera_id] = cap
            logger.info(f"摄像头 {camera_id} 使用视频文件: {video_path}")
            
            # 提取音频
            audio_data = self._extract_audio(video_path)
        else:
            # 使用随机生成的图像模拟
            cap = None
            audio_data = None
            logger.info(f"摄像头 {camera_id} 使用随机图像模拟")
        
        try:
            frame_count = 0
            while self.running:
                timestamp = time.time()
                
                if cap and cap.isOpened():
                    # 从视频文件读取帧
                    ret, frame = cap.read()
                    if not ret:
                        # 视频播放完毕，停止模拟
                        logger.info(f"摄像头 {camera_id} 视频播放完毕")
                        # 释放摄像头资源
                        cap.release()
                        # 从摄像头字典中移除
                        if camera_id in self.cameras:
                            del self.cameras[camera_id]
                        break
                    
                    # 调整帧大小
                    width, height = camera_config.get("resolution", [640, 480])
                    frame = cv2.resize(frame, (width, height))
                    
                    # 提取对应帧的音频
                    audio_frame = None
                    if audio_data is not None:
                        # 简单的音频帧提取，实际应用中需要更精确的同步
                        sample_rate = 16000  # 假设音频采样率为16kHz
                        # 使用模拟帧率而不是视频实际帧率，确保每帧有足够的音频数据
                        sim_fps = camera_config.get("fps", 5)
                        samples_per_frame = int(sample_rate / sim_fps)
                        start_sample = frame_count * samples_per_frame
                        end_sample = start_sample + samples_per_frame
                        if start_sample < len(audio_data):
                            audio_frame = audio_data[start_sample:end_sample]
                            if len(audio_frame) < samples_per_frame:
                                # 填充静音
                                audio_frame = np.pad(audio_frame, (0, samples_per_frame - len(audio_frame)))
                    
                    frame_count += 1
                else:
                    # 生成随机图像
                    width, height = camera_config.get("resolution", [640, 480])
                    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                    
                    # 添加模拟内容
                    cv2.putText(frame, f"Camera: {camera_id}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Time: {timestamp:.2f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 生成随机音频
                    audio_frame = np.random.randn(16000 // 5)  # 5FPS，每帧200ms音频
                
                # 添加到缓冲区
                self.frame_buffers[camera_id].append((frame, timestamp))
                if audio_frame is not None:
                    self.audio_buffers[camera_id].append((audio_frame, timestamp))
                
                # 控制帧率
                fps = camera_config.get("fps", 5)
                time.sleep(1 / fps)
        finally:
            if cap:
                cap.release()
    
    def _extract_audio(self, video_path: str) -> np.ndarray:
        """
        从视频文件中提取音频
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            音频数据（numpy数组）
        """
        try:
            # 使用ffmpeg从视频中提取音频
            import subprocess
            import tempfile
            
            # 创建临时音频文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # 使用ffmpeg提取音频
            cmd = [
                'ffmpeg', '-i', video_path, '-ac', '1', '-ar', '16000', 
                '-f', 'wav', '-y', temp_audio_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # 读取音频文件
            import soundfile as sf
            audio_data, sample_rate = sf.read(temp_audio_path)
            
            # 转换为float32格式
            audio_data = audio_data.astype(np.float32)
            
            logger.info(f"从视频中提取音频数据，长度: {len(audio_data)} 样本，采样率: {sample_rate}Hz")
            
            # 清理临时文件
            import os
            os.unlink(temp_audio_path)
            
            return audio_data
        except Exception as e:
            logger.error(f"提取音频时出错: {e}")
            # 如果提取失败，返回空数组
            return np.array([])
    
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
    
    def is_video_ended(self, camera_id: str) -> bool:
        """
        检查摄像头的视频是否播放完毕
        
        Args:
            camera_id: 摄像头ID
            
        Returns:
            bool: 视频是否播放完毕
        """
        # 检查摄像头是否存在且已关闭
        if camera_id in self.cameras:
            cap = self.cameras[camera_id]
            if cap:
                # 只有当摄像头已打开且现在关闭时，才认为视频结束
                if not cap.isOpened():
                    # 检查缓冲区是否也为空
                    if camera_id in self.frame_buffers and not self.frame_buffers[camera_id]:
                        return True
        # 摄像头未初始化或仍在运行
        return False
    
    def get_audio(self, camera_id: str) -> tuple:
        """
        获取摄像头的最新音频
        
        Args:
            camera_id: 摄像头ID
            
        Returns:
            (audio_data, timestamp) 或 (None, None)
        """
        if camera_id in self.audio_buffers and self.audio_buffers[camera_id]:
            return self.audio_buffers[camera_id][-1]
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
    
    def get_all_audio(self) -> Dict[str, tuple]:
        """
        获取所有摄像头的最新音频
        
        Returns:
            摄像头ID到(audio_data, timestamp)的映射
        """
        audio = {}
        for camera_id in self.audio_buffers:
            audio_data, timestamp = self.get_audio(camera_id)
            audio[camera_id] = (audio_data, timestamp)
        return audio


