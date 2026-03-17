import cv2
import numpy as np
import threading
import time
import os
from typing import Dict, Any, Tuple
from collections import deque

from hardware.input_device import InputDevice
from utils.logger import logger

class SimulatedInputDevice(InputDevice):
    """
    模拟输入设备实现
    """
    
    def __init__(self):
        super().__init__()
        self.cameras = {}
        self.frame_buffers = {}
        self.audio_buffers = {}
        self.threads = {}
    
    def start(self):
        """
        启动设备模拟
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
        停止设备模拟
        """
        self.running = False
        
        # 等待所有线程结束
        for device_id, thread in self.threads.items():
            if thread.is_alive():
                thread.join()
            logger.info(f"设备 {device_id} 模拟已停止")
        
        # 释放资源
        for camera_id, cap in self.cameras.items():
            if cap:
                cap.release()
    
    def _simulate_camera(self, camera_id: str, camera_config: Dict[str, Any]):
        """
        模拟单个摄像头
        """
        # 初始化帧缓冲区和音频缓冲区
        buffer_size = self.config.get("system", {}).get("max_frame_buffer", 3)
        self.frame_buffers[camera_id] = deque(maxlen=buffer_size)
        # 增大音频缓冲区，让我们能累积完整的音频！
        self.audio_buffers[camera_id] = deque(maxlen=1000)
        
        # 获取视频路径
        video_path = self.config.get("simulation", {}).get("video_paths", {}).get(camera_id)
        
        # 视频帧率
        video_fps = 5  # 默认帧率
        
        # 优先使用配置文件中的音频路径
        audio_data = None
        audio_path = self.config.get("simulation", {}).get("audio_path")
        if audio_path and os.path.exists(audio_path):
            logger.info(f"使用配置文件中的音频文件: {audio_path}")
            audio_data = self._extract_audio(audio_path)
        
        if video_path and os.path.exists(video_path):
            # 使用视频文件模拟
            cap = cv2.VideoCapture(video_path)
            # 获取视频实际帧率
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                video_fps = 30  # 如果获取失败，使用默认值30
            self.cameras[camera_id] = cap
            logger.info(f"摄像头 {camera_id} 使用视频文件: {video_path}，帧率: {video_fps:.2f}")
            
            # 如果没有配置音频，从视频中提取
            if audio_data is None:
                audio_data = self._extract_audio(video_path)
        else:
            # 使用随机生成的图像模拟
            cap = None
            if audio_data is None:
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
                        # 使用视频实际帧率
                        samples_per_frame = int(sample_rate / video_fps)
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
                    audio_frame = np.random.randn(int(16000 / video_fps))  # 根据帧率生成音频
                
                # 添加到缓冲区
                self.frame_buffers[camera_id].append((frame, timestamp))
                if audio_frame is not None:
                    self.audio_buffers[camera_id].append((audio_frame, timestamp))
                
                # 控制帧率
                time.sleep(1 / video_fps)
        finally:
            if cap:
                cap.release()
    
    def _extract_audio(self, video_path: str) -> np.ndarray:
        """
        从视频文件中提取音频
        """
        try:
            # 使用ffmpeg从视频中提取音频
            import subprocess
            import tempfile
            import soundfile as sf
            
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
    
    def get_frame(self, device_id: str) -> Tuple[Any, float]:
        """
        获取设备的最新帧
        """
        if device_id in self.frame_buffers and self.frame_buffers[device_id]:
            return self.frame_buffers[device_id][-1]
        return None, None
    
    def get_all_frames(self) -> Dict[str, Tuple[Any, float]]:
        """
        获取所有设备的最新帧
        """
        frames = {}
        for device_id in self.frame_buffers:
            frame, timestamp = self.get_frame(device_id)
            frames[device_id] = (frame, timestamp)
        return frames
    
    def get_audio(self, device_id: str) -> Tuple[Any, float]:
        """
        获取设备的最新单帧音频（恢复原来的行为，让输入线程累积）
        """
        if device_id in self.audio_buffers and self.audio_buffers[device_id]:
            return self.audio_buffers[device_id].popleft()
        return None, None
    
    def get_all_audio(self) -> Dict[str, Tuple[Any, float]]:
        """
        获取所有设备的最新音频
        """
        audio = {}
        for device_id in self.audio_buffers:
            audio_data, timestamp = self.get_audio(device_id)
            audio[device_id] = (audio_data, timestamp)
        return audio
    
    def is_ended(self, device_id: str) -> bool:
        """
        检查设备是否结束（如视频播放完毕）
        """
        # 检查设备是否已经开始处理帧（如果还没有开始处理，说明还在初始化）
        if device_id not in self.frame_buffers or not self.frame_buffers[device_id]:
            return False
        
        # 检查摄像头是否不在字典中（已被移除，说明视频已处理完毕）
        if device_id not in self.cameras:
            # 视频文件已经处理完毕
            logger.info(f"设备 {device_id} 已从摄像头字典中移除，视频播放完毕")
            return True
        # 检查摄像头是否存在且已关闭
        elif device_id in self.cameras:
            cap = self.cameras[device_id]
            if cap:
                # 只有当摄像头已打开且现在关闭时，才认为视频结束
                if not cap.isOpened():
                    logger.info(f"设备 {device_id} 摄像头已关闭，视频播放完毕")
                    return True
        # 摄像头未初始化或仍在运行
        return False
