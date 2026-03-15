import cv2
import numpy as np
import threading
import time
from typing import Dict, Any, Tuple
from collections import deque

# 尝试导入pyaudio，失败则设置为None
try:
    import pyaudio
    pyaudio_available = True
except ImportError:
    pyaudio = None
    pyaudio_available = False

from hardware.input_device import InputDevice
from utils.logger import logger

class RealInputDevice(InputDevice):
    """
    真实输入设备实现
    """
    
    def __init__(self):
        super().__init__()
        self.cameras = {}
        self.microphones = {}
        self.frame_buffers = {}
        self.audio_buffers = {}
        self.threads = {}
    
    def start(self):
        """
        启动设备
        """
        self.running = True
        
        # 启动每个摄像头
        for camera_id, camera_config in self.config.get("cameras", {}).items():
            thread = threading.Thread(
                target=self._capture_camera,
                args=(camera_id, camera_config),
                daemon=True
            )
            self.threads[camera_id] = thread
            thread.start()
            logger.info(f"摄像头 {camera_id} 已启动，设备ID: {camera_config.get('device_id', 0)}")
        
        # 启动每个麦克风
        for mic_id, mic_config in self.config.get("microphones", {}).items():
            thread = threading.Thread(
                target=self._capture_audio,
                args=(mic_id, mic_config),
                daemon=True
            )
            self.threads[mic_id] = thread
            thread.start()
            logger.info(f"麦克风 {mic_id} 已启动")
    
    def stop(self):
        """
        停止设备
        """
        self.running = False
        
        # 等待所有线程结束
        for device_id, thread in self.threads.items():
            if thread.is_alive():
                thread.join()
            logger.info(f"设备 {device_id} 已停止")
        
        # 释放摄像头资源
        for camera_id, cap in self.cameras.items():
            if cap:
                cap.release()
        
        # 释放麦克风资源
        for mic_id, stream in self.microphones.items():
            if stream:
                stream.stop_stream()
                stream.close()
        
        # 清理PyAudio资源
        if pyaudio_available and hasattr(self, 'p'):
            try:
                self.p.terminate()
            except Exception as e:
                logger.error(f"终止PyAudio失败: {e}")
    
    def _capture_camera(self, camera_id: str, camera_config: Dict[str, Any]):
        """
        捕获摄像头视频
        """
        # 初始化帧缓冲区
        buffer_size = self.config.get("system", {}).get("max_frame_buffer", 3)
        self.frame_buffers[camera_id] = deque(maxlen=buffer_size)
        
        # 打开摄像头
        device_id = camera_config.get("device_id", 0)
        cap = cv2.VideoCapture(device_id)
        
        if not cap.isOpened():
            logger.error(f"无法打开摄像头 {camera_id}，设备ID: {device_id}")
            return
        
        self.cameras[camera_id] = cap
        
        # 设置摄像头参数
        width, height = camera_config.get("resolution", [640, 480])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, camera_config.get("fps", 30))
        
        try:
            while self.running:
                timestamp = time.time()
                
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    logger.error(f"摄像头 {camera_id} 读取失败")
                    time.sleep(0.1)
                    continue
                
                # 调整帧大小
                frame = cv2.resize(frame, (width, height))
                
                # 添加到缓冲区
                self.frame_buffers[camera_id].append((frame, timestamp))
                
                # 控制帧率
                fps = camera_config.get("fps", 30)
                time.sleep(1 / fps)
        finally:
            if cap:
                cap.release()
    
    def _capture_audio(self, mic_id: str, mic_config: Dict[str, Any]):
        """
        捕获麦克风音频
        """
        # 初始化音频缓冲区
        buffer_size = self.config.get("system", {}).get("max_frame_buffer", 3)
        self.audio_buffers[mic_id] = deque(maxlen=buffer_size)
        
        if not pyaudio_available:
            logger.warning(f"pyaudio不可用，麦克风 {mic_id} 将无法捕获音频")
            while self.running:
                time.sleep(0.1)
            return
        
        # 初始化PyAudio
        if not hasattr(self, 'p'):
            try:
                self.p = pyaudio.PyAudio()
            except Exception as e:
                logger.error(f"初始化PyAudio失败: {e}")
                while self.running:
                    time.sleep(0.1)
                return
        
        # 打开麦克风
        sample_rate = mic_config.get("sample_rate", 16000)
        channels = mic_config.get("channels", 1)
        buffer_size = mic_config.get("buffer_size", 1024)
        
        try:
            stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=buffer_size
            )
            
            self.microphones[mic_id] = stream
            
            while self.running:
                timestamp = time.time()
                
                # 读取音频数据
                try:
                    data = stream.read(buffer_size)
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    
                    # 添加到缓冲区
                    self.audio_buffers[mic_id].append((audio_data, timestamp))
                except Exception as e:
                    logger.error(f"读取音频数据失败: {e}")
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"打开麦克风失败: {e}")
        finally:
            if mic_id in self.microphones:
                stream = self.microphones[mic_id]
                if stream:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except Exception as e:
                        logger.error(f"关闭麦克风失败: {e}")
                del self.microphones[mic_id]
    
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
        获取设备的最新音频
        """
        if device_id in self.audio_buffers and self.audio_buffers[device_id]:
            return self.audio_buffers[device_id][-1]
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
        检查设备是否结束
        对于真实设备，始终返回False，因为它们不会自动结束
        """
        return False
