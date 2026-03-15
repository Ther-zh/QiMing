import abc
from typing import Dict, Any, Tuple
import time

from utils.logger import logger
from utils.config_loader import config_loader

class InputDevice(metaclass=abc.ABCMeta):
    """
    输入设备抽象基类
    """
    
    def __init__(self):
        self.config = config_loader.get_config()
        self.running = False
    
    @abc.abstractmethod
    def start(self):
        """
        启动设备
        """
        pass
    
    @abc.abstractmethod
    def stop(self):
        """
        停止设备
        """
        pass
    
    @abc.abstractmethod
    def get_frame(self, device_id: str) -> Tuple[Any, float]:
        """
        获取设备的最新帧
        
        Args:
            device_id: 设备ID
            
        Returns:
            (frame, timestamp) 或 (None, None)
        """
        pass
    
    @abc.abstractmethod
    def get_all_frames(self) -> Dict[str, Tuple[Any, float]]:
        """
        获取所有设备的最新帧
        
        Returns:
            设备ID到(frame, timestamp)的映射
        """
        pass
    
    @abc.abstractmethod
    def get_audio(self, device_id: str) -> Tuple[Any, float]:
        """
        获取设备的最新音频
        
        Args:
            device_id: 设备ID
            
        Returns:
            (audio_data, timestamp) 或 (None, None)
        """
        pass
    
    @abc.abstractmethod
    def get_all_audio(self) -> Dict[str, Tuple[Any, float]]:
        """
        获取所有设备的最新音频
        
        Returns:
            设备ID到(audio_data, timestamp)的映射
        """
        pass
    
    @abc.abstractmethod
    def is_ended(self, device_id: str) -> bool:
        """
        检查设备是否结束（如视频播放完毕）
        
        Args:
            device_id: 设备ID
            
        Returns:
            bool: 设备是否结束
        """
        pass
