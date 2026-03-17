import threading
import time
import numpy as np
from typing import Dict, Tuple, Any

from utils.logger import logger
from utils.message_queue import message_queue
from hardware.input_device_factory import InputDeviceFactory

class InputThread(threading.Thread):
    """
    输入设备线程，负责读取视频和音频数据
    """
    
    def __init__(self):
        super().__init__(daemon=True)
        self.running = False
        self.input_device = InputDeviceFactory.create_input_device()
    
    def run(self):
        """
        线程运行方法
        """
        self.running = True
        logger.info("输入设备线程已启动")
        
        # 启动输入设备
        self.input_device.start()
        
        # 主摄像头ID
        main_camera = "camera1"
        
        # 音频数据处理 - 实时发送短片段音频
        audio_buffer = []  # 音频缓冲区
        sample_rate = 16000  # 采样率为16kHz
        segment_duration = 5  # 片段长度（秒）
        overlap_duration = 1  # 重叠长度（秒）
        segment_samples = sample_rate * segment_duration  # 5秒音频样本数
        overlap_samples = sample_rate * overlap_duration  # 1秒重叠样本数
        
        try:
            while self.running:
                # 检查视频是否播放完毕（仅模拟模式）
                if hasattr(self.input_device, 'is_ended'):
                    if self.input_device.is_ended(main_camera):
                        logger.info("视频播放完毕，通知其他线程")
                        # 发送剩余的音频数据
                        if audio_buffer:
                            logger.info(f"[Input] 发送剩余音频数据，长度: {len(audio_buffer)}")
                            message_queue.send_message("audio", {
                                "type": "audio_data",
                                "audio_data": np.array(audio_buffer),
                                "timestamp": time.time(),
                                "is_final": True
                            })
                        # 发送视频结束消息
                        message_queue.send_message("control", {"type": "video_ended"})
                        break
                
                # 获取摄像头帧
                frames = self.input_device.get_all_frames()
                
                # 处理主摄画面
                if main_camera in frames:
                    frame, timestamp = frames[main_camera]
                    if frame is not None:
                        # 发送帧到视觉处理队列
                        message_queue.send_message("vision", {
                            "type": "frame",
                            "frame": frame,
                            "timestamp": timestamp,
                            "camera_id": main_camera
                        })
                
                # 获取音频数据
                audio_data, audio_timestamp = self.input_device.get_audio(main_camera)
                if audio_data is not None:
                    # 添加到缓冲区
                    audio_buffer.extend(audio_data)
                    
                    # 当缓冲区达到5秒时，发送音频片段
                    if len(audio_buffer) >= segment_samples:
                        # 发送完整的5秒片段
                        segment = audio_buffer[:segment_samples]
                        message_queue.send_message("audio", {
                            "type": "audio_data",
                            "audio_data": np.array(segment),
                            "timestamp": time.time(),
                            "is_final": False
                        })
                        # 保留1秒重叠部分
                        audio_buffer = audio_buffer[segment_samples - overlap_samples:]
                
                # 短暂休眠，避免占用过多CPU
                time.sleep(0.01)
        finally:
            # 停止输入设备
            self.input_device.stop()
            logger.info("输入设备线程已停止")
    
    def stop(self):
        """
        停止线程
        """
        self.running = False