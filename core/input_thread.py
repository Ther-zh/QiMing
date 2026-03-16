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
        
        # 音频数据累积
        audio_buffer = []
        buffer_duration = 0.5  # 0.5秒音频
        sample_rate = 16000  # 假设采样率为16kHz
        buffer_size = int(sample_rate * buffer_duration)
        last_audio_send_time = time.time()
        
        try:
            while self.running:
                # 检查视频是否播放完毕（仅模拟模式）
                if hasattr(self.input_device, 'is_ended'):
                    if self.input_device.is_ended(main_camera):
                        logger.info("视频播放完毕，通知其他线程")
                        # 发送剩余的音频数据
                        if audio_buffer:
                            message_queue.send_message("audio", {
                                "type": "audio_data",
                                "audio_data": np.array(audio_buffer),
                                "timestamp": time.time()
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
                    # 累积音频数据
                    audio_buffer.extend(audio_data)
                    
                    # 当累积的音频数据达到一定长度或时间间隔时发送
                    current_time = time.time()
                    if len(audio_buffer) >= buffer_size or (current_time - last_audio_send_time) >= buffer_duration:
                        # 发送音频数据到ASR处理队列
                        message_queue.send_message("audio", {
                            "type": "audio_data",
                            "audio_data": np.array(audio_buffer),
                            "timestamp": current_time
                        })
                        # 清空缓冲
                        audio_buffer = []
                        last_audio_send_time = current_time
                
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