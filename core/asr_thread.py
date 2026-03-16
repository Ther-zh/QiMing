import threading
import time
import numpy as np
from typing import Dict, Any

from utils.logger import logger
from utils.message_queue import message_queue
from utils.config_loader import config_loader

# 导入ASR模块
from perception.asr.funasr_asr import FunASRRecognizer
from perception.asr.mock_asr import MockFunASRRecognizer

class ASRThread(threading.Thread):
    """
    ASR线程，负责处理音频数据和语音识别
    """
    
    def __init__(self):
        super().__init__(daemon=True)
        self.running = False
        self.asr = self._init_asr()
    
    def _init_asr(self):
        """
        初始化ASR模块
        """
        config = config_loader.get_config()
        asr_config = config.get("models", {}).get("asr", {})
        if asr_config.get("type", "mock") == "real":
            return FunASRRecognizer(asr_config)
        else:
            return MockFunASRRecognizer(asr_config)
    
    def run(self):
        """
        线程运行方法
        """
        self.running = True
        logger.info("ASR线程已启动")
        
        try:
            while self.running:
                # 从音频队列接收消息
                message = message_queue.receive_message("audio", block=False)
                if message and message.get("type") == "audio_data":
                    audio_data = message.get("audio_data")
                    timestamp = message.get("timestamp")
                    
                    if audio_data is not None:
                        # 执行ASR识别
                        wake_detected, asr_text = self.asr.inference(audio_data)
                        
                        if asr_text:
                            # 只在识别到完整句子时输出
                            logger.info(f"[ASR] 识别结果: {asr_text}")
                            # 发送识别结果到推理决策队列
                            message_queue.send_message("inference", {
                                "type": "asr_result",
                                "text": asr_text,
                                "wake_detected": wake_detected,
                                "timestamp": timestamp
                            })
                
                # 短暂休眠，避免占用过多CPU
                time.sleep(0.01)
        finally:
            # 释放ASR资源
            if hasattr(self, 'asr'):
                self.asr.release()
            logger.info("ASR线程已停止")
    
    def stop(self):
        """
        停止线程
        """
        self.running = False