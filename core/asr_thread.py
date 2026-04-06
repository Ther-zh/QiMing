import threading
import time
import numpy as np
from typing import Dict, Any

from utils.logger import logger
from utils.message_queue import message_queue
from utils.config_loader import config_loader
from utils.llm_gate import is_llm_busy

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
        if asr_config.get("type", "real") != "mock":
            return FunASRRecognizer(asr_config)
        else:
            return MockFunASRRecognizer(asr_config)
    
    def run(self):
        """
        线程运行方法
        """
        self.running = True
        logger.info("ASR线程已启动")
        
        pause_asr = bool(
            config_loader.get_config()
            .get("system", {})
            .get("pause_asr_during_llm", True)
        )
        pending = None
        try:
            while self.running:
                if pending is None:
                    message = message_queue.receive_message("audio", block=False)
                    if message and message.get("type") == "audio_data":
                        pending = message
                    else:
                        message = None
                else:
                    message = pending

                if message and message.get("type") == "audio_data":
                    if pause_asr and is_llm_busy():
                        # LLM（Ollama）与 FunASR 同抢 Jetson 统一内存时，Ollama 常报需 2.7GiB 而失败
                        time.sleep(0.06)
                        continue
                    pending = None
                    audio_data = message.get("audio_data")
                    timestamp = message.get("timestamp")

                    if audio_data is not None:
                        is_final = message.get("is_final", False)
                        wake_detected, asr_text = self.asr.inference(
                            audio_data, is_final
                        )

                        if asr_text:
                            logger.info(f"[ASR] 识别结果: {asr_text}")
                            message_queue.send_message(
                                "inference",
                                {
                                    "type": "asr_result",
                                    "text": asr_text,
                                    "wake_detected": wake_detected,
                                    "timestamp": timestamp,
                                },
                            )

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