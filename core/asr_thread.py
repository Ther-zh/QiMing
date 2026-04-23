import threading
import time
import numpy as np
from typing import Dict, Any

from utils.logger import logger
from utils.message_queue import message_queue
from utils.config_loader import config_loader

# 导入ASR模块（按配置选择引擎；避免在 import 时就拉起重依赖）
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
        if asr_config.get("type", "real") == "mock":
            return MockFunASRRecognizer(asr_config)

        engine = str(asr_config.get("engine", "funasr")).strip().lower()
        if engine in ("funasr", "sensevoice", "sensevoice_small"):
            from perception.asr.funasr_asr import FunASRRecognizer

            return FunASRRecognizer(asr_config)
        if engine in ("whisper", "whisper_cpp", "whispercpp", "wasr"):
            from perception.Wasr.whisper_recognizer import WhisperCppRecognizer

            return WhisperCppRecognizer(asr_config)

        raise ValueError(f"未知 ASR 引擎: {engine}（期望 funasr / whisper_cpp）")
    
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
                        is_final = message.get("is_final", False)
                        ret = self.asr.inference(audio_data, is_final)
                        # 兼容旧签名 (wake_detected, asr_text) 与新签名 (wake_detected, asr_text, is_speech)
                        wake_detected, asr_text = False, ""
                        is_speech = None
                        try:
                            if isinstance(ret, (tuple, list)) and len(ret) >= 2:
                                wake_detected = bool(ret[0])
                                asr_text = ret[1] or ""
                                if len(ret) >= 3:
                                    is_speech = bool(ret[2])
                            else:
                                raise TypeError("ASR inference returned invalid value")
                        except Exception:
                            # 保底：不让 ASR 线程崩溃
                            wake_detected, asr_text, is_speech = False, "", None
                        
                        if asr_text:
                            # 只在识别到完整句子时输出
                            logger.info(f"[ASR] 识别结果: {asr_text}")
                            # 发送识别结果到推理决策队列
                            message_queue.send_message("inference", {
                                "type": "asr_result",
                                "text": asr_text,
                                "wake_detected": wake_detected,
                                "is_speech": is_speech,
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