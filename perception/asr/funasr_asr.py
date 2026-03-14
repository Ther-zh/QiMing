import os
import numpy as np
from typing import Dict, Any, Tuple
from .funAsr import SenseVoiceASR

class FunASRRecognizer:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化FunASR语音识别模型
        
        Args:
            config: 配置字典，包含模型路径等参数
        """
        self.config = config
        self.model_path = config.get('model_path', '/root/autodl-tmp/funasr_models/modelscope_cache/iic/SenseVoiceSmall')
        self.model = None
        self._load_model()
        # 音频缓冲
        self.audio_buffer = []
        self.buffer_max_length = 16000 * 5  # 5秒音频缓冲
        # 唤醒词列表
        self.wake_words = ["你好", "导盲", "导航","小明","小明同学"]
    
    def _load_model(self):
        """
        加载FunASR模型
        """
        try:
            self.model = SenseVoiceASR(
                model_dir=self.model_path,
                device="cuda:0" if os.environ.get('CUDA_VISIBLE_DEVICES') else "cpu",
                verbose=True
            )
            print(f"[ASR] 模型加载成功: {self.model_path}")
        except Exception as e:
            print(f"[ASR] 模型加载失败: {e}")
            raise
    
    def inference(self, audio_data: np.ndarray) -> Tuple[bool, str]:
        """
        执行语音识别
        
        Args:
            audio_data: 输入音频数据
            
        Returns:
            Tuple[bool, str]: (是否检测到唤醒词, 语音转文本结果)
        """
        if self.model is None:
            raise RuntimeError("ASR模型未加载")
        
        # 执行语音识别
        try:
            # 只有当音频数据不为空时才执行识别
            if len(audio_data) > 0:
                # 将当前音频数据添加到缓冲
                self.audio_buffer.extend(audio_data)
                
                # 限制缓冲长度
                if len(self.audio_buffer) > self.buffer_max_length:
                    self.audio_buffer = self.audio_buffer[-self.buffer_max_length:]
                
                # 当缓冲达到一定长度时执行识别
                if len(self.audio_buffer) > 16000 * 1:  # 至少1秒音频
                    asr_text = self.model.recognize(np.array(self.audio_buffer), clean_output=True)
                    print(f"[ASR] 识别结果: {asr_text}")
                else:
                    asr_text = ""
            else:
                asr_text = ""
        except Exception as e:
            print(f"[ASR] 推理失败: {e}")
            asr_text = ""
        
        # 简单的唤醒词检测
        wake_detected = any(word in asr_text for word in self.wake_words)
        
        # 如果检测到唤醒词，清空缓冲
        if wake_detected:
            self.audio_buffer = []
        
        return wake_detected, asr_text
    
    def release(self):
        """
        释放模型资源
        """
        if self.model:
            # 释放模型资源
            self.model.cleanup()
            self.model = None
            print("[ASR] 模型资源已释放")
