import os
import numpy as np
from typing import Dict, Any, Tuple

class FunASRRecognizer:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化FunASR语音识别模型
        
        Args:
            config: 配置字典，包含模型路径等参数
        """
        self.config = config
        self.model_path = config.get('model_path', 'damo/speech_paraformer-small-asr_nat-zh-cn-16k-common-vocab8404-onnx')
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """
        加载FunASR模型
        """
        try:
            from funasr import AutoModel
            self.model = AutoModel(
                model=self.model_path,
                vad_model="damo/speech_fsmn_vad_zh-cn-16k-onnx",
                punc_model="damo/speech_punc_ct-transformer_zh-cn-common-vocab272727-onnx",
                device="cuda:0" if os.environ.get('CUDA_VISIBLE_DEVICES') else "cpu",
                disable_update=True,
                trust_remote_code=True,
                use_onnx=True
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
        result = self.model.predict(
            audio_data=audio_data,
            task="asr",
            vad_silence_time=0.8,
            punc=True
        )
        
        asr_text = result[0]["text"].strip()
        
        # 简单的唤醒词检测
        wake_words = ["你好", "导盲", "导航","小明","小明同学"]
        wake_detected = any(word in asr_text for word in wake_words)
        
        return wake_detected, asr_text
    
    def release(self):
        """
        释放模型资源
        """
        if self.model:
            # 释放模型资源
            self.model = None
            print("[ASR] 模型资源已释放")
