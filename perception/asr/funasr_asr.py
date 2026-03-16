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
        self.vad_model_path = config.get('vad_model_path', '/root/autodl-tmp/funasr_models/modelscope_cache/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch')
        self.punc_model_path = config.get('punc_model_path', '/root/autodl-tmp/funasr_models/modelscope_cache/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch')
        self.model = None
        self.vad_model = None
        self.punc_model = None
        self._load_models()
        # 音频缓冲
        self.audio_buffer = []
        self.buffer_max_length = 16000 * 10  # 10秒音频缓冲
        # 流式识别缓存
        self.stream_cache = {}
        # 唤醒词列表
        self.wake_words = ["你好", "导盲", "导航","小明","小明同学"]
        # 语音活动检测状态
        self.is_speaking = False
        # 句子结束标志
        self.sentence_end = False
        # 静默计数器
        self.silence_counter = 0
        self.silence_threshold = 5  # 5个周期无语音视为静默
        # 最小识别长度（秒）
        self.min_recognition_length = 1.0
    
    def _load_models(self):
        """
        加载FunASR模型
        """
        try:
            # 加载主ASR模型
            self.model = SenseVoiceASR(
                model_dir=self.model_path,
                device="cuda:0" if os.environ.get('CUDA_VISIBLE_DEVICES') else "cpu",
                verbose=True
            )
            print(f"[ASR] 主模型加载成功: {self.model_path}")
            
            # 加载VAD模型
            try:
                from funasr import AutoModel
                self.vad_model = AutoModel(
                    model=self.vad_model_path,
                    device="cuda:0" if os.environ.get('CUDA_VISIBLE_DEVICES') else "cpu",
                    trust_remote_code=True
                )
                print(f"[ASR] VAD模型加载成功: {self.vad_model_path}")
            except Exception as e:
                print(f"[ASR] VAD模型加载失败: {e}")
                self.vad_model = None
            
            # 加载标点模型
            try:
                from funasr import AutoModel
                self.punc_model = AutoModel(
                    model=self.punc_model_path,
                    device="cuda:0" if os.environ.get('CUDA_VISIBLE_DEVICES') else "cpu",
                    trust_remote_code=True
                )
                print(f"[ASR] 标点模型加载成功: {self.punc_model_path}")
            except Exception as e:
                print(f"[ASR] 标点模型加载失败: {e}")
                self.punc_model = None
                
        except Exception as e:
            print(f"[ASR] 模型加载失败: {e}")
            raise
    
    def _detect_voice_activity(self, audio_data: np.ndarray) -> bool:
        """
        检测语音活动
        
        Args:
            audio_data: 输入音频数据
            
        Returns:
            bool: 是否检测到语音活动
        """
        if self.vad_model is None:
            # 如果没有VAD模型，简单判断能量
            energy = np.sum(np.square(audio_data)) / len(audio_data)
            return energy > 0.001
        
        try:
            result = self.vad_model.generate(input=audio_data)
            return result[0].get('value', 0) == 1
        except Exception as e:
            # 失败时使用能量判断
            energy = np.sum(np.square(audio_data)) / len(audio_data)
            return energy > 0.001
    
    def _add_punctuation(self, text: str) -> str:
        """
        添加标点
        
        Args:
            text: 无标点文本
            
        Returns:
            str: 带标点文本
        """
        if self.punc_model is None or not text:
            return text
        
        try:
            result = self.punc_model.generate(input=text)
            return result[0].get('text', text)
        except Exception as e:
            print(f"[ASR] 标点添加失败: {e}")
            return text
    
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
        
        asr_text = ""
        wake_detected = False
        
        # 执行语音识别
        try:
            # 只有当音频数据不为空时才执行处理
            if len(audio_data) > 0:
                # 检测语音活动
                has_voice = self._detect_voice_activity(audio_data)
                
                if has_voice:
                    self.is_speaking = True
                    self.silence_counter = 0
                    # 将当前音频数据添加到缓冲
                    self.audio_buffer.extend(audio_data)
                    
                    # 限制缓冲长度
                    if len(self.audio_buffer) > self.buffer_max_length:
                        self.audio_buffer = self.audio_buffer[-self.buffer_max_length:]
                else:
                    # 无语音活动
                    if self.is_speaking:
                        self.silence_counter += 1
                        # 如果持续静默，认为句子结束
                        if self.silence_counter >= self.silence_threshold:
                            self.sentence_end = True
                            self.is_speaking = False
                            # 执行最终识别
                            min_samples = int(16000 * self.min_recognition_length)
                            if len(self.audio_buffer) > min_samples:  # 至少1秒音频
                                # 使用完整识别而不是流式识别
                                asr_text = self.model.recognize(np.array(self.audio_buffer), clean_output=True)
                                # 添加标点
                                asr_text = self._add_punctuation(asr_text)
                                # 清空缓冲
                                self.audio_buffer = []
            
        except Exception as e:
            print(f"[ASR] 推理失败: {e}")
            asr_text = ""
        
        # 简单的唤醒词检测
        if asr_text:
            wake_detected = any(word in asr_text for word in self.wake_words)
        
        return wake_detected, asr_text
    
    def release(self):
        """
        释放模型资源
        """
        if self.model:
            # 释放模型资源
            self.model.cleanup()
            self.model = None
            print("[ASR] 主模型资源已释放")
        
        if self.vad_model:
            try:
                self.vad_model.cleanup()
                self.vad_model = None
                print("[ASR] VAD模型资源已释放")
            except Exception as e:
                print(f"[ASR] VAD模型释放失败: {e}")
        
        if self.punc_model:
            try:
                self.punc_model.cleanup()
                self.punc_model = None
                print("[ASR] 标点模型资源已释放")
            except Exception as e:
                print(f"[ASR] 标点模型释放失败: {e}")
