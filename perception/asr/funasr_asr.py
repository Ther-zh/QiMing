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
        self.wake_words = ["你好", "导盲", "导航", "小明", "小明同学", "小"]
        # 语音活动检测状态
        self.is_speaking = False
        # 句子结束标志
        self.sentence_end = False
        # 静默计数器
        self.silence_counter = 0
        self.silence_threshold = 5  # 5个周期无语音视为静默
        # 最小识别长度（秒）
        self.min_recognition_length = 1.0
        # 唤醒状态
        self.wake_state = False
        # 唤醒后音频缓冲
        self.wake_audio_buffer = []
        # 唤醒后静默计数器
        self.wake_silence_counter = 0
        self.wake_silence_threshold = 3  # 唤醒后3个周期无语音视为结束
        # 采样率
        self.sample_rate = 16000
    
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
            
            # 加载VAD模型 - 使用正确的FunASR API
            try:
                from funasr import AutoModel
                # 直接使用模型ID从modelscope加载，这样会自动下载和使用正确的版本
                self.vad_model = AutoModel(
                    model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                    device="cuda:0" if os.environ.get('CUDA_VISIBLE_DEVICES') else "cpu"
                )
                print(f"[ASR] VAD模型加载成功")
            except Exception as e:
                print(f"[ASR] VAD模型加载失败: {e}")
                import traceback
                traceback.print_exc()
                self.vad_model = None
            
            # 加载标点模型 - 使用正确的FunASR API
            try:
                from funasr import AutoModel
                # 直接使用模型ID从modelscope加载，这样会自动下载和使用正确的版本
                self.punc_model = AutoModel(
                    model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                    device="cuda:0" if os.environ.get('CUDA_VISIBLE_DEVICES') else "cpu"
                )
                print(f"[ASR] 标点模型加载成功")
            except Exception as e:
                print(f"[ASR] 标点模型加载失败: {e}")
                import traceback
                traceback.print_exc()
                self.punc_model = None
                
        except Exception as e:
            print(f"[ASR] 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
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
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('value', 0) == 1
            return False
        except Exception as e:
            # 失败时使用能量判断
            print(f"[ASR] VAD推理失败，使用能量检测: {e}")
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
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('text', text)
            return text
        except Exception as e:
            print(f"[ASR] 标点添加失败: {e}")
            return text
    
    def inference(self, audio_data: np.ndarray, is_final: bool = False) -> Tuple[bool, str]:
        """
        执行语音识别 - 实时处理短片段音频
        
        Args:
            audio_data: 输入音频数据
            is_final: 是否为最终片段
            
        Returns:
            Tuple[bool, str]: (是否检测到唤醒词, 语音转文本结果)
        """
        if self.model is None:
            raise RuntimeError("ASR模型未加载")
        
        asr_text = ""
        wake_detected = False
        
        # 执行语音识别
        try:
            # 只有当音频数据足够长时才执行处理
            if len(audio_data) > self.sample_rate * 0.3:  # 至少0.3秒
                print(f"[ASR] 处理音频，长度: {len(audio_data)} 样本")
                
                # 检测语音活动
                is_speech = self._detect_voice_activity(audio_data)
                print(f"[ASR] 语音活动检测: {is_speech}")
                
                # 直接使用完整识别
                try:
                    asr_text = self.model.recognize(audio_data, clean_output=True)
                    print(f"[ASR] 原始识别结果: '{asr_text}'")
                    
                    # 不添加标点（根据要求）
                    # asr_text = self._add_punctuation(asr_text)
                    # print(f"[ASR] 带标点结果: '{asr_text}'")
                except Exception as e:
                    print(f"[ASR] 模型识别失败: {e}")
                    import traceback
                    traceback.print_exc()
            
        except Exception as e:
            print(f"[ASR] 推理失败: {e}")
            import traceback
            traceback.print_exc()
            asr_text = ""
        
        # 检查是否已经处于唤醒状态
        if self.wake_state:
            print(f"[ASR] 处于唤醒状态")
            # 添加当前音频到唤醒缓冲
            self.wake_audio_buffer.extend(audio_data.tolist())
            
            # 检测语音活动
            is_speech = self._detect_voice_activity(audio_data)
            if not is_speech:
                self.wake_silence_counter += 1
                print(f"[ASR] 唤醒后静默计数: {self.wake_silence_counter}")
            else:
                self.wake_silence_counter = 0
            
            # 如果静默时间达到阈值，认为句子结束
            if self.wake_silence_counter >= self.wake_silence_threshold or is_final:
                print(f"[ASR] 唤醒后句子结束")
                # 处理完整的唤醒句子
                if len(self.wake_audio_buffer) > self.sample_rate * 0.5:  # 至少0.5秒
                    wake_audio = np.array(self.wake_audio_buffer)
                    try:
                        wake_text = self.model.recognize(wake_audio, clean_output=True)
                        print(f"[ASR] 唤醒句子识别结果: '{wake_text}'")
                        # 使用唤醒句子作为最终结果
                        asr_text = wake_text
                    except Exception as e:
                        print(f"[ASR] 唤醒句子识别失败: {e}")
                
                # 重置唤醒状态
                self.wake_state = False
                self.wake_audio_buffer = []
                self.wake_silence_counter = 0
                wake_detected = True  # 确保返回唤醒状态
                print(f"[ASR] 唤醒处理完成，返回wake_detected=True")
        elif is_final and asr_text:
            # 在最终片段中也检测唤醒词
            print(f"[ASR] 处理最终片段，检测唤醒词")
            # 去除常用标点符号，避免标点干扰唤醒词检测
            import re
            clean_text = re.sub(r'[。，、；：？！,.?!;:\s]', '', asr_text)
            print(f"[ASR] 去除标点后的文本: '{clean_text}'")
            
            # 在原始文本和清洗后的文本中都检测
            wake_detected = any(word in asr_text or word in clean_text for word in self.wake_words)
            if wake_detected:
                print(f"[ASR] 检测到唤醒词")
                for word in self.wake_words:
                    if word in asr_text or word in clean_text:
                        print(f"[ASR]   - 唤醒词 '{word}' 被检测到")
                        print(f"[ASR] 检测到唤醒词，立即返回wake_detected=True")
                        return True, asr_text
        else:
            # 简单的唤醒词检测 - 先去除标点符号再检测
            if asr_text:
                # 去除常用标点符号，避免标点干扰唤醒词检测
                import re
                clean_text = re.sub(r'[。，、；：？！,.?!;:\s]', '', asr_text)
                print(f"[ASR] 去除标点后的文本: '{clean_text}'")
                
                # 在原始文本和清洗后的文本中都检测
                wake_detected = any(word in asr_text or word in clean_text for word in self.wake_words)
                if wake_detected:
                    print(f"[ASR] 检测到唤醒词")
                    for word in self.wake_words:
                        if word in asr_text or word in clean_text:
                            print(f"[ASR]   - 唤醒词 '{word}' 被检测到")
                            # 立即返回唤醒状态，不需要等待句子结束
                            print(f"[ASR] 检测到唤醒词，立即返回wake_detected=True")
                            return True, asr_text
        
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
                if hasattr(self.vad_model, 'cleanup'):
                    self.vad_model.cleanup()
                self.vad_model = None
                print("[ASR] VAD模型资源已释放")
            except Exception as e:
                print(f"[ASR] VAD模型释放失败: {e}")
        
        if self.punc_model:
            try:
                if hasattr(self.punc_model, 'cleanup'):
                    self.punc_model.cleanup()
                self.punc_model = None
                print("[ASR] 标点模型资源已释放")
            except Exception as e:
                print(f"[ASR] 标点模型释放失败: {e}")
