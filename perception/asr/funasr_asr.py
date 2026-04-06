import os
import numpy as np
import torch
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
        _base = "/home/nvidia/models/root/autodl-tmp/funasr_models/modelscope_cache/iic"
        self.model_path = config.get("model_path", f"{_base}/SenseVoiceSmall")
        self.vad_model_path = config.get("vad_model_path", f"{_base}/speech_fsmn_vad_zh-cn-16k-common-pytorch")
        self.punc_model_path = config.get("punc_model_path", f"{_base}/punc_ct-transformer_zh-cn-common-vocab272727-pytorch")
        self.model = None
        self.vad_model = None
        self.punc_model = None
        self._enable_vad = bool(config.get("enable_vad", True))
        self._enable_punctuation = bool(config.get("enable_punctuation", True))
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
        # device：config 可指定 cuda:0 / cpu；默认 auto 与 GPU 可用时走 cuda（Jetson 8GB 上与 YOLO+VDA 并存易 OOM，可设 cpu）
        device = self.config.get("device", "auto")
        if device is None or str(device).lower() == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[ASR] 使用设备: {device}")
        # 仅使用本机已下载的 ModelScope 缓存（见 config 旁 modelscope_cache；内含 models/iic -> ../iic）
        os.environ.setdefault(
            "MODELSCOPE_CACHE",
            "/home/nvidia/models/root/autodl-tmp/funasr_models/modelscope_cache",
        )
        os.environ.setdefault("FUNASR_LOCAL_ONLY", "1")

        try:
            # 加载主ASR模型
            self.model = SenseVoiceASR(
                model_dir=self.model_path,
                device=device,
                verbose=True
            )
            print(f"[ASR] 主模型加载成功: {self.model_path}")

            # VAD / 标点：可通过 config 关闭以降低 Jetson 统一内存占用
            if self._enable_vad:
                try:
                    from funasr import AutoModel

                    if not os.path.isdir(self.vad_model_path):
                        raise FileNotFoundError(f"VAD 缓存目录不存在: {self.vad_model_path}")
                    self.vad_model = AutoModel(
                        model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                        device=device,
                        disable_update=True,
                        check_latest=False,
                    )
                    print("[ASR] VAD模型加载成功（本地缓存）")
                except Exception as e:
                    print(f"[ASR] VAD模型加载失败: {e}")
                    import traceback

                    traceback.print_exc()
                    self.vad_model = None
            else:
                print("[ASR] 已跳过 VAD 加载（enable_vad=false，使用能量阈值检测语音）")

            if self._enable_punctuation:
                try:
                    from funasr import AutoModel

                    if not os.path.isdir(self.punc_model_path):
                        raise FileNotFoundError(f"标点缓存目录不存在: {self.punc_model_path}")
                    self.punc_model = AutoModel(
                        model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                        device=device,
                        disable_update=True,
                        check_latest=False,
                    )
                    print("[ASR] 标点模型加载成功（本地缓存）")
                except Exception as e:
                    print(f"[ASR] 标点模型加载失败: {e}")
                    import traceback

                    traceback.print_exc()
                    self.punc_model = None
            else:
                print("[ASR] 已跳过标点模型加载（enable_punctuation=false）")
                
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
        from utils.logger import logger
        from utils.config_loader import config_loader
        
        # 获取verbose配置
        try:
            config = config_loader.get_config()
            verbose = config.get("system", {}).get("verbose", True)
        except Exception:
            verbose = True
        
        if self.model is None:
            raise RuntimeError("ASR模型未加载")
        
        asr_text = ""
        wake_detected = False
        
        # 执行语音识别
        try:
            # 只有当音频数据足够长时才执行处理
            if len(audio_data) > self.sample_rate * 0.3:  # 至少0.3秒
                if verbose:
                    print(f"[ASR] 处理音频，长度: {len(audio_data)} 样本")
                
                # 检测语音活动
                is_speech = self._detect_voice_activity(audio_data)
                if verbose:
                    print(f"[ASR] 语音活动检测: {is_speech}")
                
                # 直接使用完整识别
                try:
                    asr_text = self.model.recognize(audio_data, clean_output=True)
                    if verbose:
                        print(f"[ASR] 原始识别结果: '{asr_text}'")
                    if self._enable_punctuation and self.punc_model and asr_text:
                        asr_text = self._add_punctuation(asr_text)
                        if verbose:
                            print(f"[ASR] 带标点结果: '{asr_text}'")
                except Exception as e:
                    if verbose:
                        print(f"[ASR] 模型识别失败: {e}")
                        import traceback
                        traceback.print_exc()
            
        except Exception as e:
            if verbose:
                print(f"[ASR] 推理失败: {e}")
                import traceback
                traceback.print_exc()
            asr_text = ""
        
        # 检查是否已经处于唤醒状态
        if self.wake_state:
            if verbose:
                print(f"[ASR] 处于唤醒状态")
            # 添加当前音频到唤醒缓冲
            self.wake_audio_buffer.extend(audio_data.tolist())
            
            # 检测语音活动
            is_speech = self._detect_voice_activity(audio_data)
            if not is_speech:
                self.wake_silence_counter += 1
                if verbose:
                    print(f"[ASR] 唤醒后静默计数: {self.wake_silence_counter}")
            else:
                self.wake_silence_counter = 0
            
            # 如果静默时间达到阈值，认为句子结束
            if self.wake_silence_counter >= self.wake_silence_threshold or is_final:
                if verbose:
                    print(f"[ASR] 唤醒后句子结束")
                # 处理完整的唤醒句子
                if len(self.wake_audio_buffer) > self.sample_rate * 0.5:  # 至少0.5秒
                    wake_audio = np.array(self.wake_audio_buffer)
                    try:
                        wake_text = self.model.recognize(wake_audio, clean_output=True)
                        if self._enable_punctuation and self.punc_model and wake_text:
                            wake_text = self._add_punctuation(wake_text)
                        if verbose:
                            print(f"[ASR] 唤醒句子识别结果: '{wake_text}'")
                        asr_text = wake_text
                    except Exception as e:
                        if verbose:
                            print(f"[ASR] 唤醒句子识别失败: {e}")
                
                # 重置唤醒状态
                self.wake_state = False
                self.wake_audio_buffer = []
                self.wake_silence_counter = 0
                wake_detected = True  # 确保返回唤醒状态
                if verbose:
                    print(f"[ASR] 唤醒处理完成，返回wake_detected=True")
        elif is_final and asr_text:
            # 在最终片段中也检测唤醒词
            if verbose:
                print(f"[ASR] 处理最终片段，检测唤醒词")
            # 去除常用标点符号，避免标点干扰唤醒词检测
            import re
            clean_text = re.sub(r'[。，、；：？！,.?!;:\s]', '', asr_text)
            if verbose:
                print(f"[ASR] 去除标点后的文本: '{clean_text}'")
            
            # 在原始文本和清洗后的文本中都检测
            wake_detected = any(word in asr_text or word in clean_text for word in self.wake_words)
            if wake_detected:
                if verbose:
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
                if verbose:
                    print(f"[ASR] 去除标点后的文本: '{clean_text}'")
                
                # 在原始文本和清洗后的文本中都检测
                wake_detected = any(word in asr_text or word in clean_text for word in self.wake_words)
                if wake_detected:
                    if verbose:
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
