import numpy as np
import torch
import time
import sys
import os
from collections import deque

# 导入我们的ASR模块
from funAsr import SenseVoiceASR


class VoiceAssistant:
    def __init__(self, asr_model_dir=None, vad_model_dir=None, device=None):
        """
        初始化语音助手
        :param asr_model_dir: SenseVoice模型路径
        :param vad_model_dir: VAD模型路径
        :param device: 运行设备
        """
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 默认路径配置
        self.ASR_MODEL_DIR = asr_model_dir or "/root/autodl-tmp/funasr_models/modelscope_cache/iic/SenseVoiceSmall"
        self.VAD_MODEL_DIR = vad_model_dir or "/root/autodl-tmp/funasr_models/modelscope_cache/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        
        # 初始化参数
        self.sample_rate = 16000
        self.chunk_size = 960  # 60ms chunk (16000 * 0.06)
        self.silence_threshold = 1.0  # 1秒停顿判定为语义结束
        self.max_speech_duration = 30  # 最大语音时长30秒
        
        # 状态变量
        self.is_speaking = False
        self.speech_buffer = []
        self.last_speech_time = 0
        self.cache = {}
        
        # 初始化模型
        self.asr = None
        self.vad_model = None
        self._init_models()
        
    def _init_models(self):
        """初始化ASR和VAD模型"""
        print("正在初始化模型...")
        
        # 初始化ASR
        self.asr = SenseVoiceASR(
            model_dir=self.ASR_MODEL_DIR,
            device=self.device,
            verbose=False
        )
        
        # 初始化VAD
        from funasr import AutoModel
        self.vad_model = AutoModel(
            model=self.VAD_MODEL_DIR,
            device=self.device,
            disable_update=True
        )
        
        print("✅ 模型初始化完成，开始监听...\n")
        
    def _is_voice(self, audio_chunk):
        """使用VAD检测是否为语音"""
        result = self.vad_model.generate(input=audio_chunk, cache={})
        return len(result) > 0 and result[0].get("value", None) is not None
        
    def process_audio_stream(self):
        """处理音频流的主循环"""
        try:
            import sounddevice as sd
            
            # 音频回调函数
            def audio_callback(indata, frames, time_info, status):
                if status:
                    print(status, file=sys.stderr)
                
                audio_chunk = indata[:, 0].astype(np.float32)
                self._process_chunk(audio_chunk)
            
            # 开始音频流
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.chunk_size,
                callback=audio_callback
            ):
                print("🎤 正在监听... (按 Ctrl+C 停止)\n")
                while True:
                    time.sleep(0.1)
                    
        except ImportError:
            print("❌ 请先安装 sounddevice: pip install sounddevice")
            print("💡 如遇PortAudio错误，执行: apt-get install portaudio19-dev")
        except KeyboardInterrupt:
            print("\n\n👋 程序已停止")
        finally:
            if self.asr:
                self.asr.cleanup()
    
    def _process_chunk(self, audio_chunk):
        """处理单个音频块"""
        current_time = time.time()
        has_voice = self._is_voice(audio_chunk)
        
        if has_voice:
            # 检测到语音
            if not self.is_speaking:
                # 开始新的语音段
                self.is_speaking = True
                self.speech_buffer = []
                self.cache = {}
                print("🎙️  检测到语音输入...", end="", flush=True)
            
            self.speech_buffer.append(audio_chunk)
            self.last_speech_time = current_time
            
            # 实时流式识别（可选）
            # text, self.cache = self.asr.stream_recognize(audio_chunk, self.cache)
            # if text:
            #     print(f"\r🎙️  实时识别: {text}", end="", flush=True)
            
        else:
            # 没有检测到语音
            if self.is_speaking:
                # 检查是否停顿足够长时间
                if current_time - self.last_speech_time > self.silence_threshold:
                    # 语义结束，处理完整语音
                    self.is_speaking = False
                    self._process_complete_speech()
                    
                # 继续添加到缓冲区（保持短暂停顿）
                elif len(self.speech_buffer) > 0:
                    self.speech_buffer.append(audio_chunk)
    
    def _process_complete_speech(self):
        """处理完整的语音段"""
        if len(self.speech_buffer) == 0:
            return
        
        print("\r🔍  正在识别...", end="", flush=True)
        
        # 拼接所有音频块
        full_audio = np.concatenate(self.speech_buffer)
        
        # 完整识别
        text = self.asr.recognize(full_audio, clean_output=True)
        
        if text.strip():
            print(f"\r✅ 识别结果: {text}\n")
            self._call_llm(text)
        else:
            print("\r❌ 未识别到有效内容\n")
        
        # 重置缓冲区
        self.speech_buffer = []
        self.cache = {}
        print("🎤 继续监听... (按 Ctrl+C 停止)\n")
    
    def _call_llm(self, text):
        """调用大模型接口（暂时用print代替）"""
        print("="*60)
        print(f"🤖 大模型接口调用（模拟）:")
        print(f"   输入指令: \"{text}\"")
        print("="*60)
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="语音助手主程序")
    parser.add_argument("--asr_model", type=str, default=None, help="ASR模型路径")
    parser.add_argument("--vad_model", type=str, default=None, help="VAD模型路径")
    parser.add_argument("--device", type=str, default=None, help="运行设备 (cuda:0/cpu)")
    args = parser.parse_args()
    
    # 启动语音助手
    assistant = VoiceAssistant(
        asr_model_dir=args.asr_model,
        vad_model_dir=args.vad_model,
        device=args.device
    )
    assistant.process_audio_stream()