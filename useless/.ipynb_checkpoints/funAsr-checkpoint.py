import os
import sys
import warnings
import logging
import contextlib

# ======================== 终极静默：在最开始就重定向输出 ========================
# 1. 保存原始输出
original_stdout = sys.stdout
original_stderr = sys.stderr

# 2. 定义静默上下文管理器
@contextlib.contextmanager
def silence_output():
    devnull = open(os.devnull, 'w')
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = devnull, devnull
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        devnull.close()

# 3. 环境变量（在所有导入前设置）
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 4. 彻底禁用所有警告和日志
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.ERROR)
logging.getLogger("funasr").setLevel(logging.ERROR)
logging.getLogger("modelscope").setLevel(logging.ERROR)

# ======================== 现在才导入其他库 ========================
import numpy as np
import torch


class SenseVoiceASR:
    # 默认配置你的本地模型路径
    DEFAULT_MODEL_DIR = "/root/autodl-tmp/funasr_models/modelscope_cache/iic/SenseVoiceSmall"
    
    def __init__(self, model_dir=None, device=None, verbose=False):
        """
        初始化SenseVoiceSmall语音识别模块
        :param model_dir: 本地模型路径（默认使用已配置的路径）
        :param device: 运行设备（None则自动检测）
        :param verbose: 是否显示详细加载信息（默认False）
        """
        self.model_dir = model_dir or self.DEFAULT_MODEL_DIR
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.model = None
        self._load_model()

    def _load_model(self):
        """内部方法：加载模型（完全静默）"""
        try:
            from funasr import AutoModel
            
            # 验证模型路径
            if not os.path.exists(self.model_dir):
                raise FileNotFoundError(f"模型路径不存在: {self.model_dir}")
            
            # 将模型目录加入路径
            sys.path.insert(0, self.model_dir)
            
            # 完全静默加载模型
            with silence_output():
                self.model = AutoModel(
                    model=self.model_dir,
                    device=self.device,
                    trust_remote_code=True,
                    disable_update=True
                )
            
            if self.verbose:
                print("模型加载成功！")
            
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)[:500]}")

    def recognize(self, audio_input, sample_rate=16000, clean_output=True):
        """
        识别单段音频
        :param audio_input: 音频输入（文件路径或numpy数组）
        :param sample_rate: 采样率（默认16000）
        :param clean_output: 是否清理输出标签（如<|zh|><|Speech|>等）
        :return: 识别文本
        """
        if isinstance(audio_input, str):
            import librosa
            audio, _ = librosa.load(audio_input, sr=sample_rate, mono=True)
            audio = audio.astype(np.float32)
        elif isinstance(audio_input, np.ndarray):
            audio = audio_input.astype(np.float32)
        else:
            raise ValueError("audio_input 必须是文件路径或numpy数组")

        try:
            # 静默识别
            with silence_output():
                result = self.model.generate(input=audio, cache={}, disable_pbar=True)
            
            text = result[0]["text"] if result else ""
            
            # 清理输出标签
            if clean_output:
                text = self._clean_output(text)
            
            return text
        except Exception as e:
            raise RuntimeError(f"识别失败: {str(e)[:500]}")

    def _clean_output(self, text):
        """内部方法：清理SenseVoice输出的特殊标签"""
        import re
        # 移除所有<|xxx|>格式的标签
        text = re.sub(r'<\|[^|]+\|>', '', text)
        # 移除首尾空白
        return text.strip()

    def stream_recognize(self, audio_chunk, cache=None, clean_output=True):
        """
        流式识别接口
        :param audio_chunk: 音频块（numpy数组）
        :param cache: 流式缓存字典（用于保持上下文）
        :param clean_output: 是否清理输出标签
        :return: (识别文本, 更新后的cache)
        """
        if cache is None:
            cache = {}
        
        try:
            with silence_output():
                result = self.model.generate(input=audio_chunk, cache=cache, disable_pbar=True)
            
            text = result[0]["text"] if result else ""
            
            if clean_output:
                text = self._clean_output(text)
            
            return text, cache
        except Exception as e:
            raise RuntimeError(f"流式识别失败: {str(e)[:500]}")

    def cleanup(self):
        """资源清理"""
        if self.device == "cuda:0":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SenseVoiceSmall 音频识别测试")
    parser.add_argument("--model_dir", type=str, default=None, help="本地模型路径（可选）")
    parser.add_argument("--audio_path", type=str, required=True, help="测试音频文件路径")
    parser.add_argument("--device", type=str, default=None, help="运行设备 (cuda:0/cpu)")
    parser.add_argument("--verbose", action="store_true", help="显示详细加载信息")
    parser.add_argument("--keep_tags", action="store_true", help="保留输出标签（默认清理）")
    args = parser.parse_args()

    try:
        asr = SenseVoiceASR(
            model_dir=args.model_dir, 
            device=args.device,
            verbose=args.verbose
        )
        result = asr.recognize(
            args.audio_path, 
            clean_output=not args.keep_tags
        )
        print(f"\n识别结果:\n{result}")
        asr.cleanup()
    except Exception as e:
        print(f"错误: {e}")