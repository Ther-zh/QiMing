#!/usr/bin/env python3
"""
测试 ASR 模型的加载和推理功能

用法（在 QiMing 目录下）:
  python sing_test/test_asr.py              # 默认 CPU，避免 Jetson 上 CUDA 分配失败
  ASR_DEVICE=cuda python sing_test/test_asr.py
"""

import os
import sys
import numpy as np

# QiMing 项目根（本机为 /home/nvidia/MHSEE/QiMing，勿写死 /root/MHSEE）
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 与 config.yaml 中 models.asr 一致的本地缓存路径
_DEFAULT_MODEL = (
    "/home/nvidia/models/root/autodl-tmp/funasr_models/modelscope_cache/iic/SenseVoiceSmall"
)


# 测试 ASR 模型
def test_asr():
    print("开始测试 ASR 模型...")
    device = os.environ.get("ASR_DEVICE", "cpu").strip() or "cpu"
    print(f"ASR_DEVICE={device!r} (可用: cpu / cuda / cuda:0 / auto)")

    try:
        from perception.asr.funasr_asr import FunASRRecognizer
        config = {
            "model_path": os.environ.get("ASR_MODEL_PATH", _DEFAULT_MODEL),
            "device": device,
            "enable_vad": False,
            "enable_punctuation": False,
        }
        
        # 加载模型
        print("正在加载 ASR 模型...")
        asr = FunASRRecognizer(config)
        print("ASR 模型加载成功")
        
        # 生成测试音频数据
        test_audio = np.random.randn(16000).astype(np.float32)
        
        # 执行推理
        print("正在执行 ASR 推理...")
        wake_detected, asr_text, _is_speech = asr.inference(test_audio)
        print(f"ASR 推理结果: {asr_text}")
        print(f"唤醒词检测: {wake_detected}")
        
        # 释放模型
        asr.release()
        print("ASR 模型资源已释放")
        
        return True
    except Exception as e:
        print(f"ASR 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_asr()
    if success:
        print("\nASR 模型测试成功！")
    else:
        print("\nASR 模型测试失败！")
