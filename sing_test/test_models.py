#!/usr/bin/env python3
"""
测试 ASR 和 LLM 模型的加载和推理功能
"""

import os
import sys
import numpy as np
from PIL import Image

# 添加系统路径
sys.path.append('/root/MHSEE')

# 测试 ASR 模型
def test_asr():
    print("\n=== 测试 ASR 模型 ===")
    try:
        from perception.asr.funasr_asr import FunASRRecognizer
        config = {
            'model_path': '/root/autodl-tmp/funasr_models/modelscope_cache/iic/SenseVoiceSmall'
        }
        asr = FunASRRecognizer(config)
        print("ASR 模型加载成功")
        
        # 生成测试音频数据
        test_audio = np.random.randn(16000).astype(np.float32)
        wake_detected, asr_text = asr.inference(test_audio)
        print(f"ASR 推理结果: {asr_text}")
        print(f"唤醒词检测: {wake_detected}")
        
        asr.release()
        return True
    except Exception as e:
        print(f"ASR 测试失败: {e}")
        return False

# 测试 LLM 模型
def test_llm():
    print("\n=== 测试 LLM 模型 ===")
    try:
        from perception.llm.qwen_multimodal import QwenMultimodal
        config = {
            'model_path': '/root/autodl-tmp/qwen35/tclf90/Qwen3___5-4B-AWQ'
        }
        llm = QwenMultimodal(config)
        print("LLM 模型加载成功")
        
        # 测试纯文本推理
        text_response = llm.inference((None, {}, "你好，请用一句话介绍你自己。"))
        print(f"LLM 纯文本推理结果: {text_response}")
        
        llm.release()
        return True
    except Exception as e:
        print(f"LLM 测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试模型...")
    asr_success = test_asr()
    llm_success = test_llm()
    
    print("\n=== 测试结果 ===")
    print(f"ASR 模型: {'成功' if asr_success else '失败'}")
    print(f"LLM 模型: {'成功' if llm_success else '失败'}")
