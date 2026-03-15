#!/usr/bin/env python3
"""
测试 ASR 模型的加载和推理功能
"""

import os
import sys
import numpy as np

# 添加系统路径
sys.path.append('/root/MHSEE')

# 测试 ASR 模型
def test_asr():
    print("开始测试 ASR 模型...")
    
    try:
        from perception.asr.funasr_asr import FunASRRecognizer
        config = {
            'model_path': '/root/autodl-tmp/funasr_models/modelscope_cache/iic/SenseVoiceSmall'
        }
        
        # 加载模型
        print("正在加载 ASR 模型...")
        asr = FunASRRecognizer(config)
        print("ASR 模型加载成功")
        
        # 生成测试音频数据
        test_audio = np.random.randn(16000).astype(np.float32)
        
        # 执行推理
        print("正在执行 ASR 推理...")
        wake_detected, asr_text = asr.inference(test_audio)
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
