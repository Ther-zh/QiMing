#!/usr/bin/env python3
"""
测试 LLM 模型的加载和推理功能
"""

import os
import sys
import numpy as np
from PIL import Image

# 添加系统路径
sys.path.append('/root/MHSEE')

# 测试 LLM 模型
def test_llm():
    print("开始测试 LLM 模型...")
    
    try:
        from perception.llm.qwen_multimodal import QwenMultimodal
        config = {
            'model_path': '/root/autodl-tmp/qwen35/tclf90/Qwen3___5-4B-AWQ'
        }
        
        # 加载模型
        print("正在加载 LLM 模型...")
        llm = QwenMultimodal(config)
        print("LLM 模型加载成功")
        
        # 测试纯文本推理
        print("正在执行 LLM 纯文本推理...")
        text_response = llm.inference((None, {}, "你好，请用一句话介绍你自己。"))
        print(f"LLM 纯文本推理结果: {text_response}")
        
        # 释放模型
        llm.release()
        print("LLM 模型资源已释放")
        
        return True
    except Exception as e:
        print(f"LLM 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_llm()
    if success:
        print("\nLLM 模型测试成功！")
    else:
        print("\nLLM 模型测试失败！")
