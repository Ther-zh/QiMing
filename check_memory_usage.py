import torch
import numpy as np
import cv2
from pynvml import *
import time

# 初始化NVML
nvmlInit()

# 获取GPU信息
gpu_count = nvmlDeviceGetCount()
gpu_handle = nvmlDeviceGetHandleByIndex(0)

def get_gpu_memory():
    """获取GPU内存使用情况"""
    info = nvmlDeviceGetMemoryInfo(gpu_handle)
    total = info.total / 1024**3  # 转换为GB
    used = info.used / 1024**3
    free = info.free / 1024**3
    return total, used, free

def print_memory_usage(stage):
    """打印内存使用情况"""
    total, used, free = get_gpu_memory()
    print(f"[{stage}] GPU内存: 总计={total:.2f}GB, 使用={used:.2f}GB, 可用={free:.2f}GB")

# 开始测试
print("开始测试模型显存占用...")
print_memory_usage("初始状态")

# 1. 测试YOLO模型
print("\n=== 测试YOLO模型 ===")
try:
    from perception.yolo.yolo_detector import YoloDetector
    from utils.config_loader import config_loader
    
    config = config_loader.get_config()
    yolo_config = config.get("models", {}).get("yolo", {})
    
    print_memory_usage("加载YOLO前")
    yolo = YoloDetector(yolo_config)
    print_memory_usage("加载YOLO后")
    
    # 测试推理
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print_memory_usage("YOLO推理前")
    yolo_results = yolo.inference(test_frame)
    print_memory_usage("YOLO推理后")
    
    yolo.release()
    print_memory_usage("释放YOLO后")
except Exception as e:
    print(f"YOLO测试失败: {e}")

# 2. 测试VDA模型
print("\n=== 测试VDA模型 ===")
try:
    from perception.vda.vda_depth import VDADepthEstimator
    
    config = config_loader.get_config()
    vda_config = config.get("models", {}).get("vda", {})
    
    print_memory_usage("加载VDA前")
    vda = VDADepthEstimator(vda_config)
    print_memory_usage("加载VDA后")
    
    # 测试推理
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print_memory_usage("VDA推理前")
    depth_map = vda.inference(test_frame)
    print_memory_usage("VDA推理后")
    
    vda.release()
    print_memory_usage("释放VDA后")
except Exception as e:
    print(f"VDA测试失败: {e}")

# 3. 测试ASR模型
print("\n=== 测试ASR模型 ===")
try:
    from perception.asr.funasr_asr import FunASRRecognizer
    
    config = config_loader.get_config()
    asr_config = config.get("models", {}).get("asr", {})
    
    print_memory_usage("加载ASR前")
    asr = FunASRRecognizer(asr_config)
    print_memory_usage("加载ASR后")
    
    # 测试推理
    test_audio = np.random.randn(16000).astype(np.float32)
    print_memory_usage("ASR推理前")
    wake_detected, asr_text = asr.inference(test_audio)
    print_memory_usage("ASR推理后")
    
    asr.release()
    print_memory_usage("释放ASR后")
except Exception as e:
    print(f"ASR测试失败: {e}")

# 4. 测试LLM模型
print("\n=== 测试LLM模型 ===")
try:
    from perception.llm.qwen_multimodal import QwenMultimodal
    
    config = config_loader.get_config()
    llm_config = config.get("models", {}).get("llm", {})
    
    print_memory_usage("加载LLM前")
    llm = QwenMultimodal(llm_config)
    print_memory_usage("加载LLM后")
    
    # 测试推理
    from PIL import Image
    test_image = Image.new('RGB', (640, 480), color='white')
    test_metadata = {}
    test_prompt = "请分析当前场景"
    print_memory_usage("LLM推理前")
    response = llm.inference((test_image, test_metadata, test_prompt))
    print_memory_usage("LLM推理后")
    
    llm.release()
    print_memory_usage("释放LLM后")
except Exception as e:
    print(f"LLM测试失败: {e}")

# 测试所有模型同时加载
print("\n=== 测试所有模型同时加载 ===")
try:
    from perception.yolo.yolo_detector import YoloDetector
    from perception.vda.vda_depth import VDADepthEstimator
    from perception.asr.funasr_asr import FunASRRecognizer
    from perception.llm.qwen_multimodal import QwenMultimodal
    
    config = config_loader.get_config()
    
    print_memory_usage("加载所有模型前")
    
    # 加载YOLO
    yolo_config = config.get("models", {}).get("yolo", {})
    yolo = YoloDetector(yolo_config)
    print_memory_usage("加载YOLO后")
    
    # 加载VDA
    vda_config = config.get("models", {}).get("vda", {})
    vda = VDADepthEstimator(vda_config)
    print_memory_usage("加载VDA后")
    
    # 加载ASR
    asr_config = config.get("models", {}).get("asr", {})
    asr = FunASRRecognizer(asr_config)
    print_memory_usage("加载ASR后")
    
    # 加载LLM
    llm_config = config.get("models", {}).get("llm", {})
    llm = QwenMultimodal(llm_config)
    print_memory_usage("加载LLM后")
    
    # 释放所有模型
    yolo.release()
    vda.release()
    asr.release()
    llm.release()
    print_memory_usage("释放所有模型后")
except Exception as e:
    print(f"所有模型同时加载测试失败: {e}")

# 清理
nvmlShutdown()
print("\n测试完成！")
