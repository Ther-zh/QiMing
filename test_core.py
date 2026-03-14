#!/usr/bin/env python3
"""
测试系统核心功能（简化版）
"""

import os
import sys
import time
import numpy as np
from PIL import Image

# 添加系统路径
sys.path.append('/root/MHSEE')

# 测试系统核心功能
def test_core():
    print("开始测试系统核心功能...")
    
    # 加载配置
    from utils.config_loader import config_loader
    config = config_loader.get_config()
    
    print("配置加载完成")
    print(f"YOLO模型路径: {config.get('models', {}).get('yolo', {}).get('model_path')}")
    print(f"VDA模型路径: {config.get('models', {}).get('vda', {}).get('model_path')}")
    print(f"ASR模型路径: {config.get('models', {}).get('asr', {}).get('model_path')}")
    print(f"LLM模型路径: {config.get('models', {}).get('llm', {}).get('model_path')}")
    
    # 初始化模拟模块
    from simulation.camera_simulator import CameraSimulator
    camera_simulator = CameraSimulator()
    camera_simulator.start()
    print("摄像头模拟器启动完成")
    
    # 初始化感知模块
    from perception.yolo.yolo_detector import YoloDetector
    from perception.vda.vda_depth import VDADepthEstimator
    from perception.asr.funasr_asr import FunASRRecognizer
    from perception.llm.qwen_multimodal import QwenMultimodal
    
    print("正在加载 YOLO 模型...")
    yolo = YoloDetector(config.get("models", {}).get("yolo", {}))
    print("YOLO 模型加载完成")
    
    print("正在加载 VDA 模型...")
    vda = VDADepthEstimator(config.get("models", {}).get("vda", {}))
    print("VDA 模型加载完成")
    
    print("正在加载 ASR 模型...")
    asr = FunASRRecognizer(config.get("models", {}).get("asr", {}))
    print("ASR 模型加载完成")
    
    print("正在加载 LLM 模型...")
    llm = QwenMultimodal(config.get("models", {}).get("llm", {}))
    print("LLM 模型加载完成")
    
    # 初始化融合模块
    from fusion.frame_sync import FrameSync
    from fusion.depth_fusion import DepthFusion
    from fusion.target_tracker import TargetTracker
    from fusion.metadata_wrapper import MetadataWrapper
    
    frame_sync = FrameSync()
    depth_fusion = DepthFusion()
    target_tracker = TargetTracker()
    metadata_wrapper = MetadataWrapper()
    print("融合模块初始化完成")
    
    # 初始化核心调度模块
    from core.realtime_scheduler import RealtimeScheduler
    from core.complex_scene_scheduler import ComplexSceneScheduler
    from core.resource_manager import ResourceManager
    
    resource_manager = ResourceManager()
    realtime_scheduler = RealtimeScheduler()
    complex_scene_scheduler = ComplexSceneScheduler(resource_manager)
    print("核心调度模块初始化完成")
    
    # 初始化执行模块
    from execution.broadcast_scheduler import BroadcastScheduler
    from execution.tts_engine import TTSEngine
    
    broadcast_scheduler = BroadcastScheduler()
    tts_engine = TTSEngine(config.get("execution", {}))
    print("执行模块初始化完成")
    
    # 启动各个模块
    resource_manager.start()
    frame_sync.start()
    realtime_scheduler.start()
    complex_scene_scheduler.start()
    broadcast_scheduler.start()
    print("所有模块启动完成")
    
    # 测试主循环
    for i in range(3):  # 测试3帧
        print(f"\n=== 测试第 {i+1} 帧 ===")
        
        # 获取摄像头帧
        frames = camera_simulator.get_all_frames()
        main_camera = "camera1"
        
        if main_camera in frames:
            frame, timestamp = frames[main_camera]
            if frame is not None:
                print(f"获取到帧，时间戳: {timestamp}")
                
                # 添加帧到同步模块
                frame_sync.add_frame(frame, timestamp, 0)
                
                # 执行YOLO检测
                start_time = time.time()
                yolo_results = yolo.inference(frame)
                print(f"YOLO检测完成，耗时: {time.time() - start_time:.2f}s")
                print(f"检测到 {len(yolo_results)} 个目标")
                frame_sync.add_yolo_result(yolo_results, timestamp)
                
                # 执行VDA深度估计
                start_time = time.time()
                depth_map = vda.inference(frame)
                print(f"VDA深度估计完成，耗时: {time.time() - start_time:.2f}s")
                frame_sync.add_vda_result(depth_map, timestamp)
                
                # 获取同步数据
                sync_data = frame_sync.get_sync_data()
                if sync_data:
                    sync_frame, sync_yolo, sync_depth, sync_timestamp, sync_camera_id = sync_data
                    
                    # 计算目标距离
                    targets_with_distance = depth_fusion.calculate_target_distances(sync_yolo, sync_depth)
                    print(f"计算目标距离完成，共 {len(targets_with_distance)} 个目标")
                    
                    # 跟踪目标并计算速度
                    tracked_targets = target_tracker.track_targets(targets_with_distance, sync_timestamp)
                    print(f"目标跟踪完成，共 {len(tracked_targets)} 个跟踪目标")
                    
                    # 封装元数据
                    metadata = metadata_wrapper.wrap_metadata(
                        sync_frame,
                        tracked_targets,
                        sync_timestamp,
                        sync_camera_id
                    )
                    print("元数据封装完成")
                    
                    # 处理实时安全调度
                    realtime_scheduler.process_metadata(metadata)
                    
                    # 检查是否有告警
                    alert = realtime_scheduler.get_alert()
                    if alert:
                        print(f"告警: {alert.get('message')}")
                        # 添加到语音播报队列
                        broadcast_scheduler.add_message(
                            alert.get("message"),
                            priority=1 if alert.get("level") == "level1" else 2,
                            alert_type=alert.get("level")
                        )
                    
                    # 生成测试音频数据
                    audio_data = np.random.randn(16000).astype(np.float32)
                    print(f"生成测试音频数据，长度: {len(audio_data)}")
                    # 执行ASR识别
                    start_time = time.time()
                    wake_detected, asr_text = asr.inference(audio_data)
                    print(f"ASR识别完成，耗时: {time.time() - start_time:.2f}s")
                    print(f"ASR结果: {asr_text}")
                    print(f"唤醒词检测: {wake_detected}")
                    
                    # 如果检测到唤醒词，调用LLM处理
                    if wake_detected:
                        print("检测到唤醒词，调用LLM处理...")
                        # 将NumPy数组转换为PIL Image
                        sync_image = Image.fromarray(sync_frame)
                        # 调用复杂场景调度器处理
                        start_time = time.time()
                        response = complex_scene_scheduler.handle_wake_word(
                            asr_text,
                            sync_image,
                            metadata
                        )
                        print(f"LLM处理完成，耗时: {time.time() - start_time:.2f}s")
                        print(f"LLM回复: {response}")
                        # 添加到语音播报队列
                        broadcast_scheduler.add_message(
                            response,
                            priority=3,
                            alert_type="wake_word"
                        )
        
        # 控制帧率
        time.sleep(0.5)  # 2FPS
    
    # 停止各个模块
    print("\n测试完成，停止系统...")
    broadcast_scheduler.stop()
    complex_scene_scheduler.stop()
    realtime_scheduler.stop()
    frame_sync.stop()
    camera_simulator.stop()
    resource_manager.stop()
    
    # 释放模型资源
    yolo.release()
    vda.release()
    asr.release()
    llm.release()
    tts_engine.release()
    
    print("系统测试完成")

if __name__ == "__main__":
    test_core()
