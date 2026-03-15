#!/usr/bin/env python3
"""
测试摄像头模拟器的运行状态
"""

import time
import sys
import os

# 添加系统路径
sys.path.append('/root/MHSEE')

from simulation.camera_simulator import CameraSimulator

# 测试摄像头模拟器
def test_camera():
    print("开始测试摄像头模拟器...")
    
    # 创建摄像头模拟器
    camera_simulator = CameraSimulator()
    
    # 启动模拟器
    camera_simulator.start()
    print("摄像头模拟器已启动")
    
    # 等待一段时间，让模拟器初始化
    time.sleep(5)  # 增加等待时间
    
    # 测试获取所有帧
    print("\n测试获取所有帧:")
    frames = camera_simulator.get_all_frames()
    print(f"获取到 {len(frames)} 个摄像头的帧")
    
    for camera_id, (frame, timestamp) in frames.items():
        print(f"摄像头 {camera_id}:")
        print(f"  帧是否为None: {frame is None}")
        if frame is not None:
            print(f"  帧形状: {frame.shape}")
        print(f"  时间戳: {timestamp}")
    
    # 测试获取音频
    print("\n测试获取音频:")
    audio = camera_simulator.get_audio("camera1")
    print(f"音频是否为None: {audio[0] is None}")
    if audio[0] is not None:
        print(f"音频长度: {len(audio[0])}")
    print(f"音频时间戳: {audio[1]}")
    
    # 测试连续获取帧
    print("\n测试连续获取帧:")
    for i in range(10):  # 增加测试次数
        frames = camera_simulator.get_all_frames()
        for camera_id, (frame, timestamp) in frames.items():
            if frame is not None:
                print(f"第 {i+1} 帧 - 摄像头 {camera_id}: 帧形状={frame.shape}, 时间戳={timestamp}")
            else:
                print(f"第 {i+1} 帧 - 摄像头 {camera_id}: 帧为None")
        time.sleep(0.5)
    
    # 停止模拟器
    camera_simulator.stop()
    print("\n摄像头模拟器已停止")

if __name__ == "__main__":
    test_camera()
