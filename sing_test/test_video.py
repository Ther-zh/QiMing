#!/usr/bin/env python3
"""
测试视频文件是否可以被 OpenCV 打开和读取
"""

import cv2
import os

# 测试视频文件
video_path = "video/video.mp4"

print(f"测试视频文件: {video_path}")
print(f"文件是否存在: {os.path.exists(video_path)}")

# 尝试打开视频文件
cap = cv2.VideoCapture(video_path)

print(f"视频是否成功打开: {cap.isOpened()}")

if cap.isOpened():
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    print(f"视频帧率: {fps}")
    print(f"视频宽度: {width}")
    print(f"视频高度: {height}")
    print(f"视频总帧数: {frame_count}")
    
    # 尝试读取一帧
    ret, frame = cap.read()
    print(f"是否成功读取第一帧: {ret}")
    if ret:
        print(f"第一帧形状: {frame.shape}")
    
    # 释放视频捕获
    cap.release()
else:
    print("无法打开视频文件")
