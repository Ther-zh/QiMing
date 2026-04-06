#!/usr/bin/env python3
"""若 perception/yolo/model/yolov8n.pt 为 Git LFS 指针，则下载真实权重（约 6MB）。"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET = os.path.join(ROOT, "perception", "yolo", "model", "yolov8n.pt")
LFS_MARK = b"git-lfs.github.com"


def main() -> int:
    need = True
    if os.path.isfile(TARGET):
        with open(TARGET, "rb") as f:
            head = f.read(200)
        if LFS_MARK not in head and os.path.getsize(TARGET) > 100_000:
            print(f"[ensure_yolo_weights] 已存在有效权重: {TARGET}")
            return 0
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    from ultralytics import YOLO

    print("[ensure_yolo_weights] 正在从 Ultralytics 拉取 yolov8n.pt …")
    model = YOLO("yolov8n.pt")
    ckpt = getattr(model, "ckpt_path", None) or "yolov8n.pt"
    if not os.path.isfile(ckpt):
        print(f"[ensure_yolo_weights] 失败: 未找到下载文件 {ckpt}")
        return 1
    import shutil

    shutil.copy2(ckpt, TARGET)
    print(f"[ensure_yolo_weights] 已写入: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
