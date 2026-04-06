#!/usr/bin/env bash
# 将 VDA vits 权重下载到大容量盘（符合 autodl-tmp 约定），避免仓库内空占位文件。
set -euo pipefail
DEST="${1:-/home/nvidia/models/root/autodl-tmp/vda_checkpoints/video_depth_anything_vits.pth}"
URL="https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth"
mkdir -p "$(dirname "$DEST")"
echo "下载到: $DEST"
curl -fL --retry 5 --retry-delay 10 --connect-timeout 30 -o "$DEST.partial" "$URL"
mv -f "$DEST.partial" "$DEST"
ls -la "$DEST"
echo "请在 config/config.yaml 的 models.vda.checkpoint_path 指向上述路径。"
