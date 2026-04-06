#!/bin/bash
#
# QiMing / MHSEE 导盲系统 — Jetson (aarch64) 依赖安装辅助脚本
#
# 重要：请勿在此脚本中从 PyPI 安装 torch / torchvision / torchaudio。
# 在 JetPack 上错误版本会卸载厂商 CUDA 版 PyTorch，导致 GPU 推理失效。
# 安装顺序与自检命令见仓库根目录 JETSON_PYTORCH.md 与：
#   https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
#
# 用法（在已按文档装好 torch 三件套之后）：
#   bash deploy_jetson.sh
#

set -euo pipefail

echo "========================================"
echo "QiMing 环境部署（Jetson / conda）"
echo "========================================"

if ! command -v conda &> /dev/null; then
    echo "错误: 未检测到 conda，请先安装 Miniconda/Miniforge。"
    exit 1
fi

ENV_NAME="${CONDA_DEFAULT_ENV:-mhsee}"

if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "创建 conda 环境: $ENV_NAME (Python 3.10)"
    conda create -n "$ENV_NAME" python=3.10 -y
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "当前环境: $ENV_NAME"
echo "L4T 版本（供对照 NVIDIA 文档）:"
cat /etc/nv_tegra_release 2>/dev/null || echo "（非 Jetson 或未找到 nv_tegra_release）"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_JETSON="${SCRIPT_DIR}/requirements-jetson.txt"

if [[ ! -f "$REQ_JETSON" ]]; then
    echo "错误: 未找到 $REQ_JETSON"
    exit 1
fi

echo ""
echo ">>> 步骤 1/3：若尚未安装厂商 PyTorch，请先阅读并执行 JETSON_PYTORCH.md（勿跳过）。"
echo ">>> 步骤 2/3：安装项目其余依赖（不含 torch 三件套）..."
pip install --upgrade pip
pip install --no-cache-dir -r "$REQ_JETSON"

echo ""
echo ">>> 步骤 3/3：自检（torch 须为文档给出的 CUDA 构建）"
python - <<'PY'
import sys
try:
    import torch
    print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch.version.cuda:", torch.version.cuda)
except Exception as e:
    print("torch 导入失败:", e)
    sys.exit(1)
for name in ("torchvision", "torchaudio"):
    try:
        m = __import__(name)
        print(name + ":", getattr(m, "__version__", "?"))
    except Exception as e:
        print(name + ": 未安装或导入失败（Jetson 上可能为预期，见 JETSON_PYTORCH.md）->", e)
PY

echo ""
echo "========================================"
echo "pip 包检查（torch* 以导入自检为准，不要求 torchvision/torchaudio 必装）"
echo "========================================"
for pkg in numpy opencv-python PyYAML psutil ultralytics funasr Pillow protobuf sympy ollama; do
    if pip show "$pkg" &>/dev/null; then
        echo "✓ $pkg"
    else
        echo "✗ $pkg 未安装"
    fi
done

echo "========================================"
echo "完成。激活环境: conda activate $ENV_NAME"
echo "多模态 LLM（Ollama）可用 Docker：见 docker/ollama/README.md"
echo "========================================"
