#!/bin/bash

# 导盲系统环境部署脚本
# 适用于NVIDIA Jetson Nano Super (ARM64架构)

echo "========================================"
echo "导盲系统环境部署脚本"
echo "适用于NVIDIA Jetson Nano Super (ARM64)"
echo "========================================"

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未检测到conda，请先安装Anaconda或Miniconda"
    exit 1
fi

# 环境名称
ENV_NAME="mhsee"

# 检查环境是否存在
if conda env list | grep -q "^$ENV_NAME"; then
    echo "环境 $ENV_NAME 已存在，将更新依赖"
else
    echo "创建新环境: $ENV_NAME"
    conda create -n $ENV_NAME python=3.10 -y
fi

# 激活环境
echo "激活环境: $ENV_NAME"
source activate $ENV_NAME

# 升级pip并设置国内源
echo "升级pip并设置国内源"
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 设置pip默认源
echo "设置pip默认源为清华源"
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装核心依赖
echo "安装核心依赖"
pip install numpy==2.2.6
pip install opencv-python==4.11.0.86
pip install PyYAML==6.0.1
pip install psutil==7.2.2

# 安装模型依赖
echo "安装模型依赖"
pip install ultralytics==8.4.21
pip install funasr==1.3.1

# 安装Jetson Nano专用的PyTorch (ARM64架构)
echo "安装Jetson Nano专用的PyTorch (ARM64)"
# 下载并安装适合Jetson Nano的PyTorch版本
echo "正在下载PyTorch for Jetson Nano..."
wget -q https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i09j56znk96n5ia1.whl -O torch-1.13.0-cp310-cp310-linux_aarch64.whl
pip install torch-1.13.0-cp310-cp310-linux_aarch64.whl

# 安装torchvision
echo "安装torchvision"
pip install torchvision==0.14.0

# 安装torchaudio
echo "安装torchaudio"
pip install torchaudio==0.13.0

# 清理临时文件
rm -f torch-1.13.0-cp310-cp310-linux_aarch64.whl

# 安装其他依赖
echo "安装其他依赖"
pip install Pillow==12.1.1
pip install requests==2.32.5
pip install tqdm==4.67.3
pip install librosa==0.11.0
pip install soundfile==0.13.1
pip install pydub==0.25.1
pip install protobuf==6.33.5
pip install sympy==1.14.0

# 安装额外依赖
echo "安装额外依赖"
pip install transformers==4.57.6
pip install huggingface_hub==0.36.2
pip install safetensors==0.7.0
pip install scikit-learn==1.7.2
pip install scipy==1.15.3
pip install pandas==2.3.3
pip install matplotlib==3.10.8

# 安装网络和API相关依赖
echo "安装网络和API相关依赖"
pip install fastapi==0.134.0
pip install uvicorn==0.41.0
pip install httpx==0.28.1

# 安装工具库
echo "安装工具库"
pip install loguru==0.7.3
pip install python-dotenv==1.2.2

# 验证安装
echo "========================================"
echo "验证安装情况"
echo "========================================"

# 检查核心包是否安装成功
packages=("numpy" "opencv-python" "PyYAML" "psutil" "ultralytics" "funasr" "Pillow" "torch" "torchvision" "torchaudio" "protobuf" "sympy")

for pkg in "${packages[@]}"; do
    if pip show $pkg &> /dev/null; then
        echo "✓ $pkg 已安装"
    else
        echo "✗ $pkg 安装失败"
    fi
done

echo "========================================"
echo "环境部署完成！"
echo "使用以下命令激活环境："
echo "conda activate $ENV_NAME"
echo "========================================"
