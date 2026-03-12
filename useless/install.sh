#!/bin/bash

# ================= 配置区域 =================
TARGET_DIR="/root/autodl-tmp/qwen35"
MODEL_ID="tclf90/Qwen3.5-4B-AWQ" # 你指定的ModelScope模型ID
# ===========================================

echo ">>> 开始在 $TARGET_DIR 部署 Qwen3.5-4B-AWQ..."

# 1. 创建目录并进入
mkdir -p $TARGET_DIR
cd $TARGET_DIR

# 2. 检查 Python 环境 (AutoDL通常自带3.8-3.10)
python --version

# 3. 安装核心依赖
# 注意：AutoAWQ 需要与 CUDA 版本匹配，AutoDL 环境通常能直接适配
echo ">>> 正在安装 Python 依赖库 (AutoAWQ, ModelScope, Transformers)..."
pip install --upgrade pip
pip install autoawq modelscope transformers accelerate sentencepiece protobuf

# 4. 从 ModelScope 下载模型 (使用 snapshot_download 速度更快)
echo ">>> 正在从 ModelScope 下载模型 (约 3-5GB)..."
python << 'EOF'
from modelscope import snapshot_download
import os

model_dir = snapshot_download(
    'tclf90/Qwen3.5-4B-AWQ',
    cache_dir='/root/autodl-tmp/qwen35',
    revision='master'
)
# 由于 snapshot_download 会在 cache_dir 下生成嵌套目录，我们把路径写死方便后续调用
# 实际上 AutoAWQ 可以直接读取 model_id
print(f"模型下载完成，缓存路径: {model_dir}")
EOF

# 5. 生成一个简单的测试脚本
echo ">>> 生成推理测试脚本 test_inference.py..."
cat > $TARGET_DIR/test_inference.py << 'EOF'
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 模型路径 (如果上面下载的路径不对，请手动修改这里)
# 注意：AutoAWQ 也支持直接传 ModelScope ID: "tclf90/Qwen3.5-4B-AWQ"
model_name_or_path = "tclf90/Qwen3.5-4B-AWQ"

print(f"正在加载模型: {model_name_or_path} ...")

# 加载模型和分词器
model = AutoAWQForCausalLM.from_quantized(
    model_name_or_path, 
    trust_remote_code=True, 
    device_map="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

# 简单的对话测试
prompt = "你好，请介绍一下你自己。"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

# 生成回复
print("正在生成回复...")
outputs = model.generate(
    **inputs, 
    max_new_tokens=512, 
    temperature=0.7, 
    top_p=0.95
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("-" * 30)
print(f"AI回复: {response}")
print("-" * 30)
EOF

echo ""
echo "=============================================="
echo "✅ 安装完成！"
echo "目录位置: $TARGET_DIR"
echo ""
echo "下一步操作："
echo "1. 运行测试: cd $TARGET_DIR && python test_inference.py"
echo "=============================================="