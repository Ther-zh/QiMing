#!/bin/bash
# Qwen3.5 模型 Web 服务启动脚本
# 适配 Swift 工具，支持纯文本/Qwen3.5-VL 多模态版本

# ======================== 配置区 (根据你的实际情况修改) ========================
# 1. 模型路径 (替换成你实际的 Qwen3.5 模型目录)
MODEL_PATH="/root/autodl-tmp/qwen35/tclf90/Qwen3___5-4B-AWQ"

# 2. 模型类型 (二选一)
#    - 纯文本 Qwen3.5: qwen3
#    - 多模态 Qwen3.5-VL: qwen3_vl
MODEL_TYPE="qwen3_vl"

# 3. GPU 配置 (单卡设为 0，多卡设为 0,1 等)
CUDA_VISIBLE_DEVICES="0"

# 4. 生成配置
MAX_NEW_TOKENS="2048"  # 最大生成token数
TEMPERATURE="0.7"      # 生成温度 (0-1，越低越稳定)
TOP_P="0.9"            # 采样Top-P

# 5. 多模态配置 (仅 MODEL_TYPE=qwen3_vl 时生效)
MAX_PIXELS="1003520"   # 图片最大像素数
VIDEO_MAX_PIXELS="50176" # 视频单帧像素数
FPS_MAX_FRAMES="12"    # 视频最大处理帧数
# =============================================================================

# 打印配置信息
echo "========================================"
echo "          Qwen3.5 服务启动配置          "
echo "========================================"
echo "模型路径: $MODEL_PATH"
echo "模型类型: $MODEL_TYPE"
echo "使用GPU:  $CUDA_VISIBLE_DEVICES"
echo "最大生成token: $MAX_NEW_TOKENS"
echo "========================================"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export MAX_PIXELS=$MAX_PIXELS
export VIDEO_MAX_PIXELS=$VIDEO_MAX_PIXELS
export FPS_MAX_FRAMES=$FPS_MAX_FRAMES

# 启动 Swift Web 服务
swift app \
    --model $MODEL_PATH \
    --infer_backend pt \          # 推理后端：PyTorch (如果想用vLLM可改为 vllm)
    --model_type $MODEL_TYPE \
    --lang zh \                   # 界面语言：中文
    --studio_title "Qwen3.5-Instruct" \  # Web界面标题
    --attn_impl flash_attention_2 \      # FlashAttention2 加速
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --load_in_4bit True \         # 适配 AWQ 量化模型 (如果是非量化版可注释)
    --trust_remote_code True      # 加载自定义模型代码 (Qwen必备)

# 脚本说明：
# 1. 执行前先赋予执行权限：chmod +x run_qwen35.sh
# 2. 启动脚本：./run_qwen35.sh
# 3. 启动后浏览器访问 http://localhost:7860 即可使用 Web 界面