from modelscope.hub.snapshot_download import snapshot_download
import os

# ===================== 核心：强制下载到autodl-tmp（避免系统盘满）=====================
# 模型缓存根目录（AutoDL高速盘）
MODEL_ROOT = "/root/autodl-tmp/funasr_models"
CACHE_DIR = os.path.join(MODEL_ROOT, "modelscope_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"📂 模型将下载到：{CACHE_DIR}\n")

# ===================== 下载你提供的3个有效iic模型 =====================
def download_model(model_id, revision=None):
    """下载指定ModelScope模型，返回本地路径"""
    print(f"🔽 开始下载：{model_id}（版本：{revision or '默认'}）")
    try:
        model_dir = snapshot_download(
            model_id=model_id,
            revision=revision,
            cache_dir=CACHE_DIR,
            ignore_file_pattern=["*.md", "*.png", "example/*"]  # 跳过无用文件，节省空间
        )
        print(f"✅ 下载完成 → {model_dir}\n")
        return model_dir
    except Exception as e:
        print(f"❌ 下载失败：{e}\n")
        return None

# 1. VAD断句模型（有效ID+指定版本）
vad_dir = download_model(
    model_id="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    revision="v1.0.0"
)

# 2. 标点恢复模型（有效ID+指定版本）
punc_dir = download_model(
    model_id="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    revision="v1.0.0"
)

# 3. SenseVoiceSmall模型（有效ID）
sensevoice_dir = download_model(
    model_id="iic/SenseVoiceSmall"
)

# 输出最终路径（供GPU加载用）
print("📌 模型本地路径汇总（GPU加载时直接用）：")
print(f"VAD模型：{vad_dir}")
print(f"标点模型：{punc_dir}")
print(f"SenseVoiceSmall：{sensevoice_dir}")
print("\n🎉 所有模型下载完成！无卡模式安全，切GPU后运行加载脚本即可")