Whisper ggml 模型目录（ggml-*.bin）。

边缘设备移植时请阅读上级目录的 EDGE_DEPLOY.txt（需复制哪些文件、whisper-cli 与依赖）。

模型选择（中文语音）：
- tiny / tiny-q8_0：体积最小，中文容易错字、幻觉，仅适合连通性测试。
- small-q5_1（推荐默认）：量化 small，中文明显好于 tiny，体积约 180MB，CPU 可接受。
- base / small / medium 未量化：效果更好，体积与内存更大；下载名如 small、medium 等。

下载（默认 HuggingFace 镜像 hf-mirror.com）：
  bash /root/Wasr/scripts/download_ggml_model.sh small-q5_1 /root/Wasr/models
  bash /root/Wasr/scripts/download_ggml_model.sh tiny-q8_0 /root/Wasr/models
直连官方：
  HF_MIRROR=https://huggingface.co bash /root/Wasr/scripts/download_ggml_model.sh small-q5_1

Python（conda 环境 vda）：
  conda activate vda
  pip install -r /root/Wasr/requirements.txt

MHSEE：`models.asr.language` 必须与视频语言一致（中文用 `zh`）。单测里 JFK 样例使用英文 `en`，若误用会输出英文，属正常。

自检：
  PYTHONPATH=/root python /root/Wasr/scripts/benchmark_whisper_jfk.py /root/Wasr/models/ggml-tiny-q8_0.bin

单模块测试：
  PYTHONPATH=/root python /root/Wasr/tests/test_wasr_modules.py
  PYTHONPATH=/root python /root/Wasr/tests/test_wasr_modules.py --skip-whisper
