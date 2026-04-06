# Jetson 上安装 PyTorch / torchvision / torchaudio

在 NVIDIA Jetson（aarch64、L4T/JetPack）上，**不要**用 PyPI 默认源安装 `torch`、`torchvision`、`torchaudio`，否则 `pip` 可能卸载厂商提供的 CUDA 版 `torch`，换成不含 SM87 内核或与 JetPack 不匹配的轮子，导致 YOLO 等推理报错或仅能 CPU 运行。

## 官方文档（以当前 JetPack 为准）

- [Installing PyTorch for Jetson Platform](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)

请按文档选择与你的 **JetPack / L4T 版本**、**Python 小版本** 对应的 wheel 或 `pip` 命令；**torch、torchvision、torchaudio 须为同一文档给出的配套版本**。

## 推荐安装顺序

1. 确认 L4T：`cat /etc/nv_tegra_release`
2. 按上述文档安装 **torch**（及文档要求的前置包，如 cuSPARSELt 等）
3. 再按同一文档安装 **torchvision**、**torchaudio**（勿单独从 PyPI 装与当前 `torch` 不匹配的版本）
4. 使用 `pip install --no-cache-dir` 可减少坏缓存；若使用国内镜像，需确认镜像中是否包含与文档一致的 NVIDIA Jetson wheel，否则请直连文档中的下载源

## 与本项目依赖的配合

- 先完成官方三件套安装，再执行：
  - `pip install -r requirements-jetson.txt`
- 根目录 [requirements.txt](requirements.txt) 已**不再**列出 `torch*` 固定版本，避免误装 PyPI 桌面栈。

## FunASR / ModelScope 仅用本机已下载模型（不触发在线下载）

1. 缓存根目录设为：`MODELSCOPE_CACHE=/home/nvidia/models/root/autodl-tmp/funasr_models/modelscope_cache`（与 [config/config.yaml](config/config.yaml) 中 ASR 路径一致）。
2. ModelScope 期望布局为 `MODELSCOPE_CACHE/models/iic/...`。若你的目录是 `MODELSCOPE_CACHE/iic/...`，请建立符号链接：  
   `mkdir -p "$MODELSCOPE_CACHE/models" && ln -sfn ../iic "$MODELSCOPE_CACHE/models/iic"`  
   （项目已在该路径下创建过一次，可检查 `models/iic` 是否存在。）
3. 在加载 FunASR 前设置 **`FUNASR_LOCAL_ONLY=1`**（已在 [perception/asr/funasr_asr.py](perception/asr/funasr_asr.py) 中 `setdefault`），`funasr` 内 `snapshot_download` 会使用 `local_files_only=True`，**不会**再拉取新权重。
4. **已卸载** 与厂商 `torch` ABI 不匹配的 PyPI **`torchaudio`**；当前通过 `funasr` 自带补丁（`load_utils`、`wav_frontend` 的 `kaldi_compat`、`campplus/utils`）在无 `torchaudio` 时尽量工作。升级 `funasr` 后需重新评估这些补丁。
5. 若 `pip` 曾提示 `invalid distribution -orch`，已删除 `site-packages` 下以 `~` 开头的错误备份目录；勿再用手动重命名 `torch` 目录的方式“备份”包。

## mhsee / 厂商 torch 下 YOLO 跑通说明（本机已验证）

若环境中曾出现 **伪造的 `torchvision`（仅含假 `ops.nms` 的 `__init__.py`）**，ultralytics 会在 NMS 处报错 `ValueError: step must be greater than zero`。若从 PyPI 安装正式 `torchvision`，又可能因厂商 `torch` 未注册 `torchvision::nms` 等算子而报 `RuntimeError: operator torchvision::nms does not exist`。

**可行做法（不依赖可用的 torchvision wheel）**：

1. **卸载** `torchvision`：`pip uninstall -y torchvision`（并删除残留 `site-packages/torchvision*`）。
2. **修改** 当前环境内 `ultralytics` 两处（升级 ultralytics 后需重做）：
   - `ultralytics/utils/__init__.py`：对 `importlib.metadata.version("torchvision")` 使用 `try/except PackageNotFoundError`，失败时将 `TORCHVISION_VERSION` 设为 `"0.20.0"`（仅为通过版本检查；实际 NMS 走内置 **TorchNMS**）。
   - `ultralytics/models/__init__.py`：对 **SAM / FastSAM** 做**延迟导入**（`__getattr__`），避免 `from ultralytics import YOLO` 时加载依赖 `torchvision` 的 SAM 子模块。

完成后，NMS 使用 ultralytics 自带实现，**无需**与厂商 `torch` 匹配的 C++ `torchvision` 扩展。若日后接入 **Video-Depth-Anything** 等需要 `torchvision.transforms` 的代码，再另寻与当前 `torch` ABI 一致的官方 wheel 或从源码编译。

## 安装后自检

```bash
conda activate mhsee   # 或你的环境名
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)"
python sing_test/verify_torch_stack.py   # 打印 torchvision / torchaudio 状态（仓库内小脚本）
```

若 `site-packages` 下仅有 **残缺 torchvision 目录**（能 import 但无 `__version__`、无 `ops`），请 **`pip uninstall -y torchvision` 并删净残留目录**，以免误导依赖检测。

若 `torchaudio` 报 `undefined symbol` 等与 `torch` 相关的动态链接错误，说明 **torchaudio 与当前 torch 非同一构建批次**，请卸载后按 NVIDIA 文档重装与 torch 匹配的 `torchaudio`。

## YOLO 权重（Git LFS 指针）

若 `perception/yolo/model/yolov8n.pt` 只有约一百字节且内容含 `git-lfs`，加载会报 `invalid load key`。在项目根执行：

```bash
conda activate mhsee
python sing_test/ensure_yolo_weights.py
```

## 分模块视频调试中的 YOLO

[`sing_test/module_debug.py`](sing_test/module_debug.py) 对视频推理使用 **`stream=True`**，避免长视频在内存中堆积全部 `Results` 导致 Jetson OOM 被系统 kill。

## 本仓库一次实测记录（仅供参考）

| 项目 | 值 |
|------|-----|
| 日期 | 2026-04-03 |
| L4T | R36 (REVISION: 4.3)，JetPack 6 系 |
| Python | 3.10（conda-forge 环境 mhsee） |
| torch | 2.5.0a0+872d972e41.nv24.08，`cuda=True`，`torch.version.cuda` 12.6 |
| torchvision | 无有效 pip 包时可接受；ultralytics 8.4+ 对 `PackageNotFoundError` 走 TorchNMS；勿装与厂商 torch 不匹配的 PyPI `torchvision` |
| torchaudio | 未安装或为预期；FunASR 走无 torchaudio 补丁路径（见上文） |
| YOLO 视频验收 | `python sing_test/module_debug.py --module yolo`：`cuda` 可用时为 GPU；已用 `stream=True` |

大模型若通过 **Ollama** 运行，其 GPU 路径独立于本环境内的 Python `torch`；若使用 **Transformers + Qwen** 等路径，则与当前 `torch` 强相关。Ollama 容器说明见 [`docker/ollama/README.md`](docker/ollama/README.md)。

**资源摘要里的「GPU 显存」**：在 Orin 上 NVML 常读失败时，日志中的 CUDA 数字来自 **本进程 PyTorch 的 `memory_allocated`**，不是 `nvidia-smi` 式整卡占用，也不含 **ollama**。启停时若开启 `resources.show_tegrastats_in_summary`，会多一行 **`tegrastats` 采样**（RAM、GR3D 等），更接近整机观感；平时也可用 `tegrastats` / `jtop` 自行观察。

## LLM / Ollama 调试检查表（QiMing）

1. **区分真输出与代码兜底**  
   - 控制台或日志中搜索 `[LLM_META]`：`event='ok'` 为正常；`fallback_empty` / `fallback_exception` 表示走了 [LLM/qwen35.py](LLM/qwen35.py) 的兜底话术（默认与 [config/config.yaml](config/config.yaml) 中 `models.llm.fallback_phrase` 一致）。  
   - 若连续多条用户可见回复与 `fallback_phrase` 完全相同，优先查 `LLM_META` 而非猜测模型「变笨」。

2. **模型是否支持多模态**  
   - `ollama show <model_name>`，确认 families/capabilities 含 vision/VL 语义；否则 `generate(..., image=...)` 会异常或空内容。  
   - 快速探针：`python sing_test/llm_context_probe.py`（短/长 prompt，观察 `[LLM_META]` 与回复长度）。

3. **`num_ctx` 与长 prompt**  
   - 主系统唤醒词 prompt 较长（见 [core/complex_scene_scheduler.py](core/complex_scene_scheduler.py)）。若 `fallback_empty` 且 `raw_len=0`，在 [config/config.yaml](config/config.yaml) 的 `models.llm.ollama_options` 中逐步提高 `num_ctx`（如 512→1024），并注意 Jetson **统一内存 OOM**；用 `llm_context_probe` 做前后对比。

4. **Think 模式**  
   - Qwen3.5 须 `ollama_think: false` 且**不要**把 `think` 放进 Ollama `options`（见代码注释 ollama#14793）。

5. **分模块视频调试与内存**  
   - `python sing_test/module_debug.py --module all --vda-max-frames 30`：`all` 默认用 **四个子进程** 顺序跑 **VDA → ASR → YOLO → LLM**，每段退出后释放 PyTorch 占用的统一内存，再启动 Ollama，显著降低被 kill 的概率；单进程调试加 `--single-process`。  
   - 仍 OOM 时：`--module llm --video ...` 单独跑，或先停其它占内存进程。

6. **与主系统对齐**  
   - `max_reply_chars` / `max_generate_tokens` / `ollama_options` / `fallback_phrase` 在 [perception/llm/qwen_multimodal.py](perception/llm/qwen_multimodal.py) 与 `module_debug` 共用同一 [config/config.yaml](config/config.yaml) 段。

7. **融合被资源门控跳过**  
   - 日志 `[Inference] fusion 统计 ok=… skipped=…`：若 `skipped_ratio` 过高，放宽 `resources.memory_threshold` 或降低 `system.vision_sample_interval` 的代价是峰值内存上升，需与 OOM 折中。

8. **全流程 main.py 下 Ollama 报「requires more system memory」**  
   - 已默认 `system.pause_asr_during_llm: true`：唤醒词触发后尽早 `set_llm_busy`，ASR 线程暂停 FunASR 推理，减少与 Ollama 争抢统一内存。  
   - 仍失败时可再降 `models.llm.ollama_options.num_ctx` / `num_predict`，或设置环境变量 `QIMING_VL_MAX_SIDE=256` 压低送 VLM 的分辨率。
