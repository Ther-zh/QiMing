# VDA（深度估计）模块

- **封装代码**：本目录下的 `vda_depth.py`、`mock_vda.py` 为导盲工程封装；`models.vda.model_path` 应指向 **`Video-Depth-Anything` 源码根目录**。
- **唯一维护副本**：请使用 **`perception/vda/Video-Depth-Anything`** 作为子项目与依赖的安装、修改位置。
- **根目录 `vda/Video-Depth-Anything`**：为指向本目录 `Video-Depth-Anything` 的**符号链接**，仅供历史路径兼容；修改代码与权重请只动 **`perception/vda/Video-Depth-Anything`**。

`VDADepthEstimator`（`vda_depth.py`）在 `type: real` 时加载上述子项目权重并推理。

**依赖**：`easydict`、`einops`（已写入根目录 `requirements-jetson.txt`）。

**与工程 `utils` 包名冲突**：子项目使用顶层 `utils.util`；本仓库根目录也有 `utils/`。`vda_depth.py` 在导入子项目时会临时挂载子项目的 `utils.util`，导入结束后恢复，无需改子项目目录名。

**权重**：若 `checkpoints/video_depth_anything_vits.pth` 为空，请运行 `bash sing_test/download_vda_checkpoint.sh`（默认下载到 `models/root/autodl-tmp/`），并在 `config.yaml` 中设置 `models.vda.checkpoint_path`。
