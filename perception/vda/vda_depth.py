"""
Video-Depth-Anything 单帧深度估计封装：对单帧做 T=1 前向，避免整段 32 帧 infer 在 Jetson 上 OOM。
权重需自行下载至 ``{model_path}/checkpoints/video_depth_anything_{encoder}.pth``。
"""
import importlib.util
import os
import sys
import types
import gc
from typing import Dict, Any, Optional, Type

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.logger import logger

_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _import_video_depth_anything_class(model_path: str) -> Type[torch.nn.Module]:
    """
    子项目使用顶层包名 ``utils``，与 QiMing 根目录的 ``utils`` 冲突。
    在导入 ``video_depth_anything`` 前后临时挂载子项目的 ``utils.util``，再恢复工程 ``utils``。
    """
    util_path = os.path.join(model_path, "utils", "util.py")
    if not os.path.isfile(util_path):
        raise FileNotFoundError(f"VDA 子项目缺少 {util_path}")

    prev_utils = sys.modules.get("utils")
    prev_utils_util = sys.modules.get("utils.util")

    vda_utils_pkg = types.ModuleType("utils")
    spec = importlib.util.spec_from_file_location("utils.util", util_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {util_path}")
    util_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(util_mod)
    vda_utils_pkg.util = util_mod
    sys.modules["utils"] = vda_utils_pkg
    sys.modules["utils.util"] = util_mod

    try:
        if model_path not in sys.path:
            sys.path.insert(0, model_path)
        from video_depth_anything.video_depth import VideoDepthAnything

        return VideoDepthAnything
    finally:
        if prev_utils is not None:
            sys.modules["utils"] = prev_utils
        else:
            sys.modules.pop("utils", None)
        if prev_utils_util is not None:
            sys.modules["utils.util"] = prev_utils_util
        else:
            sys.modules.pop("utils.util", None)


class VDADepthEstimator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = os.path.abspath(
            config.get("model_path", os.path.join(os.path.dirname(__file__), "Video-Depth-Anything"))
        )
        self.encoder = config.get("encoder", "vits")
        ck = config.get("checkpoint_path") or os.path.join(
            self.model_path, "checkpoints", f"video_depth_anything_{self.encoder}.pth"
        )
        self.checkpoint_path = os.path.abspath(ck)
        self.device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self._input_size = int(config.get("input_size", 518))
        self._fp32 = bool(config.get("fp32", False))
        self._max_input_side = int(config.get("vda_max_input_side", 0) or 0)
        self.model: Optional[torch.nn.Module] = None
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"VDA 模型目录不存在: {self.model_path}")
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                f"未找到深度估计权重: {self.checkpoint_path}\n"
                "请从 Hugging Face 下载对应 encoder 的 .pth 放到 checkpoints/ 下，例如 vits:\n"
                "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth"
            )
        if os.path.getsize(self.checkpoint_path) < 1024:
            raise FileNotFoundError(
                f"深度权重文件无效或为空（<1KB）: {self.checkpoint_path}\n"
                "大文件建议保存到 /home/nvidia/models/root/autodl-tmp/ 后在 config 中设置 checkpoint_path，再执行：\n"
                "  curl -L -o <路径> 'https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth'"
            )
        if self.encoder not in _MODEL_CONFIGS:
            raise ValueError(f"不支持的 VDA encoder: {self.encoder}，可选: {list(_MODEL_CONFIGS)}")

        try:
            VideoDepthAnything = _import_video_depth_anything_class(self.model_path)
        except Exception as e:
            raise ImportError(
                f"无法导入 Video-Depth-Anything（路径 {self.model_path}）。"
                "请确认子模块完整且已安装依赖（如 easydict、einops，见 requirements-jetson.txt）。"
            ) from e

        params = dict(_MODEL_CONFIGS[self.encoder])
        net = VideoDepthAnything(**params)
        state = torch.load(self.checkpoint_path, map_location="cpu")
        net.load_state_dict(state, strict=True)
        net = net.to(self.device).eval()
        self.model = net
        logger.info(
            f"[VDA] Video-Depth-Anything 加载成功 encoder={self.encoder}, device={self.device}, ckpt={self.checkpoint_path}"
        )

    def inference(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("VDA 模型未加载")
        if image is None or image.size == 0:
            raise ValueError("空图像")

        if self._max_input_side > 0:
            h, w = image.shape[:2]
            m = max(h, w)
            if m > self._max_input_side:
                sc = self._max_input_side / m
                image = cv2.resize(
                    image,
                    (int(w * sc), int(h * sc)),
                    interpolation=cv2.INTER_AREA,
                )

        if self.model_path not in sys.path:
            sys.path.insert(0, self.model_path)
        from video_depth_anything.util.compose_shim import Compose
        from video_depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        frame_height, frame_width = rgb.shape[:2]
        input_size = self._input_size
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        transform = Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        tin = torch.from_numpy(transform({"image": rgb / 255.0})["image"]).unsqueeze(0).unsqueeze(0)
        dev = "cuda" if str(self.device).startswith("cuda") else "cpu"
        use_fp32 = self._fp32 or dev == "cpu"
        tin = tin.to(dev)
        with torch.no_grad():
            with torch.autocast(device_type=dev, enabled=(not use_fp32)):
                depth = self.model.forward(tin)
        depth = depth.to(tin.dtype)
        depth = F.interpolate(
            depth.flatten(0, 1).unsqueeze(1),
            size=(frame_height, frame_width),
            mode="bilinear",
            align_corners=True,
        )
        depth_map = depth[0, 0].detach().cpu().numpy().astype(np.float32)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return depth_map

    def release(self) -> None:
        if self.model is not None:
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[VDA] 模型资源已释放")
