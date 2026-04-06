"""视频录制辅助：深度伪彩色与目标框绘制。"""
import cv2
import numpy as np
from typing import Any, Dict, List


def depth_to_colormap_bgr(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    dmin, dmax = float(d.min()), float(d.max())
    if dmax - dmin < 1e-8:
        norm = np.zeros_like(d, dtype=np.uint8)
    else:
        norm = ((d - dmin) / (dmax - dmin + 1e-8) * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def overlay_depth_on_frame(frame_bgr: np.ndarray, depth_map: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    depth_resized = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
    colored = depth_to_colormap_bgr(depth_resized)
    out = frame_bgr.copy()
    cv2.addWeighted(colored, float(alpha), out, 1.0 - float(alpha), 0, dst=out)
    return out


def draw_tracked_targets(frame: np.ndarray, tracked_targets: List[Dict[str, Any]]) -> np.ndarray:
    for target in tracked_targets:
        x1, y1, x2, y2 = target.get("bbox", [0, 0, 0, 0])
        class_name = target.get("class_name", "unknown")
        distance = target.get("distance", 0)
        speed = target.get("speed", 0)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        info_text = f"{class_name} - 距离: {distance:.1f}m - 速度: {speed:.1f}m/s"
        cv2.putText(frame, info_text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame
