import cv2
import numpy as np
import threading
from typing import Dict, Any, List

from utils.logger import logger
from utils.config_loader import config_loader

class DebugViewer:
    def __init__(self):
        """
        初始化调试查看器
        """
        self.config = config_loader.get_config()
        self.running = False
        self.debug_frame = None
        self.lock = threading.Lock()
    
    def start(self):
        """
        启动调试查看器
        """
        self.running = True
        logger.info("调试查看器已启动")
    
    def stop(self):
        """
        停止调试查看器
        """
        self.running = False
        cv2.destroyAllWindows()
        logger.info("调试查看器已停止")
    
    def update_frame(self, frame: np.ndarray, targets: List[Dict[str, Any]], depth_map: np.ndarray = None, risk_level: str = "level4"):
        """
        更新调试画面
        
        Args:
            frame: 原始图像
            targets: 目标列表
            depth_map: 深度图
            risk_level: 危险等级
        """
        with self.lock:
            # 复制帧以避免修改原始数据
            debug_frame = frame.copy()
            
            # 绘制目标框
            for target in targets:
                self._draw_target(debug_frame, target)
            
            # 绘制深度图（如果提供）
            if depth_map is not None:
                debug_frame = self._draw_depth_map(debug_frame, depth_map)
            
            # 绘制危险等级
            self._draw_risk_level(debug_frame, risk_level)
            
            self.debug_frame = debug_frame
    
    def show(self):
        """
        显示调试画面
        """
        while self.running:
            with self.lock:
                if self.debug_frame is not None:
                    cv2.imshow("Debug Viewer", self.debug_frame)
            
            # 按ESC键退出
            if cv2.waitKey(1) == 27:
                break
    
    def _draw_target(self, frame: np.ndarray, target: Dict[str, Any]):
        """
        绘制目标框
        
        Args:
            frame: 图像
            target: 目标信息
        """
        try:
            # 获取边界框坐标
            roi_coords = target.get("roi_coords", [])
            if len(roi_coords) != 4:
                return
            
            x1, y1, x2, y2 = map(int, roi_coords)
            
            # 确定颜色
            category = target.get("category", "unknown")
            color = self._get_category_color(category)
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{category}: {target.get('confidence', 0):.2f}"
            if "distance" in target:
                label += f" {target['distance']:.1f}m"
            if "speed" in target:
                label += f" {target['speed']:.1f}m/s"
            
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except Exception as e:
            logger.error(f"绘制目标框时出错: {e}")
    
    def _draw_depth_map(self, frame: np.ndarray, depth_map: np.ndarray):
        """
        绘制深度图
        
        Args:
            frame: 原始图像
            depth_map: 深度图
            
        Returns:
            带有深度图的图像
        """
        try:
            # 调整深度图大小以匹配原始图像
            depth_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
            
            # 归一化深度图
            depth_normalized = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX)
            depth_normalized = depth_normalized.astype(np.uint8)
            
            # 应用伪彩色
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # 将深度图叠加到原始图像上
            overlay = cv2.addWeighted(frame, 0.7, depth_colored, 0.3, 0)
            
            return overlay
        except Exception as e:
            logger.error(f"绘制深度图时出错: {e}")
            return frame
    
    def _draw_risk_level(self, frame: np.ndarray, risk_level: str):
        """
        绘制危险等级
        
        Args:
            frame: 图像
            risk_level: 危险等级
        """
        # 危险等级配置
        risk_config = {
            "level1": {"text": "高危", "color": (0, 0, 255)},
            "level2": {"text": "中危", "color": (0, 165, 255)},
            "level3": {"text": "低危", "color": (0, 255, 255)},
            "level4": {"text": "安全", "color": (0, 255, 0)}
        }
        
        config = risk_config.get(risk_level, risk_config["level4"])
        
        # 绘制危险等级
        cv2.rectangle(frame, (10, 10), (100, 50), config["color"], -1)
        cv2.putText(frame, config["text"], (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    def _get_category_color(self, category: str) -> tuple:
        """
        获取类别对应的颜色
        
        Args:
            category: 目标类别
            
        Returns:
            BGR颜色元组
        """
        color_map = {
            "person": (0, 255, 0),
            "car": (255, 0, 0),
            "truck": (255, 165, 0),
            "bicycle": (0, 255, 255),
            "motorcycle": (255, 0, 255),
            "traffic_light": (255, 255, 0),
            "crosswalk": (0, 191, 255),
            "obstacle": (128, 128, 128),
            "construction": (165, 42, 42),
            "unknown": (128, 128, 128)
        }
        
        return color_map.get(category, color_map["unknown"])
