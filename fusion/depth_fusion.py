import numpy as np
from typing import Dict, Any, List

from utils.logger import logger

class DepthFusion:
    def __init__(self):
        """
        初始化深度融合模块
        """
        pass
    
    def calculate_target_distances(self, yolo_results: List[Dict[str, Any]], depth_map: np.ndarray) -> List[Dict[str, Any]]:
        """
        计算每个目标的距离
        
        Args:
            yolo_results: YOLO检测结果列表
            depth_map: 深度图
            
        Returns:
            带有距离信息的目标列表
        """
        enhanced_results = []
        
        for target in yolo_results:
            # 计算目标距离
            distance = self._calculate_distance(target, depth_map)
            
            # 添加距离信息
            enhanced_target = target.copy()
            enhanced_target["distance"] = distance
            enhanced_results.append(enhanced_target)
        
        return enhanced_results
    
    def _calculate_distance(self, target: Dict[str, Any], depth_map: np.ndarray) -> float:
        """
        计算单个目标的距离
        
        Args:
            target: 目标信息
            depth_map: 深度图
            
        Returns:
            目标距离（米）
        """
        try:
            # 获取目标的ROI坐标
            roi_coords = target.get("roi_coords", [])
            if len(roi_coords) != 4:
                return 0.0
            
            x1, y1, x2, y2 = roi_coords
            
            # 转换为整数坐标
            height, width = depth_map.shape[:2]
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(width - 1, int(x2))
            y2 = min(height - 1, int(y2))
            
            # 确保ROI有效
            if x1 >= x2 or y1 >= y2:
                return 0.0
            
            # 提取ROI区域的深度值
            roi_depth = depth_map[y1:y2, x1:x2]
            
            # 剔除边缘10%像素
            edge_percent = 0.1
            h, w = roi_depth.shape
            h_start = int(h * edge_percent)
            h_end = int(h * (1 - edge_percent))
            w_start = int(w * edge_percent)
            w_end = int(w * (1 - edge_percent))
            
            if h_start >= h_end or w_start >= w_end:
                return 0.0
            
            center_roi = roi_depth[h_start:h_end, w_start:w_end]
            
            # 过滤异常值
            filtered_depth = center_roi[(center_roi >= 0.1) & (center_roi <= 20.0)]
            
            if len(filtered_depth) == 0:
                return 0.0
            
            # 取中值作为目标距离
            distance = np.median(filtered_depth)
            
            return float(distance)
        except Exception as e:
            logger.error(f"计算目标距离时出错: {e}")
            return 0.0
