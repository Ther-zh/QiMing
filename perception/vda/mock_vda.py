import cv2
import numpy as np
from typing import Dict, Any

class MockVDADepthEstimator:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Mock VDA深度估计模型
        
        Args:
            config: 配置字典
        """
        self.config = config
        print("[Mock VDA] 初始化成功")
    
    def inference(self, image: np.ndarray) -> np.ndarray:
        """
        模拟深度估计
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            模拟的深度图（numpy数组）
        """
        height, width = image.shape[:2]
        
        # 生成模拟深度图
        # 中心区域深度较近，边缘区域深度较远
        depth_map = np.zeros((height, width), dtype=np.float32)
        
        # 计算距离中心的距离
        center_x, center_y = width // 2, height // 2
        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                # 距离中心越远，深度越大
                depth = min(20.0, distance / max(width, height) * 40.0)
                depth_map[y, x] = depth
        
        return depth_map
    
    def release(self):
        """
        释放资源
        """
        print("[Mock VDA] 资源已释放")
