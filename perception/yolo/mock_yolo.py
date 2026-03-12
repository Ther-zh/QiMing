import cv2
import numpy as np
from typing import List, Dict, Any

class MockYoloDetector:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Mock YOLO目标检测模型
        
        Args:
            config: 配置字典
        """
        self.config = config
        print("[Mock YOLO] 初始化成功")
    
    def inference(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        模拟目标检测
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            模拟的检测结果列表，每个元素包含category、confidence、roi_coords、direction
        """
        # 模拟检测结果
        height, width = image.shape[:2]
        
        # 生成固定的3-5个模拟目标
        mock_detections = [
            {
                "category": "person",
                "confidence": 0.95,
                "roi_coords": [width * 0.3, height * 0.6, width * 0.4, height * 0.9],
                "direction": "front"
            },
            {
                "category": "car",
                "confidence": 0.90,
                "roi_coords": [width * 0.6, height * 0.5, width * 0.8, height * 0.7],
                "direction": "right"
            },
            {
                "category": "traffic_light",
                "confidence": 0.85,
                "roi_coords": [width * 0.2, height * 0.2, width * 0.25, height * 0.3],
                "direction": "left"
            },
            {
                "category": "obstacle",
                "confidence": 0.80,
                "roi_coords": [width * 0.45, height * 0.7, width * 0.55, height * 0.8],
                "direction": "front"
            }
        ]
        
        return mock_detections
    
    def release(self):
        """
        释放资源
        """
        print("[Mock YOLO] 资源已释放")
