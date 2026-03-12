import cv2
import numpy as np
from typing import List, Dict, Any

class YoloDetector:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化YOLO目标检测模型
        
        Args:
            config: 配置字典，包含模型路径、置信度阈值等参数
        """
        self.config = config
        self.model_path = config.get('model_path', '/root/MHSEE/MHSEE/model/yolov8l-world.pt')
        self.conf_threshold = config.get('conf_threshold', 0.15)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """
        加载YOLO模型
        """
        try:
            # 这里使用YOLOv8的官方库
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"[YOLO] 模型加载成功: {self.model_path}")
        except Exception as e:
            print(f"[YOLO] 模型加载失败: {e}")
            raise
    
    def inference(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        执行目标检测
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            检测结果列表，每个元素包含category、confidence、roi_coords、direction
        """
        if self.model is None:
            raise RuntimeError("YOLO模型未加载")
        
        # 执行检测
        results = self.model(image, conf=self.conf_threshold)
        
        # 处理检测结果
        detections = []
        for result in results:
            for box in result.boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # 获取类别和置信度
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # 获取类别名称
                category = result.names.get(class_id, "unknown")
                
                # 计算中心点
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 简单的方向判断
                height, width = image.shape[:2]
                if center_x < width * 0.3:
                    direction = "left"
                elif center_x > width * 0.7:
                    direction = "right"
                else:
                    direction = "front"
                
                detections.append({
                    "category": category,
                    "confidence": confidence,
                    "roi_coords": [float(x1), float(y1), float(x2), float(y2)],
                    "direction": direction
                })
        
        return detections
    
    def release(self):
        """
        释放模型资源
        """
        if self.model:
            # YOLOv8模型没有显式的释放方法，这里可以做一些清理工作
            self.model = None
            print("[YOLO] 模型资源已释放")
