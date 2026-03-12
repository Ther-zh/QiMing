from typing import Dict, Any, List
from datetime import datetime

class DataFormatter:
    @staticmethod
    def format_metadata(
        timestamp: float,
        camera_id: int,
        targets: List[Dict[str, Any]],
        scene_score: float = 0.0
    ) -> Dict[str, Any]:
        """
        格式化环境感知元数据
        
        Args:
            timestamp: 时间戳
            camera_id: 摄像头ID
            targets: 目标列表
            scene_score: 场景复杂度评分
            
        Returns:
            格式化后的元数据字典
        """
        return {
            "timestamp": timestamp,
            "camera_id": camera_id,
            "targets": targets,
            "scene_score": scene_score,
            "processed_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def format_target(
        category: str,
        confidence: float,
        roi_coords: List[float],
        direction: str,
        distance: float = 0.0,
        speed: float = 0.0,
        target_id: int = -1
    ) -> Dict[str, Any]:
        """
        格式化目标信息
        
        Args:
            category: 目标类别
            confidence: 置信度
            roi_coords: 边界框坐标
            direction: 方向
            distance: 距离
            speed: 速度
            target_id: 目标ID
            
        Returns:
            格式化后的目标字典
        """
        return {
            "id": target_id,
            "category": category,
            "confidence": confidence,
            "roi_coords": roi_coords,
            "direction": direction,
            "distance": distance,
            "speed": speed
        }
    
    @staticmethod
    def format_alert(
        level: str,
        message: str,
        timestamp: float,
        target_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        格式化告警信息
        
        Args:
            level: 告警级别
            message: 告警消息
            timestamp: 时间戳
            target_info: 目标信息
            
        Returns:
            格式化后的告警字典
        """
        return {
            "level": level,
            "message": message,
            "timestamp": timestamp,
            "target_info": target_info,
            "created_at": datetime.now().isoformat()
        }
