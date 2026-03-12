from typing import Dict, Any, List

from utils.data_formatter import DataFormatter
from utils.logger import logger

class MetadataWrapper:
    def __init__(self):
        """
        初始化元数据封装模块
        """
        pass
    
    def wrap_metadata(self, frame: Any, targets: List[Dict[str, Any]], timestamp: float, camera_id: int) -> Dict[str, Any]:
        """
        封装环境感知元数据
        
        Args:
            frame: 原始图像
            targets: 目标列表
            timestamp: 时间戳
            camera_id: 摄像头ID
            
        Returns:
            标准化的元数据字典
        """
        try:
            # 计算场景复杂度评分
            scene_score = self._calculate_scene_score(targets)
            
            # 格式化元数据
            metadata = DataFormatter.format_metadata(
                timestamp=timestamp,
                camera_id=camera_id,
                targets=targets,
                scene_score=scene_score
            )
            
            return metadata
        except Exception as e:
            logger.error(f"封装元数据时出错: {e}")
            # 返回默认元数据
            return DataFormatter.format_metadata(
                timestamp=timestamp,
                camera_id=camera_id,
                targets=[],
                scene_score=0.0
            )
    
    def _calculate_scene_score(self, targets: List[Dict[str, Any]]) -> float:
        """
        计算场景复杂度评分
        
        Args:
            targets: 目标列表
            
        Returns:
            场景复杂度评分
        """
        scene_score = 0.0
        
        # 场景复杂度相关的目标类别
        complexity_categories = {
            "traffic_light": 50,
            "crosswalk": 30,
            "intersection": 40,
            "construction": 35
        }
        
        for target in targets:
            category = target.get("category")
            if category in complexity_categories:
                scene_score += complexity_categories[category]
        
        return scene_score
