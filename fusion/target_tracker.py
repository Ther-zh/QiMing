import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict

from utils.logger import logger
from utils.config_loader import config_loader

class TargetTracker:
    def __init__(self):
        """
        初始化目标跟踪模块
        """
        self.config = config_loader.get_config()
        self.track_history = defaultdict(list)  # 存储目标的历史轨迹
        self.next_target_id = 1  # 下一个目标ID
    
    def track_targets(self, targets: List[Dict[str, Any]], timestamp: float) -> List[Dict[str, Any]]:
        """
        跟踪目标并计算速度
        
        Args:
            targets: 目标列表
            timestamp: 时间戳
            
        Returns:
            带有ID和速度信息的目标列表
        """
        tracked_targets = []
        
        # 为每个目标分配ID并计算速度
        for target in targets:
            # 查找匹配的历史目标
            matched_id = self._find_matching_target(target)
            
            if matched_id is None:
                # 新目标
                target_id = self.next_target_id
                self.next_target_id += 1
            else:
                # 已有目标
                target_id = matched_id
            
            # 添加ID信息
            target["id"] = target_id
            
            # 记录目标历史
            self.track_history[target_id].append((timestamp, target))
            
            # 限制历史记录长度
            if len(self.track_history[target_id]) > 3:
                self.track_history[target_id] = self.track_history[target_id][-3:]
            
            # 计算速度
            speed = self._calculate_speed(target_id)
            target["speed"] = speed
            
            tracked_targets.append(target)
        
        # 清理过期的目标
        self._clean_expired_targets(timestamp)
        
        return tracked_targets
    
    def _find_matching_target(self, target: Dict[str, Any]) -> Optional[int]:
        """
        查找匹配的历史目标
        
        Args:
            target: 当前目标
            
        Returns:
            匹配的目标ID或None
        """
        best_match_id = None
        best_iou = 0.0
        
        # 获取当前目标的边界框
        current_box = target.get("roi_coords", [])
        if len(current_box) != 4:
            return None
        
        # 遍历历史目标
        for target_id, history in self.track_history.items():
            if not history:
                continue
            
            # 获取最新的历史记录
            latest_record = history[-1]
            latest_target = latest_record[1]
            
            # 计算IOU
            iou = self._calculate_iou(current_box, latest_target.get("roi_coords", []))
            
            if iou > best_iou and iou > 0.3:  # IOU阈值
                best_iou = iou
                best_match_id = target_id
        
        return best_match_id
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        计算两个边界框的IOU
        
        Args:
            box1: 第一个边界框 [x1, y1, x2, y2]
            box2: 第二个边界框 [x1, y1, x2, y2]
            
        Returns:
            IOU值
        """
        if len(box1) != 4 or len(box2) != 4:
            return 0.0
        
        # 计算交集
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x1 >= x2 or y1 >= y2:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # 计算并集
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_speed(self, target_id: int) -> float:
        """
        计算目标的速度
        
        Args:
            target_id: 目标ID
            
        Returns:
            速度（米/秒）
        """
        history = self.track_history.get(target_id, [])
        
        if len(history) < 2:
            return 0.0
        
        try:
            # 计算最近3帧的速度
            speeds = []
            for i in range(1, len(history)):
                t1, target1 = history[i-1]
                t2, target2 = history[i]
                
                # 计算时间差
                time_diff = t2 - t1
                if time_diff <= 0:
                    continue
                
                # 计算距离差
                distance1 = target1.get("distance", 0.0)
                distance2 = target2.get("distance", 0.0)
                distance_diff = abs(distance2 - distance1)
                
                # 计算速度
                speed = distance_diff / time_diff
                speeds.append(speed)
            
            if not speeds:
                return 0.0
            
            # 取平均速度
            return sum(speeds) / len(speeds)
        except Exception as e:
            logger.error(f"计算目标速度时出错: {e}")
            return 0.0
    
    def _clean_expired_targets(self, current_timestamp: float):
        """
        清理过期的目标
        
        Args:
            current_timestamp: 当前时间戳
        """
        expired_ids = []
        
        for target_id, history in self.track_history.items():
            if not history:
                expired_ids.append(target_id)
                continue
            
            # 检查最后一次出现的时间
            latest_timestamp = history[-1][0]
            if current_timestamp - latest_timestamp > 5.0:  # 5秒未出现视为过期
                expired_ids.append(target_id)
        
        # 清理过期目标
        for target_id in expired_ids:
            del self.track_history[target_id]
            logger.debug(f"目标 {target_id} 已过期，已清理")
