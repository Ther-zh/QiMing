import numpy as np
from typing import Dict, Any, List, Optional
from utils.logger import logger
from utils.config_loader import config_loader

class RiskEvaluator:
    def __init__(self):
        """
        初始化危险评价器
        """
        self.config = config_loader.get_config()
        self.risk_rules = config_loader.get_risk_rules()
        
        # AHP权重矩阵
        self.ahp_weights = self._calculate_ahp_weights()
        
        # 模糊评价参数
        self.fuzzy_params = {
            'distance': {
                'very_close': (0, 0, 1),
                'close': (0.5, 1.5, 2.5),
                'medium': (2, 3.5, 5),
                'far': (4, 7, 10),
                'very_far': (8, 15, 20)
            },
            'speed': {
                'very_slow': (0, 0, 3),
                'slow': (2, 4, 6),
                'medium': (5, 8, 12),
                'fast': (10, 15, 20),
                'very_fast': (18, 25, 30)
            },
            'category': {
                'low_risk': (0, 0, 5),
                'medium_risk': (4, 7, 10),
                'high_risk': (8, 12, 15),
                'very_high_risk': (14, 18, 20)
            }
        }
        
        # 特殊场景权重调整
        self.scene_adjustments = {
            'crosswalk': 1.5,  # 人行横道风险提升
            'intersection': 1.8,  # 交叉路口风险提升
            'construction': 1.3,  # 施工区域风险提升
            'crowded_area': 1.4,  # 人群密集区域风险提升
            'dark_area': 1.2  # 光线较暗区域风险提升
        }
    
    def _calculate_ahp_weights(self) -> Dict[str, float]:
        """
        使用AHP计算各因素权重
        
        Returns:
            各因素权重字典
        """
        # 构造判断矩阵（1-9标度法）
        # 因素：距离、速度、类别、场景复杂度
        judgment_matrix = np.array([
            [1, 3, 2, 4],  # 距离
            [1/3, 1, 1/2, 2],  # 速度
            [1/2, 2, 1, 3],  # 类别
            [1/4, 1/2, 1/3, 1]  # 场景复杂度
        ])
        
        # 计算权重
        eigenvalues, eigenvectors = np.linalg.eig(judgment_matrix)
        max_eigenvalue = np.max(eigenvalues)
        max_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
        
        # 归一化权重
        weights = max_eigenvector.real / np.sum(max_eigenvector.real)
        
        # 一致性检验
        n = judgment_matrix.shape[0]
        ci = (max_eigenvalue - n) / (n - 1)
        ri = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
        cr = ci / ri[n-1] if n > 2 else 0
        
        if cr < 0.1:
            logger.info(f"AHP一致性检验通过，CR={cr:.3f}")
        else:
            logger.warning(f"AHP一致性检验未通过，CR={cr:.3f}")
        
        return {
            'distance': weights[0],
            'speed': weights[1],
            'category': weights[2],
            'scene_complexity': weights[3]
        }
    
    def _fuzzy_evaluation(self, value: float, param_name: str) -> Dict[str, float]:
        """
        模糊评价函数
        
        Args:
            value: 评价值
            param_name: 参数名称
            
        Returns:
            模糊评价结果
        """
        params = self.fuzzy_params.get(param_name, {})
        memberships = {}
        
        for level, (a, m, b) in params.items():
            if value <= a:
                memberships[level] = 0
            elif a < value <= m:
                memberships[level] = (value - a) / (m - a)
            elif m < value <= b:
                memberships[level] = (b - value) / (b - m)
            else:
                memberships[level] = 0
        
        return memberships
    
    def _calculate_scene_complexity(self, targets: List[Dict[str, Any]]) -> float:
        """
        计算场景复杂度
        
        Args:
            targets: 目标列表
            
        Returns:
            场景复杂度得分
        """
        complexity_items = self.risk_rules.get("complexity_items", {})
        scene_score = 0.0
        scene_types = set()
        
        for target in targets:
            category = target.get("category")
            if category in complexity_items:
                scene_score += complexity_items[category]
                scene_types.add(category)
        
        # 考虑目标数量
        target_count = len(targets)
        if target_count > 5:
            scene_score += (target_count - 5) * 5
        
        return scene_score
    
    def _get_category_risk_score(self, category: str) -> float:
        """
        获取类别风险分数
        
        Args:
            category: 目标类别
            
        Returns:
            风险分数
        """
        category_weights = self.risk_rules.get("category_weights", {})
        # 从配置文件获取类别权重，如果不存在则返回默认值
        if category in category_weights:
            return category_weights[category]
        # 对于未明确配置的类别，根据其性质设置默认值
        elif category in ['car', 'bus', 'truck']:
            return 15
        elif category in ['person', 'bicycle', 'wheelchair']:
            return 10
        elif category in ['construction_zone', 'obstacle', 'pothole']:
            return 12
        elif category in ['crosswalk', 'traffic_light', 'bus_stop']:
            return 5
        else:
            return 5
    
    def evaluate_risk(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        综合评价危险等级
        
        Args:
            metadata: 环境感知元数据
            
        Returns:
            危险评价结果
        """
        targets = metadata.get("targets", [])
        if not targets:
            return {
                'risk_level': 'level4',
                'risk_score': 0,
                'message': '道路安全',
                'details': '无目标',
                'skip_alert': True  # 标记为跳过告警
            }
        
        # 计算每个目标的风险
        target_risks = []
        for target in targets:
            category = target.get("category", "unknown")
            distance = target.get("distance", 10.0)
            speed = target.get("speed", 0.0)
            
            # 模糊评价
            distance_membership = self._fuzzy_evaluation(distance, 'distance')
            speed_membership = self._fuzzy_evaluation(speed, 'speed')
            category_score = self._get_category_risk_score(category)
            category_membership = self._fuzzy_evaluation(category_score, 'category')
            
            # 计算各因素得分
            # 1. 距离因素：距离越近，风险越高
            # 距离得分 = 10 - 距离（最多10分），最小0分
            distance_score = max(0, 10 - min(distance, 10))
            
            # 2. 速度因素：
            # 对于车辆：速度越快，风险越高；速度为0（静止）时，风险降低
            # 对于行人：适中速度风险较低，过快或过慢风险较高
            if category in ['car', 'bus', 'truck', 'bicycle', 'motorcycle']:
                if speed < 1.0:  # 静止或几乎静止
                    speed_score = 2  # 基础风险
                else:
                    speed_score = min(10, speed * 0.3)  # 速度越快，风险越高
            else:  # 行人等其他目标
                if speed < 0.5 or speed > 5.0:
                    speed_score = 6  # 过慢或过快
                else:
                    speed_score = 3  # 正常速度
            
            # 3. 类别因素
            category_score = self._get_category_risk_score(category)
            
            # 4. 场景复杂度
            scene_complexity = self._calculate_scene_complexity(targets)
            scene_score = min(scene_complexity / 100, 1.0) * 10
            
            # 5. 静止车辆特殊处理：如果车辆静止但距离很近，视为障碍物
            if category in ['car', 'bus', 'truck'] and speed < 1.0 and distance < 3.0:
                # 静止车辆离得近，风险提高
                distance_score *= 1.5
                category_score *= 1.2
            
            # 使用AHP权重计算综合得分
            weights = self.ahp_weights
            comprehensive_score = (
                distance_score * weights['distance'] +
                speed_score * weights['speed'] +
                category_score * weights['category'] +
                scene_score * weights['scene_complexity']
            )
            
            # 特殊场景调整
            for scene_type, adjustment in self.scene_adjustments.items():
                if scene_type in [t.get('category') for t in targets]:
                    comprehensive_score *= adjustment
                    break
            
            target_risks.append({
                'target': target,
                'risk_score': comprehensive_score
            })
        
        # 取风险最高的目标
        highest_risk = max(target_risks, key=lambda x: x['risk_score'])
        risk_score = highest_risk['risk_score']
        high_risk_target = highest_risk['target']
        
        # 确定危险等级
        skip_alert = False
        if risk_score >= 80:
            level = "level1"
            message = "危险！请立即避让！"
        elif risk_score >= 50:
            level = "level2"
            message = "注意！前方有危险！"
        elif risk_score >= 30:
            level = "level3"
            message = "请注意前方情况"
        else:
            level = "level4"
            message = "道路安全"
            skip_alert = True
        
        result = {
            'risk_level': level,
            'risk_score': risk_score,
            'message': message,
            'target_info': high_risk_target,
            'details': {
                'target_count': len(targets),
                'highest_risk_target': high_risk_target.get('category', 'unknown'),
                'distance': high_risk_target.get('distance', 0),
                'speed': high_risk_target.get('speed', 0)
            }
        }
        
        if skip_alert:
            result['skip_alert'] = True
        
        return result
    
    def evaluate_special_scene(self, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        特殊场景处理
        
        Args:
            metadata: 环境感知元数据
            
        Returns:
            特殊场景评价结果
        """
        targets = metadata.get("targets", [])
        scene_types = [t.get('category') for t in targets]
        
        # 人行横道场景
        if 'crosswalk' in scene_types:
            # 检查是否有行人
            has_person = any(t.get('category') == 'person' for t in targets)
            if has_person:
                return {
                    'risk_level': 'level2',
                    'risk_score': 60,
                    'message': '注意！前方人行横道有行人！',
                    'special_scene': 'crosswalk'
                }
        
        # 交叉路口场景
        if 'traffic_light' in scene_types:
            # 检查是否有车辆
            has_vehicle = any(t.get('category') in ['car', 'truck', 'motorcycle'] for t in targets)
            if has_vehicle:
                return {
                    'risk_level': 'level2',
                    'risk_score': 65,
                    'message': '注意！前方交叉路口有车辆！',
                    'special_scene': 'intersection'
                }
        
        # 施工区域场景
        if 'construction' in scene_types:
            return {
                'risk_level': 'level3',
                'risk_score': 40,
                'message': '请注意！前方施工区域！',
                'special_scene': 'construction'
            }
        
        return None