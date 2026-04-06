import numpy as np
from typing import Dict, Any, Tuple, Optional
from PIL import Image

class MockQwenMultimodal:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Mock Qwen多模态大模型
        
        Args:
            config: 配置字典
        """
        self.config = config
        print("[Mock LLM] 初始化成功")
    
    def inference(self, input_data: Tuple[Optional[Image.Image], Dict[str, Any], str]) -> str:
        """
        模拟多模态推理
        
        Args:
            input_data: Tuple[image, metadata, prompt]，包含图像、环境元数据和用户指令
            
        Returns:
            生成的口语化文本（≤100字）
        """
        image, metadata, prompt = input_data
        metadata = metadata or {}

        print(f"[Mock LLM] 接收到的prompt: {prompt}")

        targets = metadata.get("targets", [])
        target_info = ""
        if targets:
            target_info = "当前环境中的目标有："
            for target in targets:
                category = target.get("category")
                distance = target.get("distance", 0)
                direction = target.get("direction")
                target_info += f"{direction}方向{distance:.1f}米处的{category}，"
            target_info = target_info.rstrip("，") + "。"
            print(f"[Mock LLM] 分析到的目标信息: {target_info}")
        
        # 根据prompt和目标信息生成回复
        if "路况" in prompt:
            if any(target.get("category") == "car" for target in targets):
                return "前方有车辆，请小心通行。"
            elif any(target.get("category") == "person" for target in targets):
                return "前方有行人，建议减速慢行。"
            else:
                return "前方道路畅通，可以正常前进。"
        elif "过马路" in prompt:
            if any(target.get("category") == "traffic_light" for target in targets):
                return "前方有红绿灯，请观察信号灯后再过马路。"
            else:
                return "当前路口没有红绿灯，请左右观察确保安全后再过马路。"
        elif "导航" in prompt:
            return "请沿当前道路直行，前方100米左转。"
        elif "障碍物" in prompt:
            if any(target.get("category") == "obstacle" for target in targets):
                return "前方有障碍物，请小心避让。"
            else:
                return "前方没有障碍物，可以正常通行。"
        else:
            return "当前道路安全，可以正常通行。"
    
    def release(self):
        """
        释放资源
        """
        print("[Mock LLM] 资源已释放")
