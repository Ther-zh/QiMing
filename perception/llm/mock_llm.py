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
        # 模拟生成的回复
        mock_responses = [
            "前方道路安全，可以正常通行。",
            "请注意，前方有行人，建议减速慢行。",
            "前方路口为红灯，请等待。",
            "左侧有障碍物，请注意避让。",
            "前方道路畅通，可以正常前进。"
        ]
        
        import random
        return random.choice(mock_responses)
    
    def release(self):
        """
        释放资源
        """
        print("[Mock LLM] 资源已释放")
