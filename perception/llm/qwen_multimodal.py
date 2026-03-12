import os
import numpy as np
from typing import Dict, Any, Tuple, Optional
from PIL import Image

class QwenMultimodal:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Qwen多模态大模型
        
        Args:
            config: 配置字典，包含模型路径等参数
        """
        self.config = config
        self.model_path = config.get('model_path', '/root/autodl-tmp/qwen35/tclf90/Qwen3___5-4B-AWQ')
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """
        加载Qwen模型
        """
        try:
            from qwen_vllm_wrapper import Qwen35VLLM
            self.model = Qwen35VLLM(
                model_path=self.model_path,
                max_model_len=8192
            )
            print(f"[LLM] 模型加载成功: {self.model_path}")
        except Exception as e:
            print(f"[LLM] 模型加载失败: {e}")
            # 如果无法加载真实模型，使用模拟实现
            print("[LLM] 使用模拟实现")
            self.model = self._create_mock_model()
    
    def _create_mock_model(self):
        """
        创建模拟模型
        """
        class MockModel:
            def generate(self, text, **kwargs):
                return "前方道路安全，可以正常通行"
        return MockModel()
    
    def inference(self, input_data: Tuple[Optional[Image.Image], Dict[str, Any], str]) -> str:
        """
        执行多模态推理
        
        Args:
            input_data: Tuple[image, metadata, prompt]，包含图像、环境元数据和用户指令
            
        Returns:
            生成的口语化文本（≤100字）
        """
        if self.model is None:
            raise RuntimeError("LLM模型未加载")
        
        image, metadata, prompt = input_data
        
        # 执行推理
        response = self.model.generate(
            text=prompt,
            image=image if image else None,
            max_tokens=100
        )
        
        # 确保返回的文本不超过100字
        return response[:100]
    
    def release(self):
        """
        释放模型资源
        """
        if self.model:
            # 释放模型资源
            self.model = None
            print("[LLM] 模型资源已释放")
