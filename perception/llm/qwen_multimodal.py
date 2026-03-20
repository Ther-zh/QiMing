import os
import numpy as np
from typing import Dict, Any, Tuple, Optional
from PIL import Image

class QwenMultimodal:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Qwen多模态大模型 (Ollama版本)
        
        Args:
            config: 配置字典，包含模型名称等参数
        """
        self.config = config
        self.model_name = config.get('model_name', 'qwen3.5-4b')
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """
        加载Qwen模型 (使用Ollama)
        """
        try:
            import sys
            sys.path.append('/root/MHSEE')
            from LLM.qwen35 import Qwen35Ollama
            self.model = Qwen35Ollama(model_name=self.model_name)
            print(f"[LLM] Ollama模型加载成功: {self.model_name}")
        except Exception as e:
            print(f"[LLM] 模型加载失败: {e}")
            raise
    
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
        
        response = self.model.generate(
            text=prompt,
            image=image if image is not None else None,
            max_tokens=100
        )
        
        return response[:100]
    
    def release(self):
        """
        释放模型资源
        """
        if self.model:
            self.model = None
            print("[LLM] 模型资源已释放")
