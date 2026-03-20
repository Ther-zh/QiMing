import os
import re
import numpy as np
from typing import Optional, Union, List
from PIL import Image
import ollama

class Qwen35Ollama:
    def __init__(
        self,
        model_name: str = "qwen3.5-4b",
        **kwargs
    ):
        """
        初始化 Qwen3.5 Ollama 推理引擎。
        
        Args:
            model_name: Ollama模型名称
            **kwargs: 其他参数（兼容旧接口）
        """
        self.model_name = model_name
        self.default_params = {
            "temperature": 0.6,
            "top_p": 0.8,
            "top_k": 20,
        }
        print(f"[System] Ollama模型已就绪: {model_name}")

    def _clean_response(self, text: str) -> str:
        """
        清理响应文本，移除思考标签等内容
        """
        if not text:
            return text
        
        text = re.sub(r'<think[^>]*>.*?</think\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<think[^>]*>.*', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Thinking Process:.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\*\*分析.*?\*\*.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = text.strip()
        
        return text

    def generate(
        self,
        text: str,
        image: Optional[Union[str, Image.Image]] = None,
        sampling_params=None,
        **kwargs
    ) -> str:
        """
        生成回复。
        
        Args:
            text: 输入的文本提示词
            image: 输入的图片 (暂不支持，保留接口兼容)
            sampling_params: 采样参数 (兼容旧接口，暂不使用)
            **kwargs: 临时覆盖参数 (如 temperature, max_tokens 等)
        
        Returns:
            生成的文本字符串
        """
        params = self.default_params.copy()
        if kwargs:
            if "temperature" in kwargs:
                params["temperature"] = kwargs["temperature"]
            if "max_tokens" in kwargs:
                params["num_predict"] = kwargs["max_tokens"]
        
        try:
            import sys
            print(f"[LLM] 执行生成...")
            sys.stdout.flush()
            
            response = ollama.generate(
                model=self.model_name,
                prompt=text,
                options=params
            )
            
            print("[LLM] 生成完成")
            sys.stdout.flush()
            
            result = response.get("response", "").strip()
            result = self._clean_response(result)
            
            if result:
                print(f"[LLM] 生成结果: {result[:100]}...")
                sys.stdout.flush()
                return result
            else:
                print("[LLM] 生成结果为空")
                sys.stdout.flush()
                return "前方道路安全，可以正常通行"
        except Exception as e:
            print(f"[LLM] 生成失败: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            return "前方道路安全，可以正常通行"

    def chat(
        self,
        messages: List[dict],
        **kwargs
    ) -> str:
        """
        对话模式生成。
        
        Args:
            messages: 消息列表
            **kwargs: 额外参数
        
        Returns:
            生成的文本字符串
        """
        params = self.default_params.copy()
        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            params["num_predict"] = kwargs["max_tokens"]
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options=params
            )
            return response.get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"[LLM] Chat失败: {e}")
            return "前方道路安全，可以正常通行"

    def batch_generate(
        self,
        inputs: List[dict],
        sampling_params=None,
    ) -> List[str]:
        """
        批量生成。
        
        Args:
            inputs: 字典列表，每个字典包含 "text"
            sampling_params: 采样参数
        
        Returns:
            生成结果列表
        """
        results = []
        for item in inputs:
            text = item.get("text", "")
            result = self.generate(text)
            results.append(result)
        return results


class Qwen35VLLM(Qwen35Ollama):
    """
    兼容旧接口的别名类
    """
    pass


if __name__ == "__main__":
    model = Qwen35Ollama(model_name="qwen3.5-4b")
    
    print("\n--- 测试纯文本生成 ---")
    response = model.generate("你好，请用一句话介绍你自己。")
    print("回复:", response)
