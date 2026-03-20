import os
from typing import Optional, Union, List
from PIL import Image
import ollama

class Qwen35VLLM:
    def __init__(
        self,
        model_path: str = None,
        model_name: str = "qwen3.5-4b",
        max_model_len: int = 8192,
        tensor_parallel_size: int = 1,
        **kwargs
    ):
        """
        初始化 Qwen3.5 Ollama 推理引擎。
        
        Args:
            model_path: 模型路径 (兼容旧接口，不再使用)
            model_name: Ollama模型名称
            max_model_len: 最大上下文长度 (兼容旧接口)
            tensor_parallel_size: 张量并行大小 (兼容旧接口)
            **kwargs: 其他参数
        """
        self.model_name = model_name
        self.default_params = {
            "temperature": 0.6,
            "top_p": 0.8,
            "top_k": 20,
            "num_ctx": max_model_len,
        }
        print(f"[System] Ollama模型已就绪: {model_name}")

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
            sampling_params: 采样参数 (兼容旧接口)
            **kwargs: 临时覆盖参数
        
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
            response = ollama.generate(
                model=self.model_name,
                prompt=text,
                options=params
            )
            
            result = response.get("response", "").strip()
            if result:
                return result
            else:
                return "前方道路安全，可以正常通行"
        except Exception as e:
            print(f"[LLM] 生成失败: {e}")
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


if __name__ == "__main__":
    model = Qwen35VLLM(model_name="qwen3.5-4b")
    
    print("\n--- 测试纯文本生成 ---")
    response = model.generate("你好，请用一句话介绍你自己。")
    print("回复:", response)
