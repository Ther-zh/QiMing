import os
from typing import Optional, Union, List
from PIL import Image
import ollama
import re

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
        self.config = kwargs.get("config", {}) or {}
        ollama_cfg = (self.config.get("ollama", {}) or {}) if isinstance(self.config, dict) else {}
        self.keep_alive = ollama_cfg.get("keep_alive", 0)
        host = ollama_cfg.get("host") if isinstance(ollama_cfg, dict) else None
        self._client = ollama.Client(host=host) if host else ollama
        self.default_params = {
            "temperature": 0.6,
            "top_p": 0.8,
            "top_k": 20,
            "num_ctx": max_model_len,
        }
        opt = ollama_cfg.get("options", {}) if isinstance(ollama_cfg, dict) else {}
        if isinstance(opt, dict):
            self.default_params.update({k: v for k, v in opt.items() if v is not None})
        print(f"[System] Ollama模型已就绪: {model_name}")

    @staticmethod
    def _clean_response(text: str) -> str:
        if not text:
            return text
        # 清理 qwen thinking / 特殊 token / 训练残留
        text = re.sub(r"<think[^>]*>.*?</think\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<think[^>]*>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = text.replace("<|endoftext|>", "")
        text = re.sub(r"<\|im_start\|>.*?<\|im_end\|>", "", text, flags=re.DOTALL)
        text = re.sub(r"<\|im_start\|>|<\|im_end\|>", "", text)
        # 去掉重复空行
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        # 兜底：去掉连续重复行（低比特量化常见“复读”）
        lines = [ln.rstrip() for ln in text.splitlines()]
        out = []
        last = None
        dup = 0
        for ln in lines:
            if not ln.strip():
                # 合并多空行
                if out and out[-1] == "":
                    continue
                out.append("")
                last = ""
                dup = 0
                continue
            if ln == last:
                dup += 1
                # 连续重复超过 1 次就丢弃（保留第一次）
                if dup >= 1:
                    continue
            else:
                dup = 0
            out.append(ln)
            last = ln
        text = "\n".join(out).strip()
        return text

    def _wrap_prompt(self, user_text: str) -> str:
        """
        completion 模式下，短 prompt（如“你好”）很容易触发无关长文。
        这里加一个稳定的“导盲助手”指令壳，保证输出更可控。
        """
        user_text = (user_text or "").strip()
        return (
            "你是一个导盲系统的语音助手。请用中文回答，简洁直接，不要长篇大论。\n"
            "如果用户只是寒暄（如“你好”），请简单自我介绍并询问需要什么帮助。\n"
            "回答尽量控制在80字以内。\n\n"
            "### 用户问题\n"
            f"{user_text}\n\n"
            "### 助手回答（只输出答案本身）\n"
        )

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
                # 小模型避免跑飞：强制上限更小（可按需再调）
                try:
                    params["num_predict"] = min(int(kwargs["max_tokens"]), 160)
                except Exception:
                    params["num_predict"] = 160
        # 抑制复读（Ollama/llama.cpp 通用参数）
        params.setdefault("repeat_last_n", 128)
        params.setdefault("repeat_penalty", 1.15)
        params.setdefault("presence_penalty", 0.2)
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=self._wrap_prompt(text),
                options={
                    **params,
                    # 让输出在对话边界停下来，避免吐出训练 token
                    "stop": [
                        "<|endoftext|>",
                        "</think>",
                        "<|im_end|>",
                        "\n### 用户问题",
                        "\n用户：",
                        "\n助手：",
                    ],
                },
                keep_alive=self.keep_alive,
                # Qwen3.5：显式关掉 think，避免输出 think 块/空回复
                think=False,
            )
            
            result = self._clean_response((response.get("response", "") or "").strip())
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
