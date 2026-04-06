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
        self.model_name = config.get("model_name", "qwen3.5-4b")
        self.model_name_text = config.get("model_name_text") or self.model_name
        self.model = None
        self._text_model = None
        self._load_model()
    
    def _load_model(self):
        """
        加载Qwen模型 (使用Ollama)
        """
        try:
            import sys
            # 添加项目根目录到路径
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from LLM.qwen35 import Qwen35Ollama
            # 未配置时默认 False：Qwen3.5 须在 API 顶层 think=False，不能写在 options（ollama#14793）
            _think = self.config["ollama_think"] if "ollama_think" in self.config else False
            _opts = dict(self.config.get("ollama_options") or {})
            _fb = self.config.get("fallback_phrase") or None
            _kw = dict(think=_think, ollama_options=_opts)
            if _fb:
                _kw["fallback_phrase"] = _fb
            self.model = Qwen35Ollama(model_name=self.model_name, **_kw)
            print(f"[LLM] Ollama 多模态模型就绪: {self.model_name}")
            if self.model_name_text != self.model_name:
                self._text_model = Qwen35Ollama(model_name=self.model_name_text, **_kw)
                print(f"[LLM] Ollama 纯文本模型就绪: {self.model_name_text}")
        except Exception as e:
            print(f"[LLM] 模型加载失败: {e}")
            raise
    
    def inference(self, input_data: Tuple[Optional[Image.Image], Dict[str, Any], str]) -> str:
        """
        执行多模态推理
        
        Args:
            input_data: Tuple[image, metadata, prompt]，包含图像、环境元数据和用户指令
            
        Returns:
            生成的口语化文本（长度由 config models.llm.max_reply_chars 控制，默认与 module_debug 对齐）
        """
        if self.model is None:
            raise RuntimeError("LLM模型未加载")
        
        image, metadata, prompt = input_data
        max_tok = int(self.config.get("max_generate_tokens", 512))
        max_chars = int(self.config.get("max_reply_chars", 512))

        if image is None and self._text_model is not None:
            response = self._text_model.generate(text=prompt, image=None, max_tokens=max_tok)
        else:
            tfm = (
                self.model_name_text
                if self.model_name_text != self.model_name
                else None
            )
            response = self.model.generate(
                text=prompt,
                image=image if image is not None else None,
                max_tokens=max_tok,
                text_fallback_model=tfm,
            )

        return response[:max_chars] if max_chars > 0 else response
    
    def release(self):
        """
        释放模型资源
        """
        if self.model:
            self.model = None
        if self._text_model:
            self._text_model = None
        print("[LLM] 模型资源已释放")
