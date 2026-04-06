import os
import numpy as np
from typing import Dict, Any, Tuple, Optional
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2VLForConditionalGeneration, AutoProcessor

class QwenMultimodalTransformers:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Qwen多模态大模型（基于transformers）
        
        Args:
            config: 配置字典，包含模型路径等参数
        """
        self.config = config
        self.model_path = config.get('model_path', '/home/nvidia/models/root/autodl-tmp/qwen35/tclf90/Qwen3___5-4B-AWQ')
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """
        加载Qwen模型
        """
        try:
            print(f"[LLM] 正在加载模型: {self.model_path}")
            print(f"[LLM] 使用设备: {self.device}")
            
            # 加载tokenizer和processor
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 尝试加载processor（用于多模态）
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
            except:
                self.processor = None
                print("[LLM] 警告: 无法加载processor，将使用纯文本模式")
            
            print(f"[LLM] 模型加载成功: {self.model_path}")
        except Exception as e:
            print(f"[LLM] 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
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
        
        try:
            # 构建输入文本
            input_text = self._build_prompt(prompt, metadata)
            
            # 如果有图像且processor可用，使用多模态推理
            if image is not None and self.processor is not None:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": input_text},
                        ],
                    }
                ]
                
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt",
                    padding=True
                )
                
                # 移动到正确的设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 生成回复
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9
                    )
                
                # 解码输出
                response = self.processor.batch_decode(
                    outputs, skip_special_tokens=True
                )[0]
                
            else:
                # 纯文本推理
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # 移动到正确的设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 生成回复
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9
                    )
                
                # 解码输出
                response = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
            
            # 清理输出，确保不超过100字
            response = self._clean_response(response)
            
            return response[:100]
            
        except Exception as e:
            print(f"[LLM] 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return "系统繁忙，请稍后再试。"
    
    def _build_prompt(self, prompt: str, metadata: Dict[str, Any]) -> str:
        """
        构建提示词
        
        Args:
            prompt: 用户指令
            metadata: 环境元数据
            
        Returns:
            构建好的提示词
        """
        # 分析metadata中的目标信息
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
        
        # 构建完整的提示词
        full_prompt = f"""你是一个导盲系统助手，需要根据环境信息回答用户的问题，回答要简洁友好，不超过100字。

环境信息：
{target_info}

用户问题：{prompt}

请给出友好的回答："""
        
        return full_prompt
    
    def _clean_response(self, response: str) -> str:
        """
        清理模型输出
        
        Args:
            response: 原始输出
            
        Returns:
            清理后的输出
        """
        # 移除多余的空白字符
        response = response.strip()
        
        # 移除重复的标点符号
        while "。。" in response:
            response = response.replace("。。", "。")
        
        # 移除回答标记
        if "回答：" in response:
            response = response.split("回答：")[-1].strip()
        
        return response
    
    def release(self):
        """
        释放模型资源
        """
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        # 清理GPU缓存
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        print("[LLM] 模型资源已释放")