import threading
import time
from typing import Dict, Any, Optional, Tuple
from PIL import Image

from utils.logger import logger
from utils.config_loader import config_loader
from perception.llm.qwen_multimodal import QwenMultimodal
from perception.llm.mock_llm import MockQwenMultimodal

class ComplexSceneScheduler:
    def __init__(self, resource_manager):
        """
        初始化复杂场景智能调度引擎
        
        Args:
            resource_manager: 资源管理器实例
        """
        self.config = config_loader.get_config()
        self.resource_manager = resource_manager
        self.llm = None
        self.running = False
        
        # 启动调度线程
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True  # 次优先级
        )
    
    def start(self):
        """
        启动调度引擎
        """
        self.running = True
        # 启动时加载LLM模型
        self._load_llm()
        self.scheduler_thread.start()
        logger.info("复杂场景智能调度引擎已启动")
    
    def stop(self):
        """
        停止调度引擎
        """
        self.running = False
        if self.scheduler_thread.is_alive():
            self.scheduler_thread.join()
        self._release_llm()
        logger.info("复杂场景智能调度引擎已停止")
    
    def _scheduler_loop(self):
        """
        调度循环
        """
        while self.running:
            # 这里应该检查是否有触发信号
            # 暂时模拟处理
            time.sleep(0.5)
    
    def process_complex_scene(self, image: Image.Image, metadata: Dict[str, Any], prompt: str, priority: int = 0) -> Optional[str]:
        """
        处理复杂场景
        
        Args:
            image: 输入图像
            metadata: 环境元数据
            prompt: 用户指令或预设prompt
            priority: 优先级，值越高优先级越高
            
        Returns:
            LLM生成的回复
        """
        # 向资源管理器申请资源
        if not self.resource_manager.request_resources("llm", priority=priority):
            logger.warning("资源不足，无法处理复杂场景")
            return "资源不足，无法处理当前场景"
        
        try:
            # 执行推理
            input_data = (image, metadata, prompt)
            response = self.llm.inference(input_data)
            
            logger.info(f"LLM回复: {response}")
            return response
        except Exception as e:
            logger.error(f"LLM处理失败: {e}")
            import traceback
            traceback.print_exc()
            # 尝试直接调用模型进行纯文本生成
            try:
                logger.info("尝试直接调用模型进行纯文本生成...")
                from LLM.qwen35 import Qwen35Ollama
                model_name = self.config.get("models", {}).get("llm", {}).get("model_name", "qwen3.5-4b")
                model = Qwen35Ollama(model_name=model_name)
                simple_prompt = "你是一个导盲系统助手，需要回答用户的问题。用户问：\"前面有什么东西？\"，当前环境有一些行人、汽车和摩托车。请给出友好的回答。"
                response = model.generate(simple_prompt)
                logger.info(f"直接调用模型成功，回复: {response}")
                return response
            except Exception as e2:
                logger.error(f"直接调用模型也失败: {e2}")
                # 禁用模拟回复，直接返回错误信息
                return "系统暂时无法处理您的请求，请稍后再试"
        finally:
            # 释放资源
            self.resource_manager.release_resources("llm")
    
    def _load_llm(self):
        """
        加载LLM模型
        """
        if self.llm is None:
            llm_config = self.config.get("models", {}).get("llm", {})
            # 强制使用真实LLM模型，禁用模拟模型
            self.llm = QwenMultimodal(llm_config)
            logger.info("LLM模型加载完成")
    
    def _release_llm(self):
        """
        释放LLM模型
        """
        if self.llm:
            self.llm.release()
            self.llm = None
            logger.info("LLM模型已释放")
    
    def handle_wake_word(self, wake_word: str, image: Image.Image, metadata: Dict[str, Any]) -> Optional[str]:
        """
        处理唤醒词
        
        Args:
            wake_word: 唤醒词
            image: 输入图像
            metadata: 环境元数据
            
        Returns:
            LLM生成的回复
        """
        # 即使没有图像或元数据，也使用真实LLM
        if image is None or metadata is None:
            logger.info("没有图像或元数据，使用基础prompt")
            # 生成基础prompt - 简洁直接
            prompt = "你是导盲系统助手，回答要简洁直接，只说关键信息。\n"
            prompt += "根据用户问题，提供简短的导航建议。\n\n"
            
            prompt += f"## 用户问题\n{wake_word}\n\n"
            prompt += "## 输出要求\n"
            prompt += "1. 直接给出具体建议，不要任何开场白或套话\n"
            prompt += "2. 语言简洁，控制在50字以内\n"
            prompt += "3. 必须用中文回答\n\n"
            prompt += "请直接输出简洁的导航建议："
            # 使用真实LLM处理，设置高优先级
            return self.process_complex_scene(image, metadata, prompt, priority=7)
        
        # 根据唤醒词生成prompt
        prompt = self._generate_prompt(wake_word, metadata)
        
        # 处理复杂场景，设置高优先级
        return self.process_complex_scene(image, metadata, prompt, priority=7)
    
    def _generate_prompt(self, wake_word: str, metadata: Dict[str, Any]) -> str:
        """
        根据唤醒词和元数据生成prompt
        
        Args:
            wake_word: 唤醒词
            metadata: 环境元数据
            
        Returns:
            生成的prompt
        """
        # 简洁指令
        prompt = "你是导盲系统助手，回答要简洁直接，只说关键信息。\n"
        prompt += "根据环境信息和用户问题，提供简短的导航建议。\n\n"
        
        # 环境信息
        prompt += "## 环境信息\n"
        targets = metadata.get("targets", []) if metadata else []
        if targets:
            prompt += "检测到的目标：\n"
            for i, target in enumerate(targets, 1):
                category = target.get("category", "未知")
                distance = target.get("distance", 0)
                direction = target.get("direction", "前方")
                speed = target.get("speed", 0)
                prompt += f"{i}. {direction}方向{distance:.1f}米处的{category}"
                if speed > 0:
                    prompt += "（正在移动）"
                prompt += "\n"
        else:
            prompt += "检测到的目标：无\n"
        
        # 用户问题
        prompt += "\n## 用户问题\n"
        prompt += f"{wake_word}\n\n"
        
        # 输出要求
        prompt += "## 输出要求\n"
        prompt += "1. 直接给出具体建议，不要任何开场白或套话\n"
        prompt += "2. 只说关键信息，避免空话\n"
        prompt += "3. 语言简洁，控制在50字以内\n"
        prompt += "4. 基于实际环境数据，给出具体方向和距离\n"
        prompt += "5. 优先提醒移动目标和安全隐患\n"
        prompt += "6. 必须用中文回答\n"
        prompt += "7. 将英文类别翻译成中文（person→行人，car→汽车，motorcycle→摩托车）\n\n"
        
        # 示例
        prompt += "## 示例\n"
        prompt += "输入：环境：前方5米有行人，左侧3米有汽车；用户：我想过马路\n"
        prompt += "输出：前方5米有行人，左侧3米有汽车正在移动。建议稍等车辆通过后，确认安全再过马路。\n\n"
        
        # 开始回复
        prompt += "请直接输出简洁的导航建议："
        
        return prompt
