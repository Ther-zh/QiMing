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
    
    def process_complex_scene(self, image: Image.Image, metadata: Dict[str, Any], prompt: str) -> Optional[str]:
        """
        处理复杂场景
        
        Args:
            image: 输入图像
            metadata: 环境元数据
            prompt: 用户指令或预设prompt
            
        Returns:
            LLM生成的回复
        """
        # 向资源管理器申请资源
        if not self.resource_manager.request_resources("llm"):
            logger.warning("资源不足，无法处理复杂场景")
            return None
        
        try:
            # 加载LLM
            self._load_llm()
            
            # 执行推理
            input_data = (image, metadata, prompt)
            response = self.llm.inference(input_data)
            
            logger.info(f"LLM回复: {response}")
            return response
        finally:
            # 释放LLM和资源
            self._release_llm()
            self.resource_manager.release_resources("llm")
    
    def _load_llm(self):
        """
        加载LLM模型
        """
        if self.llm is None:
            llm_config = self.config.get("models", {}).get("llm", {})
            llm_type = llm_config.get("type", "mock")
            
            if llm_type == "real":
                self.llm = QwenMultimodal(llm_config)
            else:
                self.llm = MockQwenMultimodal(llm_config)
            
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
        # 根据唤醒词生成prompt
        prompt = self._generate_prompt(wake_word, metadata)
        
        # 处理复杂场景
        return self.process_complex_scene(image, metadata, prompt)
    
    def _generate_prompt(self, wake_word: str, metadata: Dict[str, Any]) -> str:
        """
        根据唤醒词和元数据生成prompt
        
        Args:
            wake_word: 唤醒词
            metadata: 环境元数据
            
        Returns:
            生成的prompt
        """
        # Instruct模式prompt
        prompt = "你是一个专业的导盲系统助手，致力于为视障人士提供安全、准确、清晰的导航指导。\n"
        prompt += "请根据以下环境信息和用户问题，直接提供详细、准确的导航建议。\n\n"
        
        # 环境信息
        prompt += "## 环境信息\n"
        targets = metadata.get("targets", [])
        if targets:
            prompt += "检测到的目标：\n"
            for i, target in enumerate(targets, 1):
                category = target.get("category", "未知")
                distance = target.get("distance", 0)
                direction = target.get("direction", "前方")
                speed = target.get("speed", 0)
                prompt += f"{i}. {direction}方向{distance:.1f}米处的{category}"
                if speed > 0:
                    prompt += f"（移动速度：{speed:.1f}m/s）"
                prompt += "\n"
        else:
            prompt += "检测到的目标：无\n"
        
        # 用户问题
        prompt += "\n## 用户问题\n"
        prompt += f"{wake_word}\n\n"
        
        # 输出要求
        prompt += "## 输出要求\n"
        prompt += "1. 直接提供导航建议，不要包含思考过程\n"
        prompt += "2. 语言简洁明了，避免使用复杂句子\n"
        prompt += "3. 信息准确，基于当前环境数据\n"
        prompt += "4. 优先考虑用户安全\n"
        prompt += "5. 提供具体的导航建议，包括方向、距离和注意事项\n"
        prompt += "6. 如果有多个目标，按优先级排序（距离最近的优先）\n"
        prompt += "7. 对于移动目标，特别提醒用户注意\n\n"
        
        # 示例
        prompt += "## 示例\n"
        prompt += "输入：环境：前方5米处有一个行人，左侧3米处有一辆汽车；用户：我想过马路\n"
        prompt += "输出：当前前方5米处有一个行人，左侧3米处有一辆汽车。目前车辆距离较近，建议等待车辆通过后再过马路。当车辆通过后，确认左右方向安全，然后以正常步速穿过马路。\n\n"
        
        # 开始回复
        prompt += "请直接输出导航建议："
        
        return prompt
