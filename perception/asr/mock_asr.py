import numpy as np
from typing import Dict, Any, Tuple
import time

class MockFunASRRecognizer:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Mock FunASR语音识别模型
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.last_wake_time = 0
        print("[Mock ASR] 初始化成功")
    
    def inference(self, audio_data: np.ndarray) -> Tuple[bool, str]:
        """
        模拟语音识别
        
        Args:
            audio_data: 输入音频数据
            
        Returns:
            Tuple[bool, str]: (是否检测到唤醒词, 语音转文本结果)
        """
        current_time = time.time()
        
        # 每隔5秒模拟一次唤醒词检测，增加测试频率
        wake_detected = False
        if current_time - self.last_wake_time > 5:
            wake_detected = True
            self.last_wake_time = current_time
            # 随机选择一个唤醒词
            wake_words = ["你好", "导盲", "导航", "小明", "小明同学"]
            import random
            selected_wake = random.choice(wake_words)
            asr_text = f"{selected_wake}，请问前方路况如何？"
            print(f"[Mock ASR] 检测到唤醒词: {selected_wake}")
            print(f"[Mock ASR] 语音识别结果: {asr_text}")
        else:
            # 模拟普通语音识别结果
            mock_responses = [
                "前方路况如何？",
                "请问现在可以过马路吗？",
                "我需要去地铁站",
                "前面有障碍物吗？"
            ]
            import random
            asr_text = random.choice(mock_responses)
            print(f"[Mock ASR] 语音识别结果: {asr_text}")
        
        return wake_detected, asr_text
    
    def release(self):
        """
        释放资源
        """
        print("[Mock ASR] 资源已释放")
