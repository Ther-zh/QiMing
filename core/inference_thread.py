import threading
import time
from typing import Dict, Any

from utils.logger import logger
from utils.message_queue import message_queue
from utils.config_loader import config_loader

# 导入核心模块
from core.resource_manager import ResourceManager
from core.realtime_scheduler import RealtimeScheduler
from core.complex_scene_scheduler import ComplexSceneScheduler

# 导入融合模块
from fusion.depth_fusion import DepthFusion
from fusion.target_tracker import TargetTracker
from fusion.metadata_wrapper import MetadataWrapper

# 导入执行模块
from execution.broadcast_scheduler import BroadcastScheduler
from execution.tts_engine import TTSEngine

class InferenceThread(threading.Thread):
    """
    推理决策线程，负责安全调度和复杂场景分析
    """
    
    def __init__(self):
        super().__init__(daemon=True)
        self.running = False
        self.resource_manager = ResourceManager()
        self.realtime_scheduler = RealtimeScheduler()
        self.complex_scene_scheduler = ComplexSceneScheduler(self.resource_manager)
        self.depth_fusion = DepthFusion()
        self.target_tracker = TargetTracker()
        self.metadata_wrapper = MetadataWrapper()
        self.broadcast_scheduler = BroadcastScheduler()
        self.tts_engine = TTSEngine(config_loader.get_config().get("execution", {}))
        
        # 存储ASR和LLM结果
        self.asr_results = []
        self.llm_results = []
    
    def run(self):
        """
        线程运行方法
        """
        self.running = True
        logger.info("推理决策线程已启动")
        
        # 启动资源管理器
        self.resource_manager.start()
        
        # 启动实时安全调度
        self.realtime_scheduler.start()
        
        # 启动复杂场景调度
        self.complex_scene_scheduler.start()
        
        # 启动语音播报调度
        self.broadcast_scheduler.start()
        
        try:
            while self.running:
                # 从推理队列接收消息
                message = message_queue.receive_message("inference", block=False)
                if message:
                    msg_type = message.get("type")
                    
                    if msg_type == "asr_result":
                        # 处理ASR结果
                        self._handle_asr_result(message)
                    elif msg_type == "vision_result":
                        # 处理视觉结果
                        self._handle_vision_result(message)
                
                # 检查控制消息
                control_message = message_queue.receive_message("control", block=False)
                if control_message and control_message.get("type") == "video_ended":
                    logger.info("收到视频结束消息，准备停止系统")
                    # 发送停止消息
                    message_queue.send_message("control", {"type": "stop"})
                    break
                
                # 短暂休眠，避免占用过多CPU
                time.sleep(0.01)
        finally:
            # 停止各个模块
            self.broadcast_scheduler.stop()
            self.complex_scene_scheduler.stop()
            self.realtime_scheduler.stop()
            self.resource_manager.stop()
            
            # 释放资源
            if hasattr(self, 'tts_engine'):
                self.tts_engine.release()
            
            logger.info("推理决策线程已停止")
    
    def _handle_asr_result(self, message):
        """
        处理ASR结果
        """
        asr_text = message.get("text")
        wake_detected = message.get("wake_detected")
        timestamp = message.get("timestamp")
        
        # 记录ASR结果
        if asr_text:
            self.asr_results.append(asr_text)
            logger.info(f"[Inference] 已记录ASR结果: {asr_text}")
    
    def _handle_vision_result(self, message):
        """
        处理视觉结果
        """
        frame = message.get("frame")
        yolo_results = message.get("yolo_results")
        depth_map = message.get("depth_map")
        timestamp = message.get("timestamp")
        camera_id = message.get("camera_id")
        
        # 申请资源
        if self.resource_manager.request_resources("inference", priority=0):
            try:
                # 计算目标距离
                targets_with_distance = self.depth_fusion.calculate_target_distances(yolo_results, depth_map)
                
                # 跟踪目标并计算速度
                tracked_targets = self.target_tracker.track_targets(targets_with_distance, timestamp)
                
                # 封装元数据
                metadata = self.metadata_wrapper.wrap_metadata(
                    frame,
                    tracked_targets,
                    timestamp,
                    camera_id
                )
                
                # 处理实时安全调度
                self.realtime_scheduler.process_metadata(metadata)
                
                # 检查是否有告警
                alert = self.realtime_scheduler.get_alert()
                if alert:
                    # 高优先级处理危险警报
                    if self.resource_manager.request_resources("alert", priority=10):
                        try:
                            # 添加到语音播报队列
                            self.broadcast_scheduler.add_message(
                                alert.get("message"),
                                priority=1 if alert.get("level") == "level1" else 2,
                                alert_type=alert.get("level")
                            )
                            logger.info(f"[Alert] 危险警报: {alert.get('message')}")
                        finally:
                            self.resource_manager.release_resources("alert")
                
                # 检查是否触发复杂场景
                if self.realtime_scheduler.is_complex_scene_triggered():
                    # 中优先级处理复杂场景
                    if self.resource_manager.request_resources("llm", priority=5):
                        try:
                            # 处理复杂场景
                            response = self.complex_scene_scheduler.process_complex_scene(
                                frame,
                                metadata,
                                "请分析当前场景并提供导航建议"
                            )
                            if response:
                                self.broadcast_scheduler.add_message(
                                    response,
                                    priority=3,
                                    alert_type="complex_scene"
                                )
                                # 记录LLM结果
                                self.llm_results.append(response)
                        finally:
                            self.resource_manager.release_resources("llm")
                    # 重置触发信号
                    self.realtime_scheduler.reset_complex_scene_trigger()
            finally:
                # 释放资源
                self.resource_manager.release_resources("inference")
    
    def get_results(self):
        """
        获取ASR和LLM结果
        """
        return {
            "asr_results": self.asr_results,
            "llm_results": self.llm_results
        }
    
    def stop(self):
        """
        停止线程
        """
        self.running = False