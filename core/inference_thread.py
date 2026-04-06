import gc
import threading
import time
from typing import Dict, Any, Optional

import cv2
import torch

from utils.logger import logger
from utils.message_queue import message_queue
from utils.config_loader import config_loader
from utils.llm_gate import set_llm_busy

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
from utils.video_record import depth_to_colormap_bgr, draw_tracked_targets, overlay_depth_on_frame

class InferenceThread(threading.Thread):
    """
    推理决策线程，负责安全调度和复杂场景分析
    """
    
    def __init__(self, resource_manager: Optional[ResourceManager] = None):
        super().__init__(daemon=True)
        self.running = False
        self.resource_manager = resource_manager or ResourceManager()
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
        # 唤醒词列表
        self.wake_words = ["你好", "导盲", "导航", "小明", "小明同学", "小"]
        # 累积的ASR文本（用于检测跨片段的唤醒词）
        self.cumulative_asr_text = ""
        # 存储最近的视觉数据
        self.latest_frame = None
        self.latest_metadata = None
        self.latest_vision_timestamp = 0

    def _trim_cuda_before_llm(self) -> None:
        """在调用 Ollama 前尽量压低 PyTorch CUDA 峰值，减轻 Jetson 统一内存 OOM。"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
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
        
        video_ended = False
        video_ended_time = None
        
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
                    if not video_ended:
                        logger.info("收到视频结束消息，继续运行15秒处理剩余消息...")
                        video_ended = True
                        video_ended_time = time.time()
                
                # 如果视频已结束，检查是否已经过了足够长的时间
                if video_ended and video_ended_time:
                    if time.time() - video_ended_time >= 15:
                        logger.info("15秒等待时间已过，准备停止系统")
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
        import re
        asr_text = message.get("text")
        wake_detected = message.get("wake_detected")
        timestamp = message.get("timestamp")

        # 记录ASR结果
        if asr_text:
            self.asr_results.append(asr_text)
            logger.info(f"[Inference] 已记录ASR结果: {asr_text}")
            
            # 累积ASR文本，用于跨片段唤醒词检测
            self.cumulative_asr_text += asr_text
            logger.info(f"[Inference] 当前累积ASR文本: '{self.cumulative_asr_text}'")
            
            # 去除标点符号后的累积文本
            clean_cumulative_text = re.sub(r'[。，、；：？！,.?!;:\s]', '', self.cumulative_asr_text)
            logger.info(f"[Inference] 去除标点后的累积文本: '{clean_cumulative_text}'")
            
            # 如果ASR端没检测到，但推理端检测到了唤醒词，也触发！
            if not wake_detected:
                logger.info(f"[Inference] ASR端未检测到唤醒词，开始跨片段检测...")
                for word in self.wake_words:
                    if word in self.cumulative_asr_text or word in clean_cumulative_text:
                        wake_detected = True
                        logger.info(f"[Inference] 跨片段检测到唤醒词: '{word}'")
                        break
        
        # 如果检测到唤醒词，调用LLM处理
        if wake_detected:
            logger.info(f"[Inference] 检测到唤醒词，准备调用LLM处理...")
            logger.info(f"[Inference] 完整查询文本: '{self.cumulative_asr_text}'")
            # 首帧视觉可能尚未到达，或推理资源曾拒绝导致未缓存帧；短暂等待以免误落「无图」分支
            if self.latest_frame is None:
                _deadline = time.time() + 2.5
                while self.latest_frame is None and self.running and time.time() < _deadline:
                    time.sleep(0.05)
                if self.latest_frame is not None:
                    logger.info("[Inference] 已等到最近一帧视觉，将结合图像调用 LLM")
            self._trim_cuda_before_llm()
            set_llm_busy(True)
            try:
                if self.resource_manager.request_resources("llm", priority=5):
                    try:
                        full_query = self.cumulative_asr_text
                        response = self.complex_scene_scheduler.handle_wake_word(
                            full_query,
                            self.latest_frame,
                            self.latest_metadata,
                        )
                        if response:
                            logger.info(f"[LLM] 回复: {response}")
                            self.broadcast_scheduler.add_message(
                                response,
                                priority=3,
                                alert_type="wake_word",
                            )
                            self.llm_results.append(response)
                            self.cumulative_asr_text = ""
                        else:
                            logger.warning(f"[LLM] 没有返回回复")
                    finally:
                        self.resource_manager.release_resources("llm")
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    logger.warning(f"[Inference] 无法获取LLM资源")
            finally:
                set_llm_busy(False)
    
    def _handle_vision_result(self, message):
        """
        处理视觉结果
        """
        frame = message.get("frame")
        yolo_results = message.get("yolo_results")
        depth_map = message.get("depth_map")
        timestamp = message.get("timestamp")
        camera_id = message.get("camera_id")

        # 无论推理资源是否申请成功，都缓存最近一帧，避免资源阈值拒绝时唤醒词 LLM 拿不到图
        if frame is not None:
            self.latest_frame = frame
            self.latest_vision_timestamp = timestamp

        # 申请资源（priority>3 时若 CPU/内存瞬时超标仍允许融合，避免 8GB+ASR 峰值下长期 fusion_skipped）
        if self.resource_manager.request_resources("inference", priority=4):
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
                
                # 更新最近的视觉数据（完整融合元数据）
                self.latest_metadata = metadata
                
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
                    self._trim_cuda_before_llm()
                    set_llm_busy(True)
                    try:
                        if self.resource_manager.request_resources("llm", priority=5):
                            try:
                                response = self.complex_scene_scheduler.process_complex_scene(
                                    frame,
                                    metadata,
                                    "请分析当前场景并提供导航建议",
                                    manage_resources=False,
                                )
                                if response:
                                    self.broadcast_scheduler.add_message(
                                        response,
                                        priority=3,
                                        alert_type="complex_scene",
                                    )
                                    self.llm_results.append(response)
                            finally:
                                self.resource_manager.release_resources("llm")
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                    finally:
                        set_llm_busy(False)
                    # 重置触发信号
                    self.realtime_scheduler.reset_complex_scene_trigger()

                # 录制可视化帧（主线程写入视频文件）
                exec_cfg = config_loader.get_config().get("execution", {})
                if (
                    exec_cfg.get("save_video_output", True)
                    and frame is not None
                    and depth_map is not None
                ):
                    cam_wh = (
                        config_loader.get_config()
                        .get("cameras", {})
                        .get("camera1", {})
                        .get("resolution", [640, 480])
                    )
                    cam_w, cam_h = int(cam_wh[0]), int(cam_wh[1])
                    fr_r = cv2.resize(frame, (cam_w, cam_h), interpolation=cv2.INTER_AREA)
                    oh, ow = frame.shape[:2]
                    scaled_targets = []
                    for t in tracked_targets:
                        nt = dict(t)
                        x1, y1, x2, y2 = t.get("bbox", [0, 0, 0, 0])
                        nt["bbox"] = [
                            x1 * cam_w / ow,
                            y1 * cam_h / oh,
                            x2 * cam_w / ow,
                            y2 * cam_h / oh,
                        ]
                        scaled_targets.append(nt)
                    main_bgr = overlay_depth_on_frame(
                        draw_tracked_targets(fr_r.copy(), scaled_targets),
                        depth_map,
                        alpha=0.35,
                    )
                    payload: Dict[str, Any] = {"main": main_bgr}
                    if exec_cfg.get("save_depth_colormap_video", True):
                        dm_r = cv2.resize(
                            depth_map, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR
                        )
                        payload["depth"] = depth_to_colormap_bgr(dm_r)
                    message_queue.send_message("record", payload)
            finally:
                # 释放资源
                self.resource_manager.release_resources("inference")
        else:
            # 资源不足未跑融合时，仍提供非 None 元数据，便于带图 prompt（目标列表为空）
            if frame is not None:
                self.latest_metadata = {
                    "targets": [],
                    "timestamp": timestamp,
                    "camera_id": camera_id,
                    "fusion_skipped": True,
                }
                logger.debug(
                    "[Inference] 推理资源未申请成功，已仅缓存图像与空目标元数据（供唤醒词多模态）"
                )

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