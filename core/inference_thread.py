import gc
import threading
import time
from typing import Dict, Any, Optional, Tuple

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

        # ===== 唤醒/录句态逻辑（配置化）=====
        cfg = config_loader.get_config()
        self._wake_cfg = (cfg.get("models", {}).get("asr", {}) or {}).get("wake_logic", {}) or {}
        self.endpoint_silence_sec = float(self._wake_cfg.get("endpoint_silence_sec", 1.2))
        self.max_utterance_sec = float(self._wake_cfg.get("max_utterance_sec", 6.0))
        self.wake_cooldown_sec = float(self._wake_cfg.get("wake_cooldown_sec", 4.0))
        self.min_query_chars = int(self._wake_cfg.get("min_query_chars", 2))
        self.single_xiao_policy = str(self._wake_cfg.get("single_xiao_policy", "medium")).strip().lower()

        # 唤醒词：强唤醒（直接进入录句态）；弱唤醒（需额外条件）
        self.strong_wake_words = ["你好", "导盲", "导航", "小明同学", "小明"]
        self.weak_wake_words = ["小"]  # 中等严格：不直接触发

        # 状态机
        self.asr_state = "idle"  # idle | listening
        self.wake_phrase: str = ""
        self.query_buffer: str = ""
        self.wake_ts: float = 0.0
        self.last_speech_ts: Optional[float] = None
        self.last_wake_trigger_ts: float = 0.0

        # 弱唤醒“小”的防误触计数（按片段）
        self._xiao_hit_streak = 0
        self._xiao_last_ts: float = 0.0

        # 用于跨片段检测唤醒词的滚动文本（短窗口）
        self._wake_roll_text: str = ""
        # 存储最近的视觉数据
        self.latest_frame = None
        self.latest_metadata = None
        self.latest_vision_timestamp = 0

    @staticmethod
    def _clean_text(s: str) -> str:
        import re

        return re.sub(r"[。，、；：？！,.?!;:\s]", "", s or "")

    def _wake_cooldown_ok(self, now_ts: float) -> bool:
        return (now_ts - self.last_wake_trigger_ts) >= self.wake_cooldown_sec

    def _match_wake_word(self, text: str, clean_text: str, ts: float) -> Tuple[bool, str]:
        """
        返回 (是否唤醒, 命中的唤醒词/短语)。
        推理侧比 ASR 侧更严格，避免单字/错字导致误触。
        """
        # 强唤醒：直接命中
        for w in self.strong_wake_words:
            if w in text or w in clean_text:
                return True, w

        # 弱唤醒：单字“小”中等严格，不直接触发
        if "小" in text or "小" in clean_text:
            # 若同段或滚动窗口内出现“小明”，直接按“小明”处理
            if ("小明" in text) or ("小明" in clean_text) or ("小明" in self._wake_roll_text):
                return True, "小明"

            if self.single_xiao_policy == "medium":
                # 连续两段命中“小”才算（降低偶发错字触发）
                if ts - self._xiao_last_ts <= 8.0:
                    self._xiao_hit_streak += 1
                else:
                    self._xiao_hit_streak = 1
                self._xiao_last_ts = ts
                if self._xiao_hit_streak >= 2:
                    return True, "小"

        # 未命中：长时间无 hit 则衰减
        if ts - self._xiao_last_ts > 8.0:
            self._xiao_hit_streak = 0
        return False, ""

    @staticmethod
    def _strip_wake_prefix(text: str, wake_phrase: str) -> str:
        if not text or not wake_phrase:
            return text or ""
        t = (text or "").lstrip()
        if t.startswith(wake_phrase):
            return t[len(wake_phrase) :].lstrip("，。！？,.?!;:：；、 ")
        return text

    def _enter_listening(self, ts: float, wake_phrase: str, first_text: str, is_speech: Optional[bool]) -> None:
        self.asr_state = "listening"
        self.wake_phrase = wake_phrase
        self.query_buffer = ""
        self.wake_ts = ts
        self.last_speech_ts = ts if (is_speech is None or is_speech) else None

        rest = self._strip_wake_prefix(first_text, wake_phrase)
        if rest:
            self.query_buffer += rest
        logger.info(
            f"[Inference] 进入录句态 wake_phrase={self.wake_phrase!r} init_query={self.query_buffer!r}"
        )

    def _should_finalize(self, ts: float) -> bool:
        if self.max_utterance_sec > 0 and (ts - self.wake_ts) >= self.max_utterance_sec:
            return True
        if self.last_speech_ts is None:
            return False
        return (ts - self.last_speech_ts) >= self.endpoint_silence_sec

    def _finalize_and_call_llm(self, ts: float) -> None:
        query = (self.query_buffer or "").strip()
        clean_query = self._clean_text(query)
        if len(clean_query) < self.min_query_chars:
            logger.info("[Inference] 唤醒后内容过短，提示用户继续说问题")
            self.broadcast_scheduler.add_message(
                "我在，请说您的问题。",
                priority=3,
                alert_type="wake_word",
            )
            self.last_wake_trigger_ts = ts
            self.asr_state = "idle"
            self.wake_phrase = ""
            self.query_buffer = ""
            self._wake_roll_text = ""
            return

        full_query = f"{self.wake_phrase}{query}"
        logger.info(f"[Inference] 端点触发，调用LLM full_query={full_query!r}")

        # 是否允许带图多模态（默认可关掉以避免 Ollama 在 Jetson 上因内存紧张崩溃）
        llm_cfg = config_loader.get_config().get("models", {}).get("llm", {}) or {}
        enable_vision = bool(llm_cfg.get("enable_vision", True))

        if enable_vision and self.latest_frame is None:
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
                    response = self.complex_scene_scheduler.handle_wake_word(
                        full_query,
                        self.latest_frame if enable_vision else None,
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
                    else:
                        logger.warning("[LLM] 没有返回回复")
                finally:
                    self.resource_manager.release_resources("llm")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                logger.warning("[Inference] 无法获取LLM资源")
        finally:
            set_llm_busy(False)

        self.last_wake_trigger_ts = ts
        self.asr_state = "idle"
        self.wake_phrase = ""
        self.query_buffer = ""
        self._wake_roll_text = ""

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
        asr_text = (message.get("text") or "").strip()
        wake_detected = bool(message.get("wake_detected") or False)
        is_speech = message.get("is_speech")  # Optional[bool]
        ts = float(message.get("timestamp") or time.time())

        if not asr_text:
            return

        self.asr_results.append(asr_text)
        logger.info(f"[Inference] 已记录ASR结果: {asr_text}")

        clean_text = self._clean_text(asr_text)
        self._wake_roll_text = (self._wake_roll_text + clean_text)[-40:]

        if self.asr_state == "idle":
            if not self._wake_cooldown_ok(ts):
                return

            matched, phrase = self._match_wake_word(asr_text, clean_text, ts)
            if wake_detected and not matched:
                # ASR侧可能更宽松；推理侧仍允许进入录句态，但用更通用的标识
                matched, phrase = True, "唤醒"

            if matched:
                self._enter_listening(ts, phrase or "唤醒", asr_text, is_speech)
            return

        if self.asr_state == "listening":
            if is_speech is None or is_speech:
                self.last_speech_ts = ts

            self.query_buffer += asr_text
            logger.info(f"[Inference] 录句态累计 query_buffer={self.query_buffer!r}")

            if self._should_finalize(ts):
                self._finalize_and_call_llm(ts)
            return
    
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