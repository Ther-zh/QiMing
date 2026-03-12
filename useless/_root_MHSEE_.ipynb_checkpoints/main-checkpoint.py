import time
import threading
import signal
import sys
from typing import Dict, Any

# 导入工具模块
from utils.logger import logger
from utils.config_loader import config_loader

# 导入核心模块
from core.resource_manager import ResourceManager
from core.realtime_scheduler import RealtimeScheduler
from core.complex_scene_scheduler import ComplexSceneScheduler

# 导入感知模块
from perception.yolo.yolo_detector import YoloDetector
from perception.yolo.mock_yolo import MockYoloDetector
from perception.vda.vda_depth import VDADepthEstimator
from perception.vda.mock_vda import MockVDADepthEstimator
from perception.asr.funasr_asr import FunASRRecognizer
from perception.asr.mock_asr import MockFunASRRecognizer
from perception.llm.qwen_multimodal import QwenMultimodal
from perception.llm.mock_llm import MockQwenMultimodal

# 导入融合模块
from fusion.frame_sync import FrameSync
from fusion.depth_fusion import DepthFusion
from fusion.target_tracker import TargetTracker
from fusion.metadata_wrapper import MetadataWrapper

# 导入执行模块
from execution.broadcast_scheduler import BroadcastScheduler
from execution.tts_engine import TTSEngine

# 导入模拟模块
from simulation.camera_simulator import CameraSimulator
from simulation.debug_viewer import DebugViewer

class BlindGuideSystem:
    def __init__(self):
        """
        初始化导盲系统
        """
        # 加载配置
        self.config = config_loader.get_config()
        self.risk_rules = config_loader.get_risk_rules()
        
        # 初始化资源管理器
        self.resource_manager = ResourceManager()
        
        # 初始化模拟模块
        self.camera_simulator = CameraSimulator()
        self.debug_viewer = DebugViewer()
        
        # 初始化感知模块
        self.yolo = self._init_yolo()
        self.vda = self._init_vda()
        self.asr = self._init_asr()
        self.llm = self._init_llm()
        
        # 初始化融合模块
        self.frame_sync = FrameSync()
        self.depth_fusion = DepthFusion()
        self.target_tracker = TargetTracker()
        self.metadata_wrapper = MetadataWrapper()
        
        # 初始化核心调度模块
        self.realtime_scheduler = RealtimeScheduler()
        self.complex_scene_scheduler = ComplexSceneScheduler(self.resource_manager)
        
        # 初始化执行模块
        self.broadcast_scheduler = BroadcastScheduler()
        self.tts_engine = TTSEngine(self.config.get("execution", {}))
        
        # 系统状态
        self.running = False
    
    def _init_yolo(self):
        """
        初始化YOLO模块
        """
        yolo_config = self.config.get("models", {}).get("yolo", {})
        if yolo_config.get("type", "mock") == "real":
            return YoloDetector(yolo_config)
        else:
            return MockYoloDetector(yolo_config)
    
    def _init_vda(self):
        """
        初始化VDA模块
        """
        vda_config = self.config.get("models", {}).get("vda", {})
        if vda_config.get("type", "mock") == "real":
            return VDADepthEstimator(vda_config)
        else:
            return MockVDADepthEstimator(vda_config)
    
    def _init_asr(self):
        """
        初始化ASR模块
        """
        asr_config = self.config.get("models", {}).get("asr", {})
        if asr_config.get("type", "mock") == "real":
            return FunASRRecognizer(asr_config)
        else:
            return MockFunASRRecognizer(asr_config)
    
    def _init_llm(self):
        """
        初始化LLM模块
        """
        llm_config = self.config.get("models", {}).get("llm", {})
        if llm_config.get("type", "mock") == "real":
            return QwenMultimodal(llm_config)
        else:
            return MockQwenMultimodal(llm_config)
    
    def start(self):
        """
        启动系统
        """
        logger.info("导盲系统启动中...")
        
        # 启动资源管理器
        self.resource_manager.start()
        
        # 启动摄像头模拟
        self.camera_simulator.start()
        
        # 启动帧同步
        self.frame_sync.start()
        
        # 启动实时安全调度
        self.realtime_scheduler.start()
        
        # 启动复杂场景调度
        self.complex_scene_scheduler.start()
        
        # 启动语音播报调度
        self.broadcast_scheduler.start()
        
        # 启动调试查看器（如果开启）
        if self.config.get("system", {}).get("debug", False):
            self.debug_viewer.start()
            debug_thread = threading.Thread(
                target=self.debug_viewer.show,
                daemon=True
            )
            debug_thread.start()
        
        self.running = True
        logger.info("导盲系统启动完成")
        
        # 启动主循环
        self._main_loop()
    
    def stop(self):
        """
        停止系统
        """
        logger.info("导盲系统停止中...")
        
        self.running = False
        
        # 停止各个模块
        self.broadcast_scheduler.stop()
        self.complex_scene_scheduler.stop()
        self.realtime_scheduler.stop()
        self.frame_sync.stop()
        self.camera_simulator.stop()
        self.resource_manager.stop()
        
        # 停止调试查看器
        if self.config.get("system", {}).get("debug", False):
            self.debug_viewer.stop()
        
        # 释放模型资源
        if hasattr(self, 'yolo'):
            self.yolo.release()
        if hasattr(self, 'vda'):
            self.vda.release()
        if hasattr(self, 'asr'):
            self.asr.release()
        if hasattr(self, 'llm'):
            self.llm.release()
        if hasattr(self, 'tts_engine'):
            self.tts_engine.release()
        
        logger.info("导盲系统已停止")
    
    def _main_loop(self):
        """
        主循环
        """
        while self.running:
            try:
                # 更新资源管理器心跳
                self.resource_manager.update_heartbeat("main")
                
                # 获取摄像头帧
                frames = self.camera_simulator.get_all_frames()
                
                # 处理主摄画面
                main_camera = "camera1"  # 主摄
                if main_camera in frames:
                    frame, timestamp = frames[main_camera]
                    if frame is not None:
                        # 添加帧到同步模块
                        self.frame_sync.add_frame(frame, timestamp, 0)
                        
                        # 执行YOLO检测
                        yolo_results = self.yolo.inference(frame)
                        self.frame_sync.add_yolo_result(yolo_results, timestamp)
                        
                        # 执行VDA深度估计
                        depth_map = self.vda.inference(frame)
                        self.frame_sync.add_vda_result(depth_map, timestamp)
                        
                        # 获取同步数据
                        sync_data = self.frame_sync.get_sync_data()
                        if sync_data:
                            sync_frame, sync_yolo, sync_depth, sync_timestamp, sync_camera_id = sync_data
                            
                            # 计算目标距离
                            targets_with_distance = self.depth_fusion.calculate_target_distances(sync_yolo, sync_depth)
                            
                            # 跟踪目标并计算速度
                            tracked_targets = self.target_tracker.track_targets(targets_with_distance, sync_timestamp)
                            
                            # 封装元数据
                            metadata = self.metadata_wrapper.wrap_metadata(
                                sync_frame,
                                tracked_targets,
                                sync_timestamp,
                                sync_camera_id
                            )
                            
                            # 处理实时安全调度
                            self.realtime_scheduler.process_metadata(metadata)
                            
                            # 检查是否有告警
                            alert = self.realtime_scheduler.get_alert()
                            if alert:
                                # 添加到语音播报队列
                                self.broadcast_scheduler.add_message(
                                    alert.get("message"),
                                    priority=1 if alert.get("level") == "level1" else 2,
                                    alert_type=alert.get("level")
                                )
                            
                            # 检查是否触发复杂场景
                            if self.realtime_scheduler.is_complex_scene_triggered():
                                # 处理复杂场景
                                response = self.complex_scene_scheduler.process_complex_scene(
                                    sync_frame,
                                    metadata,
                                    "请分析当前场景并提供导航建议"
                                )
                                if response:
                                    self.broadcast_scheduler.add_message(
                                        response,
                                        priority=3,
                                        alert_type="complex_scene"
                                    )
                                # 重置触发信号
                                self.realtime_scheduler.reset_complex_scene_trigger()
                            
                            # 更新调试画面
                            if self.config.get("system", {}).get("debug", False):
                                # 确定危险等级
                                risk_level = "level4"  # 默认安全
                                if alert:
                                    risk_level = alert.get("level")
                                
                                self.debug_viewer.update_frame(
                                    sync_frame,
                                    tracked_targets,
                                    sync_depth,
                                    risk_level
                                )
                
                # 控制帧率
                time.sleep(0.2)  # 5FPS
                
            except Exception as e:
                logger.error(f"主循环出错: {e}")
                time.sleep(0.5)
    
    def signal_handler(self, sig, frame):
        """
        信号处理函数
        """
        logger.info("收到停止信号，正在停止系统...")
        self.stop()
        sys.exit(0)

def main():
    """
    主函数
    """
    # 创建系统实例
    system = BlindGuideSystem()
    
    # 注册信号处理
    signal.signal(signal.SIGINT, system.signal_handler)
    signal.signal(signal.SIGTERM, system.signal_handler)
    
    try:
        # 启动系统
        system.start()
    except KeyboardInterrupt:
        logger.info("用户中断，正在停止系统...")
    finally:
        # 停止系统
        system.stop()

if __name__ == "__main__":
    main()
