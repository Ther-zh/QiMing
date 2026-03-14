# 导盲系统架构技术报告

## 1. 系统整体架构

### 1.1 架构概述

导盲系统采用模块化、分层架构设计，以实时感知、智能决策和安全执行为核心，构建了一个多模态融合的辅助导航系统。系统通过摄像头、深度传感器和麦克风等多源数据输入，经过感知、融合、决策和执行四个主要环节，为视障人士提供实时的环境感知和导航辅助。

### 1.2 核心数据流

系统的核心数据流如下：

```
摄像头/麦克风 → 感知模块 → 融合模块 → 核心调度模块 → 执行模块 → 用户
```

1. **数据采集**：通过 CameraSimulator 模拟摄像头输入
2. **感知处理**：YOLO 目标检测、VDA 深度估计、ASR 语音识别、LLM 多模态理解
3. **数据融合**：帧同步、深度融合、目标跟踪、元数据封装
4. **智能决策**：实时安全调度、复杂场景调度
5. **执行反馈**：语音播报、调试可视化

### 1.3 模块间交互机制

系统采用事件驱动和轮询相结合的交互方式，各模块通过明确的接口进行数据交换：

- **感知模块**：提供原始感知数据给融合模块
- **融合模块**：将多源数据融合后生成环境元数据
- **核心调度模块**：基于元数据进行风险评估和决策
- **执行模块**：根据调度结果执行相应的操作
- **资源管理器**：统一管理系统资源，协调模块间资源分配

### 1.4 任务调度逻辑

系统采用两级调度机制：

1. **实时安全调度**：最高优先级，处理紧急安全告警
2. **复杂场景调度**：次优先级，处理需要深度理解的复杂场景

调度逻辑流程：
- 实时调度器持续监控环境元数据，评估危险等级
- 当检测到高风险目标时，立即触发安全告警
- 当场景复杂度超过阈值时，触发复杂场景处理
- 复杂场景处理通过 LLM 进行深度分析，生成导航建议

## 2. 功能模块详细设计

### 2.1 核心模块

#### 2.1.1 资源管理器 (ResourceManager)

**设计思路**：统一管理系统资源，监控模块心跳，确保系统稳定运行。

**实现细节**：
- 资源状态管理：跟踪各类资源的使用状态
- 系统资源监控：实时监控 CPU、内存、GPU 使用率
- 模块心跳检测：监控各模块的运行状态，及时发现异常
- 资源申请与释放：提供资源申请接口，确保资源合理分配

**关键代码**：
```python
# 资源申请逻辑
def request_resources(self, resource_type: str) -> bool:
    with self.lock:
        # 检查资源是否可用
        if not self.resources[resource_type]:
            # 检查系统资源
            if self._check_system_resources():
                self.resources[resource_type] = True
                logger.info(f"资源 {resource_type} 申请成功")
                return True
            else:
                logger.warning(f"系统资源不足，无法申请 {resource_type}")
                return False
        else:
            logger.warning(f"资源 {resource_type} 已被占用")
            return False
```

#### 2.1.2 实时安全调度器 (RealtimeScheduler)

**设计思路**：实时评估环境危险等级，生成安全告警，确保用户安全。

**实现细节**：
- 危险评分计算：使用 RiskEvaluator 进行综合风险评估
- 场景复杂度评估：基于目标类型和数量计算场景复杂度
- 告警级别评估：根据风险分确定告警级别
- 告警触发机制：考虑连续告警帧数和冷却时间，避免误报
- 特殊场景处理：针对人行横道、交叉路口等特殊场景进行专门处理

**关键代码**：
```python
# 处理环境元数据
def process_metadata(self, metadata: Dict[str, Any]):
    timestamp = metadata.get("timestamp")
    
    # 使用新的风险评价器
    risk_result = self.risk_evaluator.evaluate_risk(metadata)
    
    # 检查特殊场景
    special_scene_result = self.risk_evaluator.evaluate_special_scene(metadata)
    if special_scene_result:
        risk_result = special_scene_result
    
    # 评估危险等级
    alert = self._evaluate_risk_level(
        risk_result['risk_score'],
        0,  # 场景复杂度已在新评价器中考虑
        risk_result.get('target_info'),
        timestamp,
        risk_result.get('risk_level'),
        risk_result.get('message')
    )
    
    if alert:
        self.alert_queue.append(alert)
```

#### 2.1.3 风险评价器 (RiskEvaluator)

**设计思路**：基于 AHP 层次分析法和模糊综合评价，构建科学的危险评价体系。

**实现细节**：
- AHP 权重计算：使用层次分析法计算各因素权重
- 模糊综合评价：使用梯形隶属度函数进行模糊评价
- 特殊场景处理：针对人行横道、交叉路口、施工区域等特殊场景
- 风险等级评估：综合考虑距离、速度、类别和场景复杂度

**关键代码**：
```python
# AHP权重计算
def _calculate_ahp_weights(self) -> Dict[str, float]:
    # 构造判断矩阵（1-9标度法）
    judgment_matrix = np.array([
        [1, 3, 2, 4],  # 距离
        [1/3, 1, 1/2, 2],  # 速度
        [1/2, 2, 1, 3],  # 类别
        [1/4, 1/2, 1/3, 1]  # 场景复杂度
    ])
    
    # 计算权重
    eigenvalues, eigenvectors = np.linalg.eig(judgment_matrix)
    max_eigenvalue = np.max(eigenvalues)
    max_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
    
    # 归一化权重
    weights = max_eigenvector.real / np.sum(max_eigenvector.real)
    
    return {
        'distance': weights[0],
        'speed': weights[1],
        'category': weights[2],
        'scene_complexity': weights[3]
    }
```

#### 2.1.4 复杂场景调度器 (ComplexSceneScheduler)

**设计思路**：处理需要深度理解的复杂场景，通过 LLM 生成导航建议。

**实现细节**：
- 资源管理：申请和释放 LLM 资源
- LLM 模型管理：按需加载和释放 LLM 模型
- 场景处理：接收图像和元数据，通过 LLM 进行分析
- 唤醒词处理：根据用户语音指令生成相应的导航建议

**关键代码**：
```python
# 复杂场景处理
def process_complex_scene(self, image: Image.Image, metadata: Dict[str, Any], prompt: str) -> Optional[str]:
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
```

### 2.2 感知模块

#### 2.2.1 YOLO 目标检测 (YoloDetector)

**设计思路**：实时检测环境中的目标，为后续分析提供基础数据。

**实现细节**：
- 模型加载：支持真实模型和模拟模型
- 目标检测：识别环境中的人、车、障碍物等目标
- 结果输出：返回目标类别、位置等信息

#### 2.2.2 VDA 深度估计 (VDADepthEstimator)

**设计思路**：估计目标距离，为风险评估提供距离信息。

**实现细节**：
- 深度图生成：根据输入图像生成深度图
- 距离计算：基于深度图计算目标距离

#### 2.2.3 ASR 语音识别 (FunASRRecognizer)

**设计思路**：识别用户语音指令，支持唤醒词检测。

**实现细节**：
- 模型加载：加载 FunASR 语音识别模型
- 语音识别：将音频数据转换为文本
- 唤醒词检测：检测预设的唤醒词

**关键代码**：
```python
# 语音识别与唤醒词检测
def inference(self, audio_data: np.ndarray) -> Tuple[bool, str]:
    if self.model is None:
        raise RuntimeError("ASR模型未加载")
    
    # 执行语音识别
    result = self.model.predict(
        audio_data=audio_data,
        task="asr",
        vad_silence_time=0.8,
        punc=True
    )
    
    asr_text = result[0]["text"].strip()
    
    # 简单的唤醒词检测
    wake_words = ["你好", "导盲", "导航","小明","小明同学"]
    wake_detected = any(word in asr_text for word in wake_words)
    
    return wake_detected, asr_text
```

#### 2.2.4 LLM 多模态理解 (QwenMultimodal)

**设计思路**：理解复杂场景，生成导航建议。

**实现细节**：
- 多模态输入：接收图像、元数据和文本指令
- 场景理解：分析环境情况
- 建议生成：生成安全的导航建议

### 2.3 融合模块

#### 2.3.1 帧同步 (FrameSync)

**设计思路**：同步多源数据，确保数据一致性。

**实现细节**：
- 帧管理：管理摄像头帧、YOLO 结果、VDA 结果
- 数据同步：根据时间戳同步多源数据

#### 2.3.2 深度融合 (DepthFusion)

**设计思路**：融合 YOLO 检测结果和 VDA 深度图，计算目标距离。

**实现细节**：
- 目标距离计算：基于深度图计算每个目标的距离
- 异常值过滤：过滤深度图中的异常值
- 距离估计：使用中值滤波提高距离估计准确性

**关键代码**：
```python
# 计算目标距离
def _calculate_distance(self, target: Dict[str, Any], depth_map: np.ndarray) -> float:
    try:
        # 获取目标的ROI坐标
        roi_coords = target.get("roi_coords", [])
        if len(roi_coords) != 4:
            return 0.0
        
        x1, y1, x2, y2 = roi_coords
        
        # 转换为整数坐标
        height, width = depth_map.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width - 1, int(x2))
        y2 = min(height - 1, int(y2))
        
        # 提取ROI区域的深度值
        roi_depth = depth_map[y1:y2, x1:x2]
        
        # 剔除边缘10%像素
        edge_percent = 0.1
        h, w = roi_depth.shape
        h_start = int(h * edge_percent)
        h_end = int(h * (1 - edge_percent))
        w_start = int(w * edge_percent)
        w_end = int(w * (1 - edge_percent))
        
        center_roi = roi_depth[h_start:h_end, w_start:w_end]
        
        # 过滤异常值
        filtered_depth = center_roi[(center_roi >= 0.1) & (center_roi <= 20.0)]
        
        if len(filtered_depth) == 0:
            return 0.0
        
        # 取中值作为目标距离
        distance = np.median(filtered_depth)
        
        return float(distance)
    except Exception as e:
        logger.error(f"计算目标距离时出错: {e}")
        return 0.0
```

#### 2.3.3 目标跟踪 (TargetTracker)

**设计思路**：跟踪目标运动，计算目标速度。

**实现细节**：
- 目标关联：将当前帧的目标与历史帧关联
- 速度计算：基于目标位置变化计算速度
- 轨迹管理：维护目标的运动轨迹

#### 2.3.4 元数据封装 (MetadataWrapper)

**设计思路**：封装环境信息，生成标准化的元数据。

**实现细节**：
- 数据整合：整合帧、目标、时间戳等信息
- 格式标准化：生成统一格式的元数据

### 2.4 执行模块

#### 2.4.1 广播调度器 (BroadcastScheduler)

**设计思路**：管理语音播报队列，确保重要信息优先播报。

**实现细节**：
- 消息队列：维护不同优先级的消息队列
- 调度策略：根据优先级和类型调度消息

#### 2.4.2 TTS 引擎 (TTSEngine)

**设计思路**：将文本转换为语音，提供语音反馈。

**实现细节**：
- 语音合成：将文本转换为语音
- 播放控制：控制语音播放

### 2.5 模拟模块

#### 2.5.1 摄像头模拟器 (CameraSimulator)

**设计思路**：模拟摄像头输入，方便系统测试。

**实现细节**：
- 帧生成：生成模拟的摄像头帧
- 多摄像头支持：支持多个摄像头的模拟

#### 2.5.2 调试查看器 (DebugViewer)

**设计思路**：可视化系统状态，方便调试和监控。

**实现细节**：
- 帧显示：显示处理后的帧
- 目标标注：标注检测到的目标
- 危险等级显示：显示当前危险等级

## 3. 系统组件划分

| 模块类型 | 组件名称 | 主要职责 | 文件路径 |
|---------|---------|---------|---------|
| 核心模块 | 资源管理器 | 管理系统资源，监控模块心跳 | core/resource_manager.py |
| 核心模块 | 实时安全调度器 | 评估危险等级，生成安全告警 | core/realtime_scheduler.py |
| 核心模块 | 风险评价器 | 基于AHP和模糊综合评价的危险评估 | core/risk_evaluator.py |
| 核心模块 | 复杂场景调度器 | 处理复杂场景，生成导航建议 | core/complex_scene_scheduler.py |
| 感知模块 | YOLO 目标检测 | 检测环境中的目标 | perception/yolo/yolo_detector.py |
| 感知模块 | VDA 深度估计 | 估计目标距离 | perception/vda/vda_depth.py |
| 感知模块 | ASR 语音识别 | 识别用户语音指令 | perception/asr/funasr_asr.py |
| 感知模块 | LLM 多模态理解 | 理解复杂场景 | perception/llm/qwen_multimodal.py |
| 融合模块 | 帧同步 | 同步多源数据 | fusion/frame_sync.py |
| 融合模块 | 深度融合 | 计算目标距离 | fusion/depth_fusion.py |
| 融合模块 | 目标跟踪 | 跟踪目标运动 | fusion/target_tracker.py |
| 融合模块 | 元数据封装 | 封装环境信息 | fusion/metadata_wrapper.py |
| 执行模块 | 广播调度器 | 管理语音播报队列 | execution/broadcast_scheduler.py |
| 执行模块 | TTS 引擎 | 文本转语音 | execution/tts_engine.py |
| 模拟模块 | 摄像头模拟器 | 模拟摄像头输入 | simulation/camera_simulator.py |
| 模拟模块 | 调试查看器 | 可视化系统状态 | simulation/debug_viewer.py |

## 4. 接口定义

### 4.1 核心模块接口

#### 4.1.1 ResourceManager
- `request_resources(resource_type: str) -> bool`：申请资源
- `release_resources(resource_type: str)`：释放资源
- `update_heartbeat(module_name: str)`：更新模块心跳
- `get_resource_status() -> Dict[str, Any]`：获取资源状态

#### 4.1.2 RealtimeScheduler
- `process_metadata(metadata: Dict[str, Any])`：处理环境元数据
- `get_alert() -> Optional[Dict[str, Any]]`：获取告警信息
- `is_complex_scene_triggered() -> bool`：检查是否触发复杂场景
- `reset_complex_scene_trigger()`：重置复杂场景触发信号

#### 4.1.3 RiskEvaluator
- `evaluate_risk(metadata: Dict[str, Any]) -> Dict[str, Any]`：评估危险等级
- `evaluate_special_scene(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]`：评估特殊场景

#### 4.1.4 ComplexSceneScheduler
- `process_complex_scene(image: Image.Image, metadata: Dict[str, Any], prompt: str) -> Optional[str]`：处理复杂场景
- `handle_wake_word(wake_word: str, image: Image.Image, metadata: Dict[str, Any]) -> Optional[str]`：处理唤醒词

### 4.2 感知模块接口

#### 4.2.1 YoloDetector
- `inference(frame: np.ndarray) -> List[Dict[str, Any]]`：执行目标检测
- `release()`：释放模型资源

#### 4.2.2 VDADepthEstimator
- `inference(frame: np.ndarray) -> np.ndarray`：执行深度估计
- `release()`：释放模型资源

#### 4.2.3 FunASRRecognizer
- `inference(audio_data: np.ndarray) -> Tuple[bool, str]`：执行语音识别
- `release()`：释放模型资源

#### 4.2.4 QwenMultimodal
- `inference(input_data: Tuple[Image.Image, Dict[str, Any], str]) -> str`：执行多模态理解
- `release()`：释放模型资源

### 4.3 融合模块接口

#### 4.3.1 FrameSync
- `add_frame(frame: np.ndarray, timestamp: float, camera_id: int)`：添加帧
- `add_yolo_result(yolo_results: List[Dict[str, Any]], timestamp: float)`：添加 YOLO 结果
- `add_vda_result(depth_map: np.ndarray, timestamp: float)`：添加 VDA 结果
- `get_sync_data() -> Optional[Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray, float, int]]`：获取同步数据

#### 4.3.2 DepthFusion
- `calculate_target_distances(yolo_results: List[Dict[str, Any]], depth_map: np.ndarray) -> List[Dict[str, Any]]`：计算目标距离

#### 4.3.3 TargetTracker
- `track_targets(targets: List[Dict[str, Any]], timestamp: float) -> List[Dict[str, Any]]`：跟踪目标

#### 4.3.4 MetadataWrapper
- `wrap_metadata(frame: np.ndarray, targets: List[Dict[str, Any]], timestamp: float, camera_id: int) -> Dict[str, Any]`：封装元数据

### 4.4 执行模块接口

#### 4.4.1 BroadcastScheduler
- `add_message(message: str, priority: int, alert_type: str)`：添加播报消息
- `start()`：启动调度器
- `stop()`：停止调度器

#### 4.4.2 TTSEngine
- `speak(text: str)`：播放语音
- `release()`：释放资源

### 4.5 模拟模块接口

#### 4.5.1 CameraSimulator
- `start()`：启动模拟器
- `stop()`：停止模拟器
- `get_all_frames() -> Dict[str, Tuple[np.ndarray, float]]`：获取所有摄像头帧

#### 4.5.2 DebugViewer
- `start()`：启动查看器
- `stop()`：停止查看器
- `update_frame(frame: np.ndarray, targets: List[Dict[str, Any]], depth_map: np.ndarray, risk_level: str)`：更新画面
- `show()`：显示画面

## 5. 关键算法说明

### 5.1 危险评估算法

**算法流程**：
1. AHP 权重计算：使用层次分析法计算距离、速度、类别和场景复杂度的权重
2. 模糊综合评价：使用梯形隶属度函数对各因素进行模糊评价
3. 综合风险评估：结合AHP权重计算综合风险得分
4. 特殊场景处理：针对人行横道、交叉路口、施工区域等特殊场景进行专门评估
5. 告警级别确定：根据风险得分确定告警级别
6. 告警触发：考虑连续告警帧数和冷却时间

**关键参数**：
- AHP 判断矩阵：基于专家知识的因素重要性判断
- 模糊评价参数：各因素的隶属度函数参数
- 特殊场景权重调整：不同特殊场景的风险调整系数
- 危险阈值：不同告警级别的阈值
- 连续告警帧数：触发告警的最小连续帧数
- 告警冷却时间：避免重复告警的时间间隔

### 5.2 深度融合算法

**算法流程**：
1. 提取目标 ROI 区域的深度值
2. 剔除边缘 10% 像素，减少边缘效应
3. 过滤异常值（0.1-20.0 米范围外的值）
4. 取中值作为目标距离，提高鲁棒性

**关键参数**：
- 边缘剔除比例：10%
- 深度值范围：0.1-20.0 米
- 距离计算方法：中值滤波

### 5.3 语音识别与唤醒词检测

**算法流程**：
1. 加载 FunASR 模型（包含 VAD 和标点模型）
2. 执行语音识别，将音频转换为文本
3. 检测文本中是否包含唤醒词

**关键参数**：
- VAD 静音时间：0.8 秒
- 唤醒词列表：["你好", "导盲", "导航", "小明", "小明同学"]

### 5.4 复杂场景处理算法

**算法流程**：
1. 申请 LLM 资源
2. 加载 LLM 模型
3. 生成场景描述 prompt
4. 执行 LLM 推理
5. 释放 LLM 资源

**关键参数**：
- 场景复杂度阈值：60
- LLM 模型：Qwen 多模态模型

## 6. 潜在优化点分析

### 6.1 性能优化

1. **模型优化**：
   - 采用模型量化技术，减少模型大小和推理时间
   - 使用模型蒸馏，提高模型推理速度
   - 考虑使用轻量级模型，平衡精度和速度

2. **并行处理**：
   - 采用多线程或多进程处理，提高并发性能
   - 优化数据传输，减少模块间数据拷贝
   - 使用 GPU 加速，提高模型推理速度

3. **资源管理**：
   - 实现动态资源分配，根据系统负载调整资源使用
   - 优化内存管理，减少内存泄漏和碎片
   - 实现资源预加载，减少模型加载时间

### 6.2 功能优化

1. **感知能力**：
   - 增加更多传感器输入，如惯性传感器、GPS 等
   - 优化目标检测和深度估计算法，提高准确性
   - 增加环境语义理解能力，如场景分类

2. **决策能力**：
   - 优化危险评估算法，减少误报和漏报
   - 增加自适应阈值，根据环境动态调整
   - 实现多场景学习，提高决策准确性

3. **交互能力**：
   - 增加更多唤醒词和语音指令
   - 优化语音合成质量，提高自然度
   - 增加触觉反馈，提供多模态交互

### 6.3 可靠性优化

1. **容错机制**：
   - 实现模块级故障检测和恢复
   - 增加传感器冗余，提高系统可靠性
   - 实现降级策略，在资源不足时保证核心功能

2. **鲁棒性**：
   - 增加异常处理，提高系统稳定性
   - 优化算法，提高在复杂环境下的鲁棒性
   - 增加系统自检功能，及时发现问题

3. **安全性**：
   - 实现数据加密，保护用户隐私
   - 增加安全审计，记录系统操作
   - 优化权限管理，防止未授权访问

## 7. 系统部署与维护

### 7.1 部署方案

1. **硬件要求**：
   - CPU：至少 4 核
   - 内存：至少 8GB
   - GPU：推荐 NVIDIA GPU（可选，用于加速模型推理）
   - 摄像头：至少 1 个 RGB 摄像头
   - 麦克风：用于语音交互

2. **软件依赖**：
   - Python 3.8+
   - PyTorch 1.10+
   - OpenCV
   - FunASR
   - psutil
   - PIL

3. **部署步骤**：
   - 安装依赖：`pip install -r requirements.txt`
   - 配置模型路径和参数
   - 启动系统：`python main.py`

### 7.2 维护策略

1. **日志管理**：
   - 定期清理日志文件，防止磁盘空间不足
   - 实现日志分级，便于问题定位

2. **模型更新**：
   - 定期更新模型，提高性能和准确性
   - 实现模型版本管理，便于回滚

3. **系统监控**：
   - 实现远程监控，及时发现问题
   - 定期生成系统报告，评估系统性能

4. **故障排查**：
   - 建立故障排查流程，快速定位问题
   - 提供故障自愈机制，减少人工干预

## 8. 总结与展望

### 8.1 系统特点

- **模块化设计**：清晰的模块划分，便于维护和扩展
- **多模态融合**：整合视觉、深度、语音等多源数据
- **实时响应**：优先处理安全相关任务，确保用户安全
- **智能决策**：结合规则和 LLM，提供智能导航建议
- **科学的危险评估**：基于 AHP 层次分析法和模糊综合评价，提高危险识别准确性
- **特殊场景处理**：针对人行横道、交叉路口等特殊场景进行专门评估
- **可扩展性**：支持真实和模拟模型，便于测试和部署

### 8.2 未来展望

1. **技术发展**：
   - 引入更先进的 AI 模型，提高感知和决策能力
   - 探索边缘计算，减少延迟，提高实时性
   - 利用 5G 网络，实现远程协助和数据共享

2. **功能扩展**：
   - 增加室内导航功能，支持复杂建筑环境
   - 实现个性化服务，根据用户习惯调整系统行为
   - 增加社交功能，支持与他人的交互

3. **应用场景**：
   - 扩展到其他辅助领域，如老年人护理
   - 与智能城市系统集成，提供更全面的导航信息
   - 支持多语言和多地区的使用场景

### 8.3 结论

导盲系统采用先进的 AI 技术，构建了一个多模态融合的辅助导航系统。系统通过实时感知、智能决策和安全执行，为视障人士提供了可靠的环境感知和导航辅助。未来，随着技术的不断发展和功能的持续扩展，系统将在更多场景中发挥重要作用，为视障人士的生活带来更多便利。