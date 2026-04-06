"""
LLM 推理与视觉 CUDA 峰值互斥：Ollama 加载/推理时跳过本轮 YOLO+VDA，减轻 Jetson 统一内存 OOM。
"""
import threading

_busy = threading.Event()


def set_llm_busy(busy: bool) -> None:
    if busy:
        _busy.set()
    else:
        _busy.clear()


def is_llm_busy() -> bool:
    return _busy.is_set()
