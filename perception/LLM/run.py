from qwen35 import Qwen35VLLM
import os
import threading
import time

try:
    import psutil
except Exception:
    psutil = None

try:
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
    )

    _NVML_OK = True
except Exception:
    _NVML_OK = False


def _chat_with_default_image(
    model_name: str,
    user_text: str,
    image_path: str,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> str:
    """
    优先走多模态（ollama.chat + images），失败则抛异常由外层兜底降级到纯文本。
    不依赖 PIL；images 直接传本地路径即可。
    """
    import ollama

    prompt = (
        "你是一个导盲系统的多模态语音助手。请用中文回答，简洁直接。\n"
        "要求：只输出最终答案本身，不要输出'用户/助手'标签，不要复读。\n"
        f"用户问题：{(user_text or '').strip()}\n"
    )
    resp = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt, "images": [image_path]}],
        options={
            "temperature": float(temperature),
            "num_predict": int(min(int(max_tokens), 256)),
            "num_ctx": 256,
            "repeat_last_n": 128,
            "repeat_penalty": 1.15,
            "presence_penalty": 0.2,
            "stop": ["\n用户：", "\n助手：", "\n### 用户问题", "<|endoftext|>", "</think>", "<|im_end|>"],
        },
        think=False,
        keep_alive=0,
    )
    msg = resp.get("message") or {}
    content = (msg.get("content") or "").strip()
    return content or "（无输出）"


class _MemoryMonitor:
    def __init__(self, interval_s: float = 0.2):
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

        self.peak_rss_gb = 0.0
        self.peak_sys_used_gb = 0.0
        self.peak_gpu_used_gb = None  # Optional[float]

        self._proc = None
        self._gpu_handle = None

    def start(self) -> None:
        if psutil is not None:
            self._proc = psutil.Process()
        if _NVML_OK:
            try:
                nvmlInit()
                self._gpu_handle = nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._gpu_handle = None
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2.0)
        if _NVML_OK:
            try:
                nvmlShutdown()
            except Exception:
                pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            # Python 进程 RSS
            if self._proc is not None:
                try:
                    rss = float(self._proc.memory_info().rss) / 1024**3
                    if rss > self.peak_rss_gb:
                        self.peak_rss_gb = rss
                except Exception:
                    pass

            # 系统内存 used
            if psutil is not None:
                try:
                    vm = psutil.virtual_memory()
                    used = float(vm.used) / 1024**3
                    if used > self.peak_sys_used_gb:
                        self.peak_sys_used_gb = used
                except Exception:
                    pass

            # GPU used（注意：这是整卡 used，不是仅 Ollama / 本进程）
            if self._gpu_handle is not None:
                try:
                    info = nvmlDeviceGetMemoryInfo(self._gpu_handle)
                    used_gb = float(info.used) / 1024**3
                    if self.peak_gpu_used_gb is None or used_gb > float(self.peak_gpu_used_gb):
                        self.peak_gpu_used_gb = used_gb
                except Exception:
                    pass

            time.sleep(self.interval_s)


def main():
    llm = Qwen35VLLM(model_name="qwen3.5:2b-q4_K_M")
    default_image = "/home/nvidia/MHSEE/QiMing/perception/LLM/testimage.jpg"

    while True:
        user_input = input("\n请输入问题 (输入 'quit' 退出): ")
        if user_input.lower() == 'quit':
            break
            
        monitor = _MemoryMonitor(interval_s=0.2)
        monitor.start()
        t0 = time.time()
        response = None
        # 默认带图做多模态测试；若模型不支持视觉或接口失败，则降级为纯文本
        if os.path.isfile(default_image):
            try:
                response = _chat_with_default_image(
                    model_name=llm.model_name,
                    user_text=user_input,
                    image_path=default_image,
                    temperature=0.7,
                    max_tokens=256,
                )
            except Exception as e:
                print(f"[VLM] 多模态调用失败，降级为纯文本：{e}")

        if response is None:
            response = llm.generate(
                text=user_input,
                temperature=0.7,
                max_tokens=256,
            )
        monitor.stop()
        dt = time.time() - t0
        
        print(f"Qwen: {response}")
        if psutil is None:
            print("[Monitor] 未安装 psutil，无法统计进程 RSS/系统内存峰值。")
        else:
            extra = ""
            if monitor.peak_gpu_used_gb is not None:
                extra = f", GPU峰值used(整卡)={monitor.peak_gpu_used_gb:.2f}GB"
            print(
                f"[Monitor] 本次生成耗时={dt:.2f}s, "
                f"进程RSS峰值={monitor.peak_rss_gb:.2f}GB, "
                f"系统内存used峰值={monitor.peak_sys_used_gb:.2f}GB"
                f"{extra}"
            )

if __name__ == '__main__':
    main()
