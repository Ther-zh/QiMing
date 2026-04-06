import base64
import gc
import io
import os
import re
import time
from typing import Optional, Union, List
import numpy as np
from PIL import Image
import ollama

try:
    from utils.logger import logger as _llm_logger
except ImportError:
    _llm_logger = None

# 送入 Ollama VLM 的长边上限，减轻 Jetson 统一内存上视觉编码峰值（需 >32 以满足 Qwen3-VL 预处理）
# 全流程 ASR+YOLO+VDA 并行时可用环境变量 QIMING_VL_MAX_SIDE 再压低（如 256）
_MAX_VL_IMAGE_SIDE = int(os.environ.get("QIMING_VL_MAX_SIDE", "288"))

# 日志与自动化 grep：真模型输出不应包含该前缀；兜底时 stderr/日志会出现 LLM_META
DEFAULT_FALLBACK_PHRASE = "前方道路安全，可以正常通行"


def _log_llm_meta(event: str, **kwargs) -> None:
    """机器可读一行，便于区分真输出与兜底（勿写入对用户播报的字符串）。"""
    parts = [f"event={event!r}"] + [f"{k}={v!r}" for k, v in kwargs.items()]
    print("[LLM_META] " + " ".join(parts), flush=True)
    if _llm_logger:
        _llm_logger.info("[LLM_META] " + " ".join(parts))


def _safe_strip_content(val) -> str:
    """Ollama 可能返回 content: null，dict.get('content','') 会得到 None，不能直接 .strip()。"""
    if val is None:
        return ""
    return str(val).strip()

class Qwen35Ollama:
    def __init__(
        self,
        model_name: str = "qwen3.5-4b",
        **kwargs
    ):
        """
        初始化 Qwen3.5 Ollama 推理引擎。
        
        Args:
            model_name: Ollama模型名称
            think: 是否启用 Ollama「思考」模式。Qwen3.5 系列默认应传 False：
                think 只能作为 chat/generate 的顶层参数；放在 options 里会被忽略，
                思考会占满 num_predict 导致 content/response 为空（见 ollama#14793）。
            ollama_options: 合并进 Ollama options 的键值，如 num_ctx、num_predict、temperature
            fallback_phrase: 空响应或异常时的兜底话术（与真输出区分靠 [LLM_META] 日志）
            **kwargs: 其他参数（兼容旧接口）
        """
        self.model_name = model_name
        # 显式 False：关闭 Qwen3.5 thinking；None 表示不向 API 传 think（由模型默认）
        self.think: Optional[bool] = kwargs.pop("think", False)
        self.fallback_phrase: str = kwargs.pop("fallback_phrase", DEFAULT_FALLBACK_PHRASE)
        ollama_options = kwargs.pop("ollama_options", None) or {}
        self.default_params = {
            "temperature": 0.6,
            "top_p": 0.8,
            "top_k": 20,
            # 默认偏小省内存；长 prompt 可在 config models.llm.ollama_options 提高 num_ctx
            "num_ctx": 256,
            "num_predict": 256,
        }
        for k, v in ollama_options.items():
            if k in self.default_params or k in ("temperature", "top_p", "top_k", "repeat_penalty"):
                self.default_params[k] = v
        if kwargs:
            print(f"[System] Ollama 忽略未识别构造参数: {list(kwargs.keys())}", flush=True)
        print(f"[System] Ollama模型已就绪: {model_name}")

    @staticmethod
    def _is_ollama_memory_shortage(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return "system memory" in msg or "requires more" in msg

    def _release_pressure_before_ollama_retry(self, attempt: int = 0) -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        # 给 FunASR 等线程留出归还页的时间；随重试递增
        time.sleep(0.45 + min(attempt, 4) * 0.2)

    def _clean_response(self, text: str) -> str:
        """
        清理响应文本，移除思考标签等内容
        """
        if not text:
            return text
        
        text = re.sub(r'<think[^>]*>.*?</think\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<think[^>]*>.*', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Thinking Process:.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL | re.IGNORECASE)
        # 勿用宽泛「**分析**」规则，易误删中文导航正文
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = text.strip()
        
        return text

    def _downscale_for_vlm(self, pil: Image.Image) -> Image.Image:
        w, h = pil.size
        m = max(w, h)
        if m <= _MAX_VL_IMAGE_SIDE:
            return pil
        scale = _MAX_VL_IMAGE_SIDE / float(m)
        nw, nh = max(32, int(w * scale)), max(32, int(h * scale))
        return pil.resize((nw, nh), getattr(Image, "Resampling", Image).LANCZOS)

    def _prepare_image_for_ollama(self, image: Union[str, Image.Image, np.ndarray]) -> Union[str, bytes]:
        """供 ollama.chat 的 messages[].images 使用：本地路径、或 PNG 的 base64 字符串。"""
        if isinstance(image, str):
            if os.path.isfile(image):
                return os.path.abspath(image)
            raise FileNotFoundError(f"图片路径不存在: {image}")
        if isinstance(image, Image.Image):
            im = image.convert("RGB") if image.mode not in ("RGB", "L") else image
            im = self._downscale_for_vlm(im)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        if isinstance(image, np.ndarray):
            import cv2
            if image.ndim == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            pil = self._downscale_for_vlm(pil)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        raise TypeError(f"不支持的 image 类型: {type(image)}")

    def generate(
        self,
        text: str,
        image: Optional[Union[str, Image.Image]] = None,
        sampling_params=None,
        **kwargs
    ) -> str:
        """
        生成回复。
        
        Args:
            text: 输入的文本提示词
            image: 本地图片路径或 PIL.Image；非空时使用 ollama.chat 多模态（需 Ollama 侧为视觉模型）
            sampling_params: 采样参数 (兼容旧接口，暂不使用)
            **kwargs: 临时覆盖参数 (如 temperature, max_tokens 等)
        
        Returns:
            生成的文本字符串
        """
        text_fallback_model = kwargs.pop("text_fallback_model", None)

        params = self.default_params.copy()
        if kwargs:
            if "temperature" in kwargs:
                params["temperature"] = kwargs["temperature"]
            if "max_tokens" in kwargs:
                params["num_predict"] = kwargs["max_tokens"]
            for opt_key in ("num_ctx", "num_predict", "top_p", "top_k", "temperature"):
                if opt_key in kwargs and kwargs[opt_key] is not None:
                    params[opt_key] = kwargs[opt_key]
        
        try:
            import sys
            print(f"[LLM] 执行生成...")
            sys.stdout.flush()

            last_err: Optional[BaseException] = None
            _mem_attempts = 6
            for attempt in range(_mem_attempts):
                try:
                    if image is not None:
                        img_payload = self._prepare_image_for_ollama(image)
                        messages = [{"role": "user", "content": text, "images": [img_payload]}]
                        response = ollama.chat(
                            model=self.model_name,
                            messages=messages,
                            options=params,
                            keep_alive=0,
                            **({"think": self.think} if self.think is not None else {}),
                        )
                        msg = response.get("message")
                        if msg is None:
                            result = ""
                        elif isinstance(msg, dict):
                            result = _safe_strip_content(msg.get("content"))
                        else:
                            result = _safe_strip_content(getattr(msg, "content", None))
                    else:
                        response = ollama.generate(
                            model=self.model_name,
                            prompt=text,
                            options=params,
                            keep_alive=0,
                            **({"think": self.think} if self.think is not None else {}),
                        )
                        result = _safe_strip_content(response.get("response"))
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if not self._is_ollama_memory_shortage(e) or attempt >= _mem_attempts - 1:
                        raise
                    print(
                        f"[LLM] Ollama 报告内存紧张，重试 {attempt + 1}/{_mem_attempts} …"
                    )
                    sys.stdout.flush()
                    self._release_pressure_before_ollama_retry(attempt)
            if last_err is not None:
                raise last_err

            raw_len = len(result or "")
            raw_preview = (result or "")[:160]
            if _llm_logger:
                _llm_logger.info(
                    f"[LLM/Ollama] 完成 model={self.model_name} multimodal={image is not None} "
                    f"raw_len={raw_len} think={self.think!r} num_ctx={params.get('num_ctx')} "
                    f"num_predict={params.get('num_predict')} preview={raw_preview!r}"
                )
            print("[LLM] 生成完成")
            sys.stdout.flush()

            cleaned = self._clean_response(result)
            if cleaned != result and _llm_logger:
                _llm_logger.info(
                    f"[LLM/Ollama] 清洗后 len={len(cleaned)}（原 len={raw_len}）"
                )

            if cleaned:
                _log_llm_meta(
                    "ok",
                    model=self.model_name,
                    multimodal=image is not None,
                    reply_len=len(cleaned),
                    raw_len=raw_len,
                )
                print(f"[LLM] 生成结果: {cleaned[:100]}...")
                sys.stdout.flush()
                return cleaned
            print("[LLM] 生成结果为空（API 无正文或清洗后为空），使用兜底")
            sys.stdout.flush()
            _log_llm_meta(
                "fallback_empty",
                model=self.model_name,
                multimodal=image is not None,
                raw_len=raw_len,
                num_ctx=params.get("num_ctx"),
                hint="check_think_false_num_ctx_model_vision",
            )
            if _llm_logger:
                _llm_logger.warning(
                    "[LLM/Ollama] 空回复兜底：请检查 think=False、num_ctx、或 Ollama 日志"
                )
            return self.fallback_phrase
        except Exception as e:
            print(f"[LLM] 生成失败: {e}")
            # Jetson 统一内存下 VLM 常固定需 ~2.7GiB；不足时改用已配置的纯文本小模型，避免整句静态兜底
            if (
                image is not None
                and text_fallback_model
                and text_fallback_model != self.model_name
                and self._is_ollama_memory_shortage(e)
            ):
                _log_llm_meta(
                    "degrade_text_only",
                    from_model=self.model_name,
                    to_model=text_fallback_model,
                    error=str(e)[:160],
                )
                print(
                    f"[LLM] VLM 内存不足，降级为纯文本模型: {text_fallback_model}",
                    flush=True,
                )
                mtok = min(160, int(kwargs.get("max_tokens", 256) or 256))
                backup = Qwen35Ollama(
                    model_name=text_fallback_model,
                    think=self.think,
                    ollama_options={"num_ctx": 256, "num_predict": 96},
                    fallback_phrase=self.fallback_phrase,
                )
                aug = (
                    text
                    + "\n\n（当前无图像，请仅根据上文用户话语与环境文字摘要，用一句话给出中文导盲建议。）"
                )
                return backup.generate(
                    text=aug,
                    image=None,
                    max_tokens=mtok,
                    text_fallback_model=None,
                )

            _log_llm_meta(
                "fallback_exception",
                model=self.model_name,
                multimodal=image is not None,
                error=str(e)[:200],
            )
            if image is not None:
                print(
                    "[LLM] 多模态失败常见原因：Ollama 中该 model_name 不是视觉模型，"
                    "请 ollama pull 支持 images 的变体并调整 config 中的 model_name。"
                )
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            return self.fallback_phrase

    def chat(
        self,
        messages: List[dict],
        **kwargs
    ) -> str:
        """
        对话模式生成。
        
        Args:
            messages: 消息列表
            **kwargs: 额外参数
        
        Returns:
            生成的文本字符串
        """
        params = self.default_params.copy()
        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            params["num_predict"] = kwargs["max_tokens"]
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options=params,
                keep_alive=0,
                **({"think": self.think} if self.think is not None else {}),
            )
            msg = response.get("message")
            if msg is None:
                out = ""
            elif isinstance(msg, dict):
                out = _safe_strip_content(msg.get("content"))
            else:
                out = _safe_strip_content(getattr(msg, "content", None))
            cleaned = self._clean_response(out)
            if cleaned:
                _log_llm_meta("ok", model=self.model_name, route="chat", reply_len=len(cleaned))
                return cleaned
            _log_llm_meta("fallback_empty", model=self.model_name, route="chat", raw_len=len(out or ""))
            return self.fallback_phrase
        except Exception as e:
            print(f"[LLM] Chat失败: {e}")
            _log_llm_meta("fallback_exception", model=self.model_name, route="chat", error=str(e)[:200])
            return self.fallback_phrase

    def batch_generate(
        self,
        inputs: List[dict],
        sampling_params=None,
    ) -> List[str]:
        """
        批量生成。
        
        Args:
            inputs: 字典列表，每个字典包含 "text"
            sampling_params: 采样参数
        
        Returns:
            生成结果列表
        """
        results = []
        for item in inputs:
            text = item.get("text", "")
            result = self.generate(text)
            results.append(result)
        return results


class Qwen35VLLM(Qwen35Ollama):
    """
    兼容旧接口的别名类
    """
    pass


if __name__ == "__main__":
    model = Qwen35Ollama(model_name="qwen3.5-4b")
    
    print("\n--- 测试纯文本生成 ---")
    response = model.generate("你好，请用一句话介绍你自己。")
    print("回复:", response)
