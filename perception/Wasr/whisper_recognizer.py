"""
Whisper.cpp-backed recognizer: same public API as FunASRRecognizer (inference, release).
Wake-word logic mirrors MHSEE perception/asr/funasr_asr.py; recognition uses whisper-cli subprocess.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .whisper_runner import transcribe_numpy


class WhisperCppRecognizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.whisper_binary = config.get(
            "whisper_binary",
            "/root/Wasr/third_party/whisper.cpp/build/bin/whisper-cli",
        )
        self.whisper_model = config.get("whisper_model", "")
        self.language = config.get("language", "zh")
        self.threads = int(config.get("threads", 4))
        extra = config.get("extra_args")
        self.extra_args: List[str] = list(extra) if isinstance(extra, list) else []
        # whisper-cli --prompt：帮助模型偏向中文口语/领域词（与 -l zh 配合）
        self.prompt = (config.get("prompt") or "").strip()

        self.sample_rate = 16000
        self.wake_words = ["你好", "导盲", "导航", "小明", "小明同学", "小"]
        self.wake_state = False
        self.wake_audio_buffer: List[float] = []
        self.wake_silence_counter = 0
        self.wake_silence_threshold = 3

        self.last_peak_memory_kb: Optional[int] = None
        self.last_inference_ms: Optional[float] = None

        self.vad_model: Any = None
        self._validate_paths()
        self._load_funasr_vad()
        self.model = True  # loaded (same truthiness contract as FunASRRecognizer)

    def _validate_paths(self) -> None:
        if not self.whisper_model:
            raise RuntimeError(
                "Wasr: config 'whisper_model' is required (path to ggml/gguf whisper model)"
            )
        if not os.path.isfile(self.whisper_binary):
            raise RuntimeError(f"Wasr: whisper binary not found: {self.whisper_binary}")
        if not os.path.isfile(self.whisper_model):
            raise RuntimeError(f"Wasr: model file not found: {self.whisper_model}")
        size = os.path.getsize(self.whisper_model)
        if size < 256 * 1024:
            raise RuntimeError(
                f"Wasr: model file too small ({size} bytes); use a real ggml model, not the repo test stub."
            )

    def _load_funasr_vad(self) -> None:
        """与 perception/asr/funasr_asr.py 一致：FunASR FSMN VAD；失败则回退能量门限。"""
        if not self.config.get("vad_enabled", True):
            print("[Wasr] 已关闭 FunASR VAD（vad_enabled: false），使用能量门限")
            return
        try:
            from funasr import AutoModel

            vad_spec = self.config.get(
                "vad_model_path",
                "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            )
            device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
            self.vad_model = AutoModel(model=vad_spec, device=device)
            print(f"[Wasr] FunASR VAD 加载成功: {vad_spec}")
        except Exception as e:
            print(f"[Wasr] FunASR VAD 加载失败，使用能量门限: {e}")
            self.vad_model = None

    def _detect_voice_activity(self, audio_data: np.ndarray) -> bool:
        if self.vad_model is None:
            energy = np.sum(np.square(audio_data)) / max(len(audio_data), 1)
            return bool(energy > 0.001)
        try:
            result = self.vad_model.generate(input=audio_data)
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("value", 0) == 1
            return False
        except Exception as e:
            print(f"[Wasr] VAD 推理失败，使用能量检测: {e}")
            energy = np.sum(np.square(audio_data)) / max(len(audio_data), 1)
            return bool(energy > 0.001)

    def _recognize(self, audio_data: np.ndarray) -> str:
        import time

        t0 = time.perf_counter()
        merged_args: List[str] = list(self.extra_args)
        if self.prompt:
            merged_args.extend(["--prompt", self.prompt])
        text, peak_kb = transcribe_numpy(
            self.whisper_binary,
            self.whisper_model,
            audio_data,
            sample_rate=self.sample_rate,
            language=self.language,
            threads=self.threads,
            extra_args=merged_args,
        )
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        self.last_peak_memory_kb = peak_kb
        try:
            from utils.logger import logger

            if peak_kb is not None:
                logger.info(
                    "[Wasr] whisper peak VmHWM/RSS ~ %.2f MB (inference %.1f ms)",
                    peak_kb / 1024.0,
                    self.last_inference_ms,
                )
        except Exception:
            if peak_kb is not None:
                print(
                    f"[Wasr] whisper peak ~ {peak_kb / 1024.0:.2f} MB, "
                    f"inference {self.last_inference_ms:.1f} ms"
                )
        return (text or "").strip()

    def inference(
        self, audio_data: np.ndarray, is_final: bool = False
    ) -> Tuple[bool, str, Optional[bool]]:
        try:
            from utils.config_loader import config_loader

            cfg = config_loader.get_config()
            verbose = cfg.get("system", {}).get("verbose", True)
        except Exception:
            verbose = True

        if self.model is None:
            raise RuntimeError("ASR模型未加载")

        asr_text = ""
        wake_detected = False
        is_speech: Optional[bool] = None

        try:
            if len(audio_data) > self.sample_rate * 0.3:
                if verbose:
                    print(f"[ASR] 处理音频，长度: {len(audio_data)} 样本")

                is_speech = self._detect_voice_activity(audio_data)
                if verbose:
                    print(f"[ASR] 语音活动检测: {is_speech}")

                try:
                    asr_text = self._recognize(np.asarray(audio_data))
                    if verbose:
                        print(f"[ASR] 原始识别结果: '{asr_text}'")
                except Exception as e:
                    if verbose:
                        print(f"[ASR] 模型识别失败: {e}")
                        import traceback

                        traceback.print_exc()

        except Exception as e:
            if verbose:
                print(f"[ASR] 推理失败: {e}")
                import traceback

                traceback.print_exc()
            asr_text = ""
            is_speech = None

        if self.wake_state:
            if verbose:
                print("[ASR] 处于唤醒状态")
            self.wake_audio_buffer.extend(audio_data.tolist())

            is_speech = self._detect_voice_activity(audio_data)
            if not is_speech:
                self.wake_silence_counter += 1
                if verbose:
                    print(f"[ASR] 唤醒后静默计数: {self.wake_silence_counter}")
            else:
                self.wake_silence_counter = 0

            if self.wake_silence_counter >= self.wake_silence_threshold or is_final:
                if verbose:
                    print("[ASR] 唤醒后句子结束")
                if len(self.wake_audio_buffer) > self.sample_rate * 0.5:
                    wake_audio = np.array(self.wake_audio_buffer)
                    try:
                        wake_text = self._recognize(wake_audio)
                        if verbose:
                            print(f"[ASR] 唤醒句子识别结果: '{wake_text}'")
                        asr_text = wake_text
                    except Exception as e:
                        if verbose:
                            print(f"[ASR] 唤醒句子识别失败: {e}")

                self.wake_state = False
                self.wake_audio_buffer = []
                self.wake_silence_counter = 0
                wake_detected = True
                if verbose:
                    print("[ASR] 唤醒处理完成，返回wake_detected=True")
        elif is_final and asr_text:
            if verbose:
                print("[ASR] 处理最终片段，检测唤醒词")
            clean_text = re.sub(r"[。，、；：？！,.?!;:\s]", "", asr_text)
            if verbose:
                print(f"[ASR] 去除标点后的文本: '{clean_text}'")

            wake_detected = any(
                word in asr_text or word in clean_text for word in self.wake_words
            )
            if wake_detected:
                if verbose:
                    print("[ASR] 检测到唤醒词")
                    for word in self.wake_words:
                        if word in asr_text or word in clean_text:
                            print(f"[ASR]   - 唤醒词 '{word}' 被检测到")
                            print("[ASR] 检测到唤醒词，立即返回wake_detected=True")
                return True, asr_text
        else:
            if asr_text:
                clean_text = re.sub(r"[。，、；：？！,.?!;:\s]", "", asr_text)
                if verbose:
                    print(f"[ASR] 去除标点后的文本: '{clean_text}'")

                wake_detected = any(
                    word in asr_text or word in clean_text for word in self.wake_words
                )
                if wake_detected:
                    if verbose:
                        print("[ASR] 检测到唤醒词")
                        for word in self.wake_words:
                            if word in asr_text or word in clean_text:
                                print(f"[ASR]   - 唤醒词 '{word}' 被检测到")
                                print("[ASR] 检测到唤醒词，立即返回wake_detected=True")
                    return True, asr_text

        return wake_detected, asr_text, is_speech

    def release(self) -> None:
        self.model = None
        if self.vad_model:
            try:
                if hasattr(self.vad_model, "cleanup"):
                    self.vad_model.cleanup()
                self.vad_model = None
                print("[Wasr] FunASR VAD 资源已释放")
            except Exception as e:
                print(f"[Wasr] FunASR VAD 释放失败: {e}")
                self.vad_model = None
        print("[Wasr] whisper 资源已释放")
