from typing import Dict, Any, Optional
import re
import shutil
import subprocess

from utils.logger import logger


def _strip_speakable(text: str) -> str:
    """去掉特殊 token，避免 TTS 读出乱码。"""
    if not text:
        return ""
    t = re.sub(r"<\|[^|]*\|>", " ", text)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


class TTSEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tts_type = config.get("tts_type", "real")
        self._espeak: Optional[str] = None
        self._spd_say: Optional[str] = None
        self._load_engine()

    def _load_engine(self) -> None:
        if self.tts_type != "real":
            logger.warning("tts_type 非 real，将仍尝试使用系统语音合成；请在 config 中设 execution.tts_type: real")
        self._espeak = shutil.which("espeak-ng") or shutil.which("espeak")
        self._spd_say = shutil.which("spd-say")
        if not self._espeak and not self._spd_say:
            raise RuntimeError(
                "未找到可用的系统 TTS（需要安装 espeak-ng 或 speech-dispatcher 的 spd-say）。"
                "Jetson/Ubuntu: sudo apt-get install -y espeak-ng"
            )
        name = "espeak-ng/espeak" if self._espeak else "spd-say"
        logger.info(f"真实 TTS 引擎: {name}")

    def speak(self, text: str) -> None:
        clean = _strip_speakable(text)
        if not clean:
            logger.warning("TTS: 空文本，跳过")
            return
        try:
            if self._espeak:
                # -v zh 使用中文（若系统无中文语音包会回退英文发音）
                subprocess.run(
                    [self._espeak, "-v", "zh", "-s", "160", clean],
                    check=False,
                    timeout=120,
                    capture_output=True,
                )
            else:
                subprocess.run(
                    [self._spd_say, "-l", "zh-cn", clean],
                    check=False,
                    timeout=120,
                    capture_output=True,
                )
        except subprocess.TimeoutExpired:
            logger.error("TTS 播放超时")
        except Exception as e:
            logger.error(f"TTS 播放失败: {e}")

    def release(self) -> None:
        self._espeak = None
        self._spd_say = None
        logger.info("TTS 引擎资源已释放")


class MockTTSEngine:
    """保留类名以兼容旧 import；行为与 TTSEngine 一致（真实系统 TTS）。"""

    def __init__(self, config: Dict[str, Any]):
        self._impl = TTSEngine({**config, "tts_type": "real"})

    def speak(self, text: str) -> None:
        self._impl.speak(text)

    def release(self) -> None:
        self._impl.release()
