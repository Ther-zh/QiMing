#!/usr/bin/env python3
"""
Wasr 单模块测试（不启动完整 MHSEE）。

用法（conda vda）：
  cd /root && PYTHONPATH=/root python Wasr/tests/test_wasr_modules.py
  PYTHONPATH=/root python Wasr/tests/test_wasr_modules.py --skip-whisper   # 跳过 whisper-cli 推理

说明：JFK 样例为英文语音，对应测试固定 language=en，输出英文是预期行为。
     中文效果请看 TestChineseModelsCompare（从 MHSEE/video/video.mp4 抽音频 + language=zh）。

依赖：numpy；whisper 相关测试需要已编译 whisper-cli 与 ggml 模型；中文对比需 ffmpeg。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import sys
import unittest
import wave
from pathlib import Path

import numpy as np

# 保证可导入 Wasr.*
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Wasr.memory_probe import read_proc_status_kb, run_monitor_thread  # noqa: E402
from Wasr.whisper_runner import (  # noqa: E402
    transcribe_file,
    transcribe_numpy,
    write_wav_mono16,
)
from Wasr.whisper_recognizer import WhisperCppRecognizer  # noqa: E402

WHISPER_CPP_ROOT = _ROOT / "Wasr" / "third_party" / "whisper.cpp"
DEFAULT_BINARY = WHISPER_CPP_ROOT / "build" / "bin" / "whisper-cli"
# 优先使用 small-q5_1（与 MHSEE 默认一致）；无则回退 tiny
_MODEL_SMALL = _ROOT / "Wasr" / "models" / "ggml-small-q5_1.bin"
_MODEL_TINY = _ROOT / "Wasr" / "models" / "ggml-tiny-q8_0.bin"
DEFAULT_MODEL = _MODEL_SMALL if _MODEL_SMALL.is_file() else _MODEL_TINY
JFK_WAV = WHISPER_CPP_ROOT / "samples" / "jfk.wav"
#
# 中文样例视频路径（不同仓库布局兼容）：
# - 老路径：<root>/MHSEE/video/video.mp4
# - 本项目：/home/nvidia/MHSEE/QiMing/video/video.mp4 （即 <root>/video/video.mp4）
#
_VIDEO_ZH_CANDIDATES = [
    _ROOT.parent / "video" / "video.mp4",
    _ROOT / "video" / "video.mp4",
    _ROOT / "MHSEE" / "video" / "video.mp4",
]
VIDEO_ZH = next((p for p in _VIDEO_ZH_CANDIDATES if p.is_file()), _VIDEO_ZH_CANDIDATES[0])


def _cjk_char_count(s: str) -> int:
    return sum(1 for c in s if "\u4e00" <= c <= "\u9fff")


def _extract_video_audio_wav(video: Path, seconds: float = 8.0) -> Path:
    fd, out = tempfile.mkstemp(suffix=".wav", prefix="wasr_zh_")
    os.close(fd)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-t",
            str(seconds),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "wav",
            out,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return Path(out)


def _load_wav_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        width = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    if width != 2:
        raise unittest.SkipTest("test expects 16-bit WAV")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    return x, sr


class TestMemoryProbe(unittest.TestCase):
    def test_read_proc_self(self) -> None:
        hwm, rss = read_proc_status_kb(os.getpid())
        self.assertGreater(rss, 0, "VmRSS 应可读")

    def test_monitor_sleep_child_peak(self) -> None:
        proc = subprocess.Popen(
            ["/bin/sleep", "0.2"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        stop_ev, th, peak_holder = run_monitor_thread(proc.pid)
        proc.wait()
        stop_ev.set()
        th.join(timeout=2.0)
        peak = int(peak_holder[0]) if peak_holder else 0
        self.assertGreater(peak, 0, "子进程峰值应 >0")


class TestWhisperRunnerWav(unittest.TestCase):
    def test_write_wav_roundtrip_size(self) -> None:
        import tempfile

        sr = 16000
        t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
        audio = (0.01 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            write_wav_mono16(path, audio, sr)
            self.assertGreater(os.path.getsize(path), 1000)
        finally:
            os.remove(path)


@unittest.skipUnless(DEFAULT_BINARY.is_file(), f"缺少 whisper-cli: {DEFAULT_BINARY}")
@unittest.skipUnless(DEFAULT_MODEL.is_file(), f"缺少模型: {DEFAULT_MODEL}")
class TestWhisperCliTranscribe(unittest.TestCase):
    def test_transcribe_file_jfk_en(self) -> None:
        self.assertTrue(JFK_WAV.is_file(), f"缺少样例: {JFK_WAV}")
        text, peak_kb = transcribe_file(
            str(DEFAULT_BINARY),
            str(DEFAULT_MODEL),
            str(JFK_WAV),
            language="en",
            threads=4,
        )
        self.assertIn("country", text.lower())
        self.assertIsNotNone(peak_kb)
        self.assertGreater(peak_kb or 0, 100)

    def test_transcribe_numpy_jfk(self) -> None:
        wav, sr = _load_wav_mono_float32(JFK_WAV)
        self.assertEqual(sr, 16000)
        text, peak_kb = transcribe_numpy(
            str(DEFAULT_BINARY),
            str(DEFAULT_MODEL),
            np.asarray(wav, dtype=np.float32),
            sample_rate=16000,
            language="en",
            threads=4,
        )
        self.assertTrue(len(text) > 10)


@unittest.skipUnless(DEFAULT_BINARY.is_file(), "no whisper-cli")
@unittest.skipUnless(DEFAULT_MODEL.is_file(), "no ggml model")
class TestWhisperCppRecognizer(unittest.TestCase):
    def test_init_and_inference_jfk(self) -> None:
        cfg = {
            "whisper_binary": str(DEFAULT_BINARY),
            "whisper_model": str(DEFAULT_MODEL),
            "language": "en",
            "threads": 4,
            "vad_enabled": False,
        }
        rec = WhisperCppRecognizer(cfg)
        wav, sr = _load_wav_mono_float32(JFK_WAV)
        self.assertEqual(sr, 16000)
        wake, text, is_speech = rec.inference(np.asarray(wav, dtype=np.float32), False)
        self.assertIsInstance(wake, bool)
        self.assertTrue(len(text) > 5)
        self.assertIsNotNone(is_speech)
        self.assertIsNotNone(rec.last_peak_memory_kb)
        rec.release()


@unittest.skipUnless(DEFAULT_BINARY.is_file(), f"缺少 whisper-cli: {DEFAULT_BINARY}")
@unittest.skipUnless(_MODEL_TINY.is_file() and _MODEL_SMALL.is_file(), "需要同时存在 tiny-q8_0 与 small-q5_1")
@unittest.skipUnless(VIDEO_ZH.is_file(), f"缺少中文测试视频: {VIDEO_ZH}")
class TestChineseModelsCompare(unittest.TestCase):
    """同一中文片段上对比两个模型；必须 language=zh，否则易偏英文。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls._wav_path = _extract_video_audio_wav(VIDEO_ZH, seconds=8.0)

    @classmethod
    def tearDownClass(cls) -> None:
        p = getattr(cls, "_wav_path", None)
        if p and p.is_file():
            try:
                os.remove(p)
            except OSError:
                pass

    def test_tiny_vs_small_same_zh_clip(self) -> None:
        results: dict[str, tuple[str, int]] = {}
        for label, model_path in (
            ("ggml-tiny-q8_0", _MODEL_TINY),
            ("ggml-small-q5_1", _MODEL_SMALL),
        ):
            text, peak_kb = transcribe_file(
                str(DEFAULT_BINARY),
                str(model_path),
                str(self._wav_path),
                language="zh",
                threads=4,
            )
            cjk = _cjk_char_count(text)
            results[label] = (text, cjk)
            # 醒目输出，便于肉眼对比
            print(
                f"\n========== {label} (language=zh) peak_kb={peak_kb} CJK字数={cjk} =========="
            )
            print(text)
            print("==========\n")

        _, tiny_cjk = results["ggml-tiny-q8_0"]
        _, small_cjk = results["ggml-small-q5_1"]

        self.assertGreater(
            tiny_cjk,
            0,
            "tiny 模型在此片段上应能识别出部分汉字；若全英文请确认 whisper 传入 -l zh",
        )
        self.assertGreater(
            small_cjk,
            0,
            "small 模型应能识别出汉字",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Wasr 单模块测试")
    parser.add_argument(
        "--skip-whisper",
        action="store_true",
        help="跳过依赖 whisper-cli / 模型的测试类",
    )
    parser.add_argument(
        "--skip-zh-compare",
        action="store_true",
        help="跳过从 video.mp4 抽音频的中英文模型对比（仅测 JFK 等）",
    )
    args = parser.parse_args()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryProbe))
    suite.addTests(loader.loadTestsFromTestCase(TestWhisperRunnerWav))
    if not args.skip_whisper:
        suite.addTests(loader.loadTestsFromTestCase(TestWhisperCliTranscribe))
        suite.addTests(loader.loadTestsFromTestCase(TestWhisperCppRecognizer))
        if not args.skip_zh_compare:
            suite.addTests(loader.loadTestsFromTestCase(TestChineseModelsCompare))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
