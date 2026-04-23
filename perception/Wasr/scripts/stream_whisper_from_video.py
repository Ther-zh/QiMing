#!/usr/bin/env python3
"""
用 video.mp4 的音频流模拟“实时输入”，分片调用 Wasr.whisper_recognizer.WhisperCppRecognizer。

特点：
- 使用 ffmpeg 抽取音频并输出 raw s16le (16kHz, mono) 到 stdout
- 按 chunk 时长读取，sleep 以模拟实时
- 可选择滑窗上下文（将最近 context 秒音频拼接后再识别），降低分片断句影响
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


def _project_root() -> Path:
    # .../perception/Wasr/scripts -> .../perception
    return Path(__file__).resolve().parents[2]


def _default_binary(root: Path) -> Path:
    return root / "Wasr" / "third_party" / "whisper.cpp" / "build" / "bin" / "whisper-cli"


def _default_model(root: Path) -> Path:
    # 优先 small-q5_1；否则 tiny
    small = root / "Wasr" / "models" / "ggml-small-q5_1.bin"
    tiny = root / "Wasr" / "models" / "ggml-tiny-q8_0.bin"
    return small if small.is_file() else tiny


def _lcp_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _run(
    video: Path,
    whisper_binary: Path,
    whisper_model: Path,
    language: str,
    threads: int,
    chunk_ms: int,
    context_sec: float,
    realtime: bool,
    prompt: str,
) -> int:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from Wasr.whisper_recognizer import WhisperCppRecognizer  # noqa: E402

    if not video.is_file():
        print(f"[stream] video not found: {video}", file=sys.stderr)
        return 2
    if not whisper_binary.is_file():
        print(f"[stream] whisper-cli not found: {whisper_binary}", file=sys.stderr)
        return 2
    if not whisper_model.is_file():
        print(f"[stream] model not found: {whisper_model}", file=sys.stderr)
        return 2

    sr = 16000
    bytes_per_sample = 2
    chunk_samples = int(sr * (chunk_ms / 1000.0))
    chunk_bytes = chunk_samples * bytes_per_sample

    cfg = {
        "whisper_binary": str(whisper_binary),
        "whisper_model": str(whisper_model),
        "language": language,
        "threads": int(threads),
        "vad_enabled": False,  # 单模块先关 VAD，避免引入 FunASR 依赖
        "prompt": prompt or "",
    }
    rec = WhisperCppRecognizer(cfg)

    ffmpeg_cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "s16le",
        "-",
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    context_max_samples = max(int(context_sec * sr), 0)
    ring: np.ndarray = np.zeros((0,), dtype=np.float32)
    last_full_text = ""

    t_start = time.time()
    chunk_idx = 0
    try:
        assert proc.stdout is not None
        while True:
            raw = proc.stdout.read(chunk_bytes)
            if not raw:
                break

            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if context_max_samples > 0:
                ring = np.concatenate([ring, x])
                if len(ring) > context_max_samples:
                    ring = ring[-context_max_samples:]
                audio_in = ring
            else:
                audio_in = x

            # 这里每个 chunk 都做一次完整推理（whisper.cpp CLI 本身不是真流式）。
            wake, text, _is_speech = rec.inference(audio_in, is_final=False)
            if text:
                # 打印增量（基于前后输出的最长公共前缀）
                k = _lcp_len(last_full_text, text)
                delta = text[k:].strip()
                if delta:
                    peak_mb: Optional[float] = (
                        (rec.last_peak_memory_kb / 1024.0) if rec.last_peak_memory_kb else None
                    )
                    cost_ms: Optional[float] = rec.last_inference_ms
                    meta = []
                    if peak_mb is not None:
                        meta.append(f"peak~{peak_mb:.1f}MB")
                    if cost_ms is not None:
                        meta.append(f"{cost_ms:.0f}ms")
                    meta_s = (" [" + ", ".join(meta) + "]") if meta else ""
                    print(f"[{chunk_idx:04d}] {delta}{meta_s}")
                last_full_text = text

            chunk_idx += 1
            if realtime:
                # 让 wall-clock 看起来像实时输入
                target = t_start + (chunk_idx * (chunk_ms / 1000.0))
                sleep_s = target - time.time()
                if sleep_s > 0:
                    time.sleep(sleep_s)

        # flush final
        if len(ring) > 0:
            _wake, final_text, _is_speech = rec.inference(ring, is_final=True)
            if final_text:
                print("\n[final]")
                print(final_text)
        return 0
    finally:
        rec.release()
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=1.0)
        except Exception:
            pass


def main() -> None:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Whisper.cpp 伪实时转写（从 video.mp4 抽音频）")
    parser.add_argument(
        "--video",
        type=str,
        default="/home/nvidia/MHSEE/QiMing/video/video.mp4",
        help="输入视频路径（从中抽取音频）",
    )
    parser.add_argument(
        "--whisper-binary",
        type=str,
        default=str(_default_binary(root)),
        help="whisper.cpp 的 whisper-cli 路径",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default=str(_default_model(root)),
        help="ggml-*.bin 模型路径",
    )
    parser.add_argument("--language", type=str, default="zh", help="语言代码，例如 zh/en")
    parser.add_argument("--threads", type=int, default=max(os.cpu_count() or 4, 4), help="CPU 线程数")
    parser.add_argument("--chunk-ms", type=int, default=1000, help="每次读取的音频分片时长（毫秒）")
    parser.add_argument(
        "--context-sec",
        type=float,
        default=8.0,
        help="滑窗上下文秒数（0 表示只识别当前 chunk；>0 表示识别最近 N 秒）",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="不 sleep，尽快跑完（用于快速验证链路）",
    )
    parser.add_argument("--prompt", type=str, default="", help="whisper-cli --prompt（可选）")
    args = parser.parse_args()

    code = _run(
        video=Path(args.video),
        whisper_binary=Path(args.whisper_binary),
        whisper_model=Path(args.whisper_model),
        language=args.language,
        threads=args.threads,
        chunk_ms=int(args.chunk_ms),
        context_sec=float(args.context_sec),
        realtime=not bool(args.no_realtime),
        prompt=args.prompt,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()

