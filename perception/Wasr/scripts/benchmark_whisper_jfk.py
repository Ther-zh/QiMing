#!/usr/bin/env python3
"""
One-shot whisper.cpp transcription + peak memory log (uses repo samples/jfk.wav).
Requires: ggml model at WHISPER_MODEL env or argv[1], whisper-cli built.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/root")

from Wasr.whisper_runner import transcribe_file  # noqa: E402


def main() -> None:
    model = os.environ.get("WHISPER_MODEL", sys.argv[1] if len(sys.argv) > 1 else "")
    binary = os.environ.get(
        "WHISPER_BINARY",
        "/root/Wasr/third_party/whisper.cpp/build/bin/whisper-cli",
    )
    wav = "/root/Wasr/third_party/whisper.cpp/samples/jfk.wav"
    if not model or not os.path.isfile(model):
        print("Set WHISPER_MODEL or pass path to ggml-*.bin (file must exist).", file=sys.stderr)
        raise SystemExit(2)
    if not os.path.isfile(wav):
        print("Missing jfk.wav sample.", file=sys.stderr)
        raise SystemExit(2)
    text, peak_kb = transcribe_file(
        binary,
        model,
        wav,
        language="en",
        threads=4,
    )
    mb = (peak_kb or 0) / 1024.0
    print("text:", text[:200])
    print(f"peak_memory_mb: {mb:.2f}")
    # tiny 约 <300MB；small-q5_1 等在 CPU 上常 500–700MB，可用环境变量调整
    max_mb = float(os.environ.get("WASR_PEAK_MB_MAX", "800"))
    if mb > max_mb:
        raise SystemExit(f"peak {mb:.2f} MB exceeds WASR_PEAK_MB_MAX={max_mb}")
    print(f"benchmark_whisper_jfk: OK (<= {max_mb} MB)")


if __name__ == "__main__":
    main()
