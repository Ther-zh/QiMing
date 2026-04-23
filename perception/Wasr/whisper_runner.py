"""Run whisper-cli on 16 kHz mono PCM; optional peak-RSS monitoring for the child process."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import wave
from typing import Any, List, Optional, Tuple

import numpy as np

from .memory_probe import run_monitor_thread

logger = logging.getLogger(__name__)


def _float_to_int16(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def write_wav_mono16(path: str, audio: np.ndarray, sample_rate: int = 16000) -> None:
    if audio.dtype != np.int16:
        pcm = _float_to_int16(audio)
    else:
        pcm = audio
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def transcribe_file(
    whisper_binary: str,
    model_path: str,
    wav_path: str,
    language: str = "zh",
    threads: int = 4,
    extra_args: Optional[List[str]] = None,
    no_flash_attn: bool = True,
) -> Tuple[str, Optional[int]]:
    """
    Run whisper-cli, read transcript from -otxt output file.
    Returns (text, peak_memory_kb) where peak is whisper child VmHWM/RSS max, or None if not measured.
    """
    extra_args = extra_args or []
    out_stem = tempfile.mktemp(prefix="wasr_whisper_", dir="/tmp")
    cmd: List[str] = [
        whisper_binary,
        "-m",
        model_path,
        "-f",
        wav_path,
        "-l",
        language,
        "-nt",
        "-otxt",
        "-of",
        out_stem,
        "-t",
        str(threads),
        "-np",
    ]
    if no_flash_attn:
        cmd.append("-nfa")
    cmd.extend(extra_args)

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", str(threads))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    stop_ev, mon_th, peak_holder = run_monitor_thread(proc.pid)
    try:
        proc.communicate()
    finally:
        stop_ev.set()
        mon_th.join(timeout=2.0)

    peak_kb = int(peak_holder[0]) if peak_holder else None
    txt_path = out_stem + ".txt"
    text = ""
    try:
        if os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
    finally:
        for p in (txt_path, out_stem + ".vtt", out_stem + ".srt"):
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass

    if proc.returncode != 0:
        logger.warning("whisper-cli exited with code %s", proc.returncode)

    return text, peak_kb


def transcribe_numpy(
    whisper_binary: str,
    model_path: str,
    audio: np.ndarray,
    sample_rate: int = 16000,
    language: str = "zh",
    threads: int = 4,
    extra_args: Optional[List[str]] = None,
) -> Tuple[str, Optional[int]]:
    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="wasr_")
    os.close(fd)
    try:
        write_wav_mono16(wav_path, audio, sample_rate)
        return transcribe_file(
            whisper_binary,
            model_path,
            wav_path,
            language=language,
            threads=threads,
            extra_args=extra_args,
        )
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass
