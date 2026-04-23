"""Poll child process peak resident / high watermark memory from /proc (Linux)."""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Tuple

_VMHWM_RE = re.compile(r"^VmHWM:\s+(\d+)\s+kB", re.MULTILINE)
_VMRSS_RE = re.compile(r"^VmRSS:\s+(\d+)\s+kB", re.MULTILINE)


def read_proc_status_kb(pid: int) -> Tuple[int, int]:
    """
    Return (vm_hwm_kb, vm_rss_kb). Missing file or parse failure -> (0, 0).
    VmHWM is peak RSS since process start (Linux).
    """
    path = f"/proc/{pid}/status"
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return 0, 0
    hwm_m = _VMHWM_RE.search(text)
    rss_m = _VMRSS_RE.search(text)
    hwm = int(hwm_m.group(1)) if hwm_m else 0
    rss = int(rss_m.group(1)) if rss_m else 0
    return hwm, rss


def peak_while_running(pid: int, poll_interval_s: float = 0.05) -> int:
    """
    Poll until pid disappears; return max VmHWM (fallback max VmRSS) seen in kB.
    """
    peak = 0
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            break
        hwm, rss = read_proc_status_kb(pid)
        cur = hwm if hwm > 0 else rss
        if cur > peak:
            peak = cur
        time.sleep(poll_interval_s)
    # final read if process just exited
    hwm, rss = read_proc_status_kb(pid)
    cur = hwm if hwm > 0 else rss
    if cur > peak:
        peak = cur
    return peak


def monitor_process_peak(
    pid: int,
    stop_event: threading.Event,
    poll_interval_s: float = 0.05,
) -> int:
    """Background poll until stop_event is set; return max kB seen."""
    peak = 0
    while not stop_event.is_set():
        hwm, rss = read_proc_status_kb(pid)
        cur = hwm if hwm > 0 else rss
        if cur > peak:
            peak = cur
        time.sleep(poll_interval_s)
    hwm, rss = read_proc_status_kb(pid)
    cur = hwm if hwm > 0 else rss
    if cur > peak:
        peak = cur
    return peak


def run_monitor_thread(pid: int) -> Tuple[threading.Event, threading.Thread, list]:
    """
    Start a daemon thread that updates result_list[0] with peak kB until stopped.
    Caller should set stop_event before join.
    """
    stop_event = threading.Event()
    result: list = [0]

    def _run() -> None:
        peak = 0
        while not stop_event.is_set():
            hwm, rss = read_proc_status_kb(pid)
            cur = hwm if hwm > 0 else rss
            if cur > peak:
                peak = cur
            time.sleep(0.05)
        hwm, rss = read_proc_status_kb(pid)
        cur = hwm if hwm > 0 else rss
        if cur > peak:
            peak = cur
        result[0] = peak

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return stop_event, th, result
