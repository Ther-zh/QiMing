#!/usr/bin/env python3
"""Sanity-check /proc VmHWM polling on a short-lived child (no whisper model required)."""
from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Wasr.memory_probe import read_proc_status_kb, run_monitor_thread  # noqa: E402


def main() -> None:
    proc = subprocess.Popen(
        ["/bin/sleep", "0.25"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    stop_ev, th, peak_holder = run_monitor_thread(proc.pid)
    proc.wait()
    stop_ev.set()
    th.join(timeout=2.0)
    hwm, rss = read_proc_status_kb(proc.pid)
    peak = int(peak_holder[0]) if peak_holder else 0
    print(f"peak_while_running_kb={peak} final_hwm={hwm} final_rss={rss}")
    if peak <= 0 and hwm <= 0 and rss <= 0:
        raise SystemExit("unexpected: no memory read from /proc")
    print("verify_peak_probe: OK")


if __name__ == "__main__":
    main()
