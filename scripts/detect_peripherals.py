#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描当前连接的外设（摄像头/麦克风），并将结果合并写入 config.yaml。

默认只打印（dry-run），加 --write 才会实际修改 yml。

用法：
  conda run -n mhsee python scripts/detect_peripherals.py
  conda run -n mhsee python scripts/detect_peripherals.py --write
  conda run -n mhsee python scripts/detect_peripherals.py --config config/config.yaml --write
"""

import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _run(cmd: List[str], timeout: float = 2.5) -> Tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        return int(p.returncode), (p.stdout or "")
    except Exception as e:
        return 1, f"{type(e).__name__}: {e}"


def detect_cameras() -> List[Dict[str, Any]]:
    """
    返回摄像头列表：[{device_path, device_id, name}]
    - device_id：按 /dev/videoN 的 N 推导
    - name：优先用 v4l2-ctl 查询，否则退化为 videoN
    """
    cams: List[Dict[str, Any]] = []
    devs = sorted(Path("/dev").glob("video*"))
    for p in devs:
        m = re.match(r"^video(\d+)$", p.name)
        if not m:
            continue
        device_id = int(m.group(1))
        name = p.name
        # 尝试用 v4l2-ctl 拿到更友好的名字
        rc, out = _run(["v4l2-ctl", "-d", str(p), "--all"], timeout=1.5)
        if rc == 0 and out:
            # 常见格式：Driver Info (not using libv4l2):\n\tCard type : xxx
            mm = re.search(r"Card type\s*:\s*(.+)", out)
            if mm:
                name = mm.group(1).strip()
        cams.append(
            {
                "device_path": str(p),
                "device_id": device_id,
                "name": name,
            }
        )
    return cams


def _read_proc_asound_cards() -> List[Dict[str, Any]]:
    cards_file = Path("/proc/asound/cards")
    if not cards_file.is_file():
        return []
    txt = cards_file.read_text(errors="ignore")
    # 示例：
    #  0 [tegrahdaxnx    ]: tegra-hda - tegra-hda-xnx
    #                       NVIDIA Jetson ...
    out: List[Dict[str, Any]] = []
    for line in txt.splitlines():
        m = re.match(r"^\s*(\d+)\s+\[([^\]]+)\]\s*:\s*(.+)$", line)
        if not m:
            continue
        out.append(
            {
                "card_index": int(m.group(1)),
                "card_id": m.group(2).strip(),
                "card_desc": m.group(3).strip(),
            }
        )
    return out


def _read_arecord_list() -> List[Dict[str, Any]]:
    rc, out = _run(["arecord", "-l"], timeout=2.0)
    if rc != 0 or not out:
        return []
    # 非严格解析：抓 card N / device M / name
    devs: List[Dict[str, Any]] = []
    for line in out.splitlines():
        # card 1: Device [USB Audio Device], device 0: USB Audio [USB Audio]
        m = re.search(
            r"card\s+(\d+)\s*:\s*([^\\[]+)\[([^\\]]+)\]\s*,\s*device\s+(\d+)\s*:\s*([^\\[]+)\[([^\\]]+)\]",
            line,
        )
        if not m:
            continue
        devs.append(
            {
                "card_index": int(m.group(1)),
                "card_name": m.group(3).strip(),
                "device_index": int(m.group(4)),
                "device_name": m.group(6).strip(),
            }
        )
    return devs


def detect_microphones() -> List[Dict[str, Any]]:
    """
    返回麦克风/录音设备信息（尽量不依赖额外 python 包）：
    - 优先 /proc/asound/cards
    - 尝试 arecord -l（如果系统有）
    """
    cards = _read_proc_asound_cards()
    arec = _read_arecord_list()
    return [
        {
            "cards": cards,
            "arecord_devices": arec,
        }
    ]


def build_config_blocks(
    cams: List[Dict[str, Any]],
    mic_meta: List[Dict[str, Any]],
    *,
    default_resolution: Tuple[int, int] = (640, 480),
    default_fps: int = 5,
    default_sample_rate: int = 16000,
    default_channels: int = 1,
    default_buffer_size: int = 1024,
) -> Dict[str, Any]:
    cameras_cfg: Dict[str, Any] = {}
    for idx, cam in enumerate(cams, 1):
        key = f"camera{idx}"
        cameras_cfg[key] = {
            "id": cam.get("device_id", idx - 1),
            "device_id": cam.get("device_id", idx - 1),
            "name": cam.get("name") or key,
            "resolution": [int(default_resolution[0]), int(default_resolution[1])],
            "fps": int(default_fps),
            "is_main": True if idx == 1 else False,
            "device_path": cam.get("device_path"),
        }

    # 目前项目 RealInputDevice 只用到 id/name/sample_rate/channels/buffer_size
    microphones_cfg: Dict[str, Any] = {
        "mic1": {
            "id": 0,
            "name": "main",
            "sample_rate": int(default_sample_rate),
            "channels": int(default_channels),
            "buffer_size": int(default_buffer_size),
        }
    }

    peripherals_cfg: Dict[str, Any] = {
        "detected": {
            "cameras": cams,
            "microphones": mic_meta,
        }
    }

    return {
        "cameras": cameras_cfg,
        "microphones": microphones_cfg,
        "peripherals": peripherals_cfg,
    }


def deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "config.yaml"),
        help="config.yaml 路径",
    )
    ap.add_argument("--write", action="store_true", help="写回 config.yaml（默认只打印）")
    ap.add_argument("--fps", type=int, default=5, help="摄像头默认 fps")
    ap.add_argument("--res", type=str, default="640x480", help="摄像头默认分辨率，如 640x480")
    args = ap.parse_args()

    m = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", args.res.lower())
    if not m:
        raise SystemExit(f"--res 格式错误: {args.res!r}，应为 640x480")
    res = (int(m.group(1)), int(m.group(2)))

    cfg_path = Path(args.config).resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"config 文件不存在: {cfg_path}")

    cams = detect_cameras()
    mic_meta = detect_microphones()
    patch = build_config_blocks(cams, mic_meta, default_resolution=res, default_fps=int(args.fps))

    raw = cfg_path.read_text()
    cfg = yaml.safe_load(raw) or {}
    if not isinstance(cfg, dict):
        raise SystemExit("config.yaml 顶层不是 dict，无法写入")

    print("=== Detected cameras ===")
    for c in cams:
        print(f"- {c.get('device_path')} id={c.get('device_id')} name={c.get('name')!r}")
    print("=== Detected audio ===")
    print(yaml.safe_dump(mic_meta, allow_unicode=True, sort_keys=False))

    if not args.write:
        print(f"[DRY-RUN] 未写入：{cfg_path}")
        return 0

    # 写入时尽量保留注释：优先使用 ruamel.yaml round-trip
    try:
        from ruamel.yaml import YAML  # type: ignore
        from ruamel.yaml.comments import CommentedMap  # type: ignore
    except Exception as e:
        raise SystemExit(
            "需要 ruamel.yaml 才能在写入时保留注释/格式。\n"
            "请在 mhsee 环境安装：\n"
            "  conda run -n mhsee python -m pip install ruamel.yaml\n"
            f"当前导入失败：{type(e).__name__}: {e}"
        )

    ry = YAML()
    ry.preserve_quotes = True
    ry.width = 120

    data = ry.load(cfg_path.read_text()) or CommentedMap()
    if not isinstance(data, dict):
        raise SystemExit("config.yaml 顶层不是 mapping，无法写入")

    # 合并写入（保留原有注释；对 cameras/microphones/peripherals 按 patch 覆盖/更新）
    deep_merge(data, patch)

    tmp = cfg_path.with_suffix(".yaml.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        ry.dump(data, f)
    os.replace(tmp, cfg_path)
    print(f"[OK] 已写入并保留注释：{cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

