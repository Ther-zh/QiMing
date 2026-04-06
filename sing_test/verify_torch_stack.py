#!/usr/bin/env python3
"""Jetson mhsee：打印 torchvision / torchaudio 是否可用（与 JETSON_PYTORCH.md 自检配套）。"""
from __future__ import annotations

import importlib.metadata as md


def main() -> None:
    try:
        v = md.version("torchvision")
        print("torchvision (pip metadata):", v)
    except md.PackageNotFoundError:
        print("torchvision: 无 pip 元数据（Jetson 上常见，可接受）")
    try:
        import torchvision

        print("torchvision (import):", getattr(torchvision, "__version__", "<no __version__>"))
    except Exception as e:
        print("torchvision (import):", e)
    try:
        import torchaudio

        print("torchaudio:", torchaudio.__version__)
    except Exception as e:
        print("torchaudio:", e)


if __name__ == "__main__":
    main()
