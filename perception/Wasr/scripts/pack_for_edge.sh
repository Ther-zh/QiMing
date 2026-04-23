#!/usr/bin/env bash
# 打包 Wasr 目录供拷贝到边缘设备（默认排除 whisper.cpp 的 .git 与 build，减小体积）。
# 用法：bash pack_for_edge.sh [输出路径]
# 若要在边缘本机编译 whisper，请改用 --full-whisper 保留源码但删 .git
set -euo pipefail
WASR_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-${WASR_ROOT}/Wasr_edge_bundle.tar.gz}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Staging from $WASR_ROOT -> $TMP/Wasr"
mkdir -p "$TMP/Wasr"
rsync -a \
  --exclude='third_party/whisper.cpp/.git' \
  --exclude='third_party/whisper.cpp/build' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  "$WASR_ROOT/" "$TMP/Wasr/"

echo "Note: models/*.bin and whisper-cli binary are NOT included by default."
echo "Add ggml-*.bin under Wasr/models/ and/or copy build/bin/whisper-cli before deploy."

tar -C "$TMP" -czf "$OUT" Wasr
echo "Created: $OUT ($(du -h "$OUT" | cut -f1))"
