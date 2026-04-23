#!/usr/bin/env bash
# Download a ggml/gguf whisper model (ggerganov/whisper.cpp).
# Usage: download_ggml_model.sh tiny|base|small|... [dest_dir]
# Default dest: /root/Wasr/models
# Override mirror: HF_MIRROR=https://huggingface.co ./download_ggml_model.sh tiny-q8_0
set -euo pipefail
MODEL="${1:?model name required, e.g. tiny or tiny-q8_0}"
DEST="${2:-/root/Wasr/models}"
mkdir -p "${DEST}"
OUT="${DEST}/ggml-${MODEL}.bin"
if [[ -f "${OUT}" ]] && [[ "$(stat -c%s "${OUT}" 2>/dev/null || echo 0)" -gt 262144 ]]; then
  echo "Already present: ${OUT}"
  exit 0
fi
# 国内镜像默认 hf-mirror；直连可设 HF_MIRROR=https://huggingface.co
HF_MIRROR="${HF_MIRROR:-https://hf-mirror.com}"
URL="${HF_MIRROR}/ggerganov/whisper.cpp/resolve/main/ggml-${MODEL}.bin"
echo "Downloading ${URL} -> ${OUT}"
curl -L --fail --retry 3 --connect-timeout 30 -o "${OUT}.part" "${URL}"
mv -f "${OUT}.part" "${OUT}"
echo "Done: ${OUT}"
