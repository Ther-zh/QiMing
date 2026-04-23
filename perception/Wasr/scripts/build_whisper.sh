#!/usr/bin/env bash
# Build whisper.cpp CPU binary (Release). Requires: cmake, g++, make.
# Run from anywhere; paths are relative to this script's parent (Wasr root).
set -euo pipefail
WASR_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${WASR_ROOT}/third_party/whisper.cpp"
if [[ ! -d "${SRC}/CMakeLists.txt" && ! -f "${SRC}/CMakeLists.txt" ]]; then
  echo "Clone whisper.cpp first, e.g.: git clone https://github.com/ggerganov/whisper.cpp.git ${SRC}"
  exit 1
fi
cmake -S "${SRC}" -B "${SRC}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${SRC}/build" -j"$(nproc)"
BIN="${SRC}/build/bin/whisper-cli"
if [[ -x "${BIN}" ]]; then
  echo "OK: ${BIN}"
else
  echo "Build finished but ${BIN} not found; check build/bin/"
  exit 1
fi
