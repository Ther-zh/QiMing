可将为本机 CPU 架构预先编译好的 whisper-cli 放在本目录，命名为 whisper-cli，
并在 MHSEE config 中设置：

  whisper_binary: "/.../Wasr/bin/whisper-cli"

若使用方式 B（不携带 whisper.cpp 源码），请用 ldd whisper-cli 确认所需动态库在边缘设备上可用。
