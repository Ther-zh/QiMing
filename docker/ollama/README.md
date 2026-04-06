# Ollama（Docker，与 QiMing 多模态 LLM 解耦）

[`LLM/qwen35.py`](../../LLM/qwen35.py) 使用 Python 包 `ollama`，默认连接 **`OLLAMA_HOST`**（未设置时为 `http://127.0.0.1:11434`）。容器化后请在运行 `main.py` / `sing_test/module_debug.py` 的 shell 中设置：

```bash
export OLLAMA_HOST=http://127.0.0.1:11434
```

## 启动与拉模

```bash
cd /home/nvidia/MHSEE/QiMing/docker/ollama
docker compose up -d
docker exec -it qiming-ollama ollama pull qwen3.5-4b
```

模型名须与 [`config/config.yaml`](../../config/config.yaml) 中 `models.llm.model_name` 一致（当前为 `qwen3.5-4b`）。

**多模态**：`module_debug.py --module llm` 会传图片给 `ollama.chat`；若该模型非视觉模型，会报错。请改用 Ollama 中支持 `images` 的模型名，并同步修改 `config.yaml` 的 `model_name`。

## Jetson / jetson-containers

若官方 `ollama/ollama` 镜像在设备上无法使用 GPU，可参考 [jetson-containers](https://github.com/dusty-nv/jetson-containers) 中 Ollama 相关配方，或在本机直接安装 Ollama deb（同样监听 `11434`）。原则不变：**Python mhsee 环境只装 `pip install ollama` 客户端即可，勿为 LLM 去替换厂商 `torch`。**

## 与宿主 Ollama 并存

若本机已在 `11434` 运行 Ollama，**不要**再映射同一端口；可改 `docker-compose.yml` 中端口为 `11435:11434`，并设置 `export OLLAMA_HOST=http://127.0.0.1:11435`。
