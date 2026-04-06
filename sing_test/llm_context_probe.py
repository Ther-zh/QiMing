#!/usr/bin/env python3
"""
Ollama 上下文与兜底探测：短 prompt vs 长 prompt，观察 raw 长度与 [LLM_META] 行。
用于二分排查 num_ctx 过小、think 未关、非视觉模型等问题。

用法（项目根）:
  conda activate mhsee
  python sing_test/llm_context_probe.py
  python sing_test/llm_context_probe.py --no-image   # 仅文本
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-image", action="store_true", help="不做多模态，仅 generate 文本")
    args = parser.parse_args()

    from utils.config_loader import config_loader
    from LLM.qwen35 import Qwen35Ollama

    cfg = config_loader.get_config()
    llm_cfg = (cfg.get("models") or {}).get("llm") or {}
    name = llm_cfg.get("model_name", "qwen3.5-4b")
    _think = llm_cfg.get("ollama_think", False)
    _opts = dict(llm_cfg.get("ollama_options") or {})
    _kw: dict = dict(think=_think, ollama_options=_opts)
    if llm_cfg.get("fallback_phrase"):
        _kw["fallback_phrase"] = str(llm_cfg["fallback_phrase"])

    try:
        import ollama

        info = ollama.show(model=name)
        if hasattr(info, "model_dump"):
            info = info.model_dump()
        print("[probe] ollama.show 原始字段键:", list(info.keys()) if isinstance(info, dict) else type(info))
        if isinstance(info, dict):
            det = info.get("details") or info.get("model_info") or {}
            if isinstance(det, dict):
                fam = det.get("family") or det.get("architecture")
                print("[probe] model details.family/architecture:", fam)
            caps = info.get("capabilities") or info.get("families")
            print("[probe] capabilities / families:", caps)
    except Exception as e:
        print("[probe] ollama.show 失败（服务未起或模型未拉取）:", e)

    llm = Qwen35Ollama(model_name=name, **_kw)
    short_p = "用一句话说你好。"
    long_p = (
        "你是导盲助手。\n"
        + "## 环境信息\n检测到的目标：无\n\n## 用户问题\n前面安全吗？\n\n"
        + "## 输出要求\n1. 直接给出具体建议\n2. 控制在50字以内\n3. 必须用中文\n\n"
        + "请直接输出简洁的导航建议：" * 3
    )

    def run(label: str, text: str, image) -> None:
        print(f"\n=== {label} ===")
        gkw = {"text": text, "image": image, "max_tokens": int(llm_cfg.get("max_generate_tokens", 256))}
        if _opts.get("num_ctx") is not None:
            gkw["num_ctx"] = int(_opts["num_ctx"])
        if _opts.get("num_predict") is not None:
            gkw["num_predict"] = int(_opts["num_predict"])
        out = llm.generate(**gkw)
        fb = str(llm_cfg.get("fallback_phrase") or "")
        same_as_fb = fb and out.strip() == fb.strip()
        print(f"[probe] 回复长度={len(out)} 与配置兜底完全相同={same_as_fb}")
        print(f"[probe] 回复预览: {out[:120]!r}")

    image = None
    if not args.no_image:
        from PIL import Image

        image = Image.new("RGB", (320, 240), color=(80, 120, 60))

    run("short_prompt", short_p, image)
    run("long_prompt", long_p, image)


if __name__ == "__main__":
    main()
