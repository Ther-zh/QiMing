#!/usr/bin/env python3
"""
分模块视频调试：以 video/ 或 --video 的 MP4 为输入，分别输出
  VDA 伪彩色深度视频、ASR 文本、多模态 LLM 回答、YOLO 画框视频。

请在项目根目录执行:
  python sing_test/module_debug.py --module all
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)


def _load_app_config():
    from utils.config_loader import config_loader

    return config_loader.get_config()


def resolve_video_paths(args: argparse.Namespace, config: dict) -> list[str]:
    if args.video:
        p = os.path.abspath(args.video)
        if not os.path.isfile(p):
            raise SystemExit(f"视频文件不存在: {p}")
        return [p]
    pattern = os.path.join(ROOT, "video", "*.mp4")
    found = sorted(glob.glob(pattern))
    if found:
        return found
    rel = (config.get("simulation") or {}).get("video_paths", {}).get("camera1")
    if rel:
        p = os.path.join(ROOT, rel)
        if os.path.isfile(p):
            return [p]
    raise SystemExit(
        "未找到输入视频：请将 MP4 放入 video/ 目录，或使用 --video 指定文件。"
    )


def extract_audio_wav_16k_mono(video_path: str) -> np.ndarray:
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            tmp,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        audio, sr = sf.read(tmp)
        if sr != 16000:
            print(f"[ASR] 警告: 期望 16kHz，得到 {sr}")
        return audio.astype(np.float32)
    finally:
        if os.path.isfile(tmp):
            os.unlink(tmp)


def depth_to_colormap_bgr(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    dmin, dmax = float(d.min()), float(d.max())
    if dmax - dmin < 1e-8:
        norm = np.zeros_like(d, dtype=np.uint8)
    else:
        norm = ((d - dmin) / (dmax - dmin + 1e-8) * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def run_vda(video_path: str, out_dir: str, config: dict, side_by_side: bool, max_frames: int | None = None) -> str | None:
    from perception.vda.vda_depth import VDADepthEstimator

    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_vda_depth.mp4")
    vda_cfg = dict((config.get("models") or {}).get("vda") or {})
    vda_max_side = int(vda_cfg.get("vda_max_input_side", 384))

    def _resize_for_vda(fr):
        ih, iw = fr.shape[:2]
        m = max(ih, iw)
        if m <= vda_max_side:
            return fr, fr.shape[1], fr.shape[0]
        sc = vda_max_side / m
        nw, nh = int(iw * sc), int(ih * sc)
        return cv2.resize(fr, (nw, nh)), nw, nh

    try:
        est = VDADepthEstimator(vda_cfg)
    except Exception as e:
        print(f"[VDA] 跳过（无法加载模型）: {e}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[VDA] 无法打开视频: {video_path}")
        est.release()
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    m0 = max(h0, w0)
    if m0 > vda_max_side:
        sc0 = vda_max_side / m0
        w, h = int(w0 * sc0), int(h0 * sc0)
    else:
        w, h = w0, h0
    ow = w * 2 if side_by_side else w
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (ow, h))
    if not writer.isOpened():
        est.release()
        cap.release()
        print(f"[VDA] VideoWriter 打开失败: {out_path}")
        return None

    frame_i = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_small, _, _ = _resize_for_vda(frame)
            depth = est.inference(frame_small)
            colored = depth_to_colormap_bgr(depth)
            if colored.shape[:2] != (h, w):
                colored = cv2.resize(colored, (w, h))
            if side_by_side:
                left = cv2.resize(frame, (w, h))
                out_f = np.hstack([left, colored])
            else:
                out_f = colored
            writer.write(out_f)
            frame_i += 1
            if max_frames is not None and frame_i >= max_frames:
                print(f"[VDA] 已达 --vda-max-frames={max_frames}，提前结束")
                break
            if frame_i % 30 == 0:
                print(f"[VDA] 已处理 {frame_i} 帧")
    finally:
        writer.release()
        cap.release()
        est.release()
    print(f"[VDA] 已写入: {out_path}（共 {frame_i} 帧）")
    return out_path


def run_asr(video_path: str, out_dir: str, config: dict) -> str | None:
    from perception.asr.funasr_asr import FunASRRecognizer

    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_asr.txt")
    asr_cfg = dict((config.get("models") or {}).get("asr") or {})
    try:
        audio = extract_audio_wav_16k_mono(video_path)
    except subprocess.CalledProcessError as e:
        print(f"[ASR] ffmpeg 提取音频失败: {e}")
        return None
    if len(audio) < 4800:
        print("[ASR] 音频过短，跳过")
        return None

    asr = None
    try:
        asr = FunASRRecognizer(asr_cfg)
        text = asr.model.recognize(audio, clean_output=True)
    except Exception as e:
        print(f"[ASR] 加载或识别失败: {e}")
        return None
    finally:
        if asr is not None:
            try:
                asr.release()
            except Exception:
                pass

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
    print(f"[ASR] 识别结果:\n{text.strip()}\n[ASR] 已写入: {out_path}")
    return out_path


def sample_frame_pil(video_path: str, position: str = "middle"):
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if position == "first":
        idx = 0
    elif position == "middle":
        idx = max(0, n // 2) if n > 0 else 0
    else:
        idx = max(0, n // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, bgr = cap.read()
    cap.release()
    if not ret or bgr is None:
        raise RuntimeError("无法读取抽帧")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def run_llm(
    video_path: str,
    out_dir: str,
    config: dict,
    prompt: str,
    frame_mode: str,
    model_name_override: str | None = None,
) -> str | None:
    from LLM.qwen35 import Qwen35Ollama

    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_llm.txt")
    llm_cfg = (config.get("models") or {}).get("llm") or {}
    model_name = model_name_override or llm_cfg.get("model_name", "qwen3.5-4b")

    try:
        pil = sample_frame_pil(video_path, position=frame_mode)
    except Exception as e:
        print(f"[LLM] 抽帧失败: {e}")
        return None

    llm = Qwen35Ollama(model_name=model_name)
    try:
        answer = llm.generate(
            text=prompt,
            image=pil,
            max_tokens=512,
        )
    finally:
        pass

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(answer + "\n")
    print(f"[LLM] 回答已写入: {out_path}")
    return out_path


def run_yolo(video_path: str, out_dir: str, config: dict) -> str | None:
    from ultralytics import YOLO
    import logging

    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    stem = os.path.splitext(os.path.basename(video_path))[0]
    yolo_cfg = (config.get("models") or {}).get("yolo") or {}
    weights = yolo_cfg.get("model_path")
    conf = float(yolo_cfg.get("conf_threshold", 0.25))
    if not weights or not os.path.isfile(weights):
        print(f"[YOLO] 权重不存在: {weights}")
        return None

    run_name = f"{stem}_yolo"
    try:
        model = YOLO(weights)
    except Exception as e:
        print(f"[YOLO] 加载失败: {e}")
        return None

    device = 0 if __import__("torch").cuda.is_available() else "cpu"
    save_dir = None
    try:
        # stream=True：逐帧推理并写出，避免长视频在 RAM 中累积全部 Results（Jetson 上易 OOM 被 kill）
        for r in model.predict(
            source=video_path,
            conf=conf,
            save=True,
            project=out_dir,
            name=run_name,
            exist_ok=True,
            device=device,
            stream=True,
            verbose=False,
        ):
            save_dir = getattr(r, "save_dir", save_dir)
    except Exception as e:
        print(f"[YOLO] 推理失败: {e}")
        return None
    base = os.path.join(out_dir, run_name)
    print(f"[YOLO] 结果目录: {save_dir or base}（Ultralytics 默认保存标注视频/帧）")
    return save_dir or base


def main():
    parser = argparse.ArgumentParser(description="QiMing 分模块视频调试")
    parser.add_argument(
        "--module",
        choices=("vda", "asr", "llm", "yolo", "all"),
        default="all",
        help="运行的模块",
    )
    parser.add_argument("--video", type=str, default=None, help="单个视频文件路径")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(ROOT, "output", "module_debug"),
        help="输出根目录",
    )
    parser.add_argument(
        "--vda-side-by-side",
        action="store_true",
        help="VDA 输出为左原图右深度伪彩色",
    )
    parser.add_argument(
        "--llm-prompt",
        type=str,
        default="请简要描述这张导盲相机画面中的场景、主要障碍物与可通行方向。",
        help="多模态 LLM 文本提示",
    )
    parser.add_argument(
        "--llm-frame",
        choices=("middle", "first"),
        default="middle",
        help="LLM 抽帧位置",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="覆盖 config 中的 Ollama 模型名（多模态需支持 images，如 moondream:latest）",
    )
    parser.add_argument(
        "--vda-max-frames",
        type=int,
        default=None,
        help="VDA 最多处理帧数（默认全片；Jetson 上全片极慢，调试可设 30～120）",
    )
    args = parser.parse_args()

    config = _load_app_config()
    videos = resolve_video_paths(args, config)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[module_debug] 项目根: {ROOT}")
    print(f"[module_debug] 输出目录: {out_dir}")
    print(f"[module_debug] 输入视频: {videos}")

    modules = ["vda", "asr", "llm", "yolo"] if args.module == "all" else [args.module]

    for vp in videos:
        print(f"\n========== 处理: {vp} ==========")
        if "vda" in modules:
            run_vda(
                vp,
                out_dir,
                config,
                side_by_side=args.vda_side_by_side,
                max_frames=args.vda_max_frames,
            )
        if "asr" in modules:
            run_asr(vp, out_dir, config)
        if "llm" in modules:
            run_llm(
                vp,
                out_dir,
                config,
                args.llm_prompt,
                args.llm_frame,
                model_name_override=args.llm_model,
            )
        if "yolo" in modules:
            run_yolo(vp, out_dir, config)


if __name__ == "__main__":
    main()
