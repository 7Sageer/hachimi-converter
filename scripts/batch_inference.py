#!/usr/bin/env python3
"""批量推理：从原曲中截取片段，转换为哈基米风格。

支持两种模型：
- GAN (U-Net + PatchGAN): 使用 inference.py
- Flow Matching: 使用 inference_fm.py

自动检测：若 hachimi_fm_best.pt 存在则默认使用 FM，否则使用 GAN。
可通过 --fm / --gan 强制指定。
"""

import sys
import argparse
import torch
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
MODEL_DIR = Path(__file__).parent.parent / "models"

# 选取风格多样的歌曲，每首取不同偏移
SAMPLES = [
    ("Rubia.wav", 30.0, 15.0),
    ("命运.wav", 20.0, 15.0),
    ("快乐的基米.wav", 10.0, 15.0),
    ("半岛铁哈.wav", 40.0, 15.0),
    ("我的猫香.wav", 15.0, 15.0),
    ("哈在基米前.wav", 60.0, 15.0),
    ("【哈基米】Seven-Tobu.wav", 50.0, 15.0),
    ("甜甜的.wav", 30.0, 15.0),
    ("基战摇.wav", 25.0, 15.0),
    ("Daisy Bell.wav", 5.0, 15.0),
]


def main():
    parser = argparse.ArgumentParser(description="批量推理")
    parser.add_argument("--fm", action="store_true", help="强制使用 Flow Matching")
    parser.add_argument("--gan", action="store_true", help="强制使用 GAN (U-Net)")
    parser.add_argument("--steps", type=int, default=10, help="FM ODE 步数 (default: 10)")
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    # 自动检测模型类型
    fm_exists = (MODEL_DIR / "hachimi_fm_best.pt").exists()
    gan_exists = (MODEL_DIR / "hachimi_unet_best.pt").exists()

    if args.fm:
        use_fm = True
    elif args.gan:
        use_fm = False
    else:
        use_fm = fm_exists  # 默认优先 FM

    if use_fm:
        from inference_fm import convert
        print(f"Using Flow Matching model ({args.steps}-step ODE)")
        if args.device == "auto":
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            dev = args.device
    else:
        from inference import convert
        print("Using GAN (U-Net + PatchGAN) model")

    OUTPUT_DIR.mkdir(exist_ok=True)

    for i, (name, offset, duration) in enumerate(SAMPLES):
        input_path = DATA_DIR / "original" / name
        if not input_path.exists():
            print(f"[skip] {name} not found")
            continue

        out_name = f"{i+1:02d}_{Path(name).stem}.wav"
        output_path = OUTPUT_DIR / out_name
        print(f"\n[{i+1}/{len(SAMPLES)}] {name} (offset={offset}s, dur={duration}s)")

        if use_fm:
            convert(str(input_path), str(output_path), duration=duration,
                    offset=offset, num_steps=args.steps, device=dev)
        else:
            convert(str(input_path), str(output_path), duration=duration, offset=offset)

    print(f"\nDone! Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
