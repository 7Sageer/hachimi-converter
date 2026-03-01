#!/usr/bin/env python3
"""批量推理：从原曲中截取片段，转换为哈基米风格。"""

import sys
from pathlib import Path
from inference import convert

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

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
    OUTPUT_DIR.mkdir(exist_ok=True)

    for i, (name, offset, duration) in enumerate(SAMPLES):
        input_path = DATA_DIR / "original" / name
        if not input_path.exists():
            print(f"[skip] {name} not found")
            continue

        out_name = f"{i+1:02d}_{Path(name).stem}.wav"
        output_path = OUTPUT_DIR / out_name
        print(f"\n[{i+1}/{len(SAMPLES)}] {name} (offset={offset}s, dur={duration}s)")
        convert(str(input_path), str(output_path), duration=duration, offset=offset)

    print(f"\nDone! Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
