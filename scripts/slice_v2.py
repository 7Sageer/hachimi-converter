#!/usr/bin/env python3
"""Slice paired audio using chroma-based offset alignment."""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from align_v2 import find_alignment

DATA_DIR = Path(__file__).parent.parent / "data"
PAIRED_DIR = DATA_DIR / "paired"
SR = 22050
HOP = 512
SEGMENT_SEC = 3


def slice_pair(name, orig_path, hach_path):
    """Find alignment offset, then slice both into paired segments."""
    y_orig, _ = librosa.load(orig_path, sr=SR)
    y_hach, _ = librosa.load(hach_path, sr=SR)

    # Find where hachimi aligns in original
    offset_frames, hach_frames, score = find_alignment(orig_path, hach_path)
    if score < 0.3:
        print(f"  [skip] score too low: {score:.3f}")
        return 0

    offset_samples = offset_frames * HOP
    seg_samples = SEGMENT_SEC * SR
    n_segments = len(y_hach) // seg_samples

    count = 0
    for i in range(n_segments):
        hach_start = i * seg_samples
        hach_end = hach_start + seg_samples
        orig_start = offset_samples + hach_start
        orig_end = orig_start + seg_samples

        if hach_end > len(y_hach) or orig_end > len(y_orig):
            break

        sf.write(PAIRED_DIR / f"{name}_orig_{count:03d}.wav",
                 y_orig[orig_start:orig_end], SR)
        sf.write(PAIRED_DIR / f"{name}_hach_{count:03d}.wav",
                 y_hach[hach_start:hach_end], SR)
        count += 1

    return count


def main():
    import shutil
    # Clean old paired data
    if PAIRED_DIR.exists():
        shutil.rmtree(PAIRED_DIR)
    PAIRED_DIR.mkdir(parents=True)

    names = [p.stem for p in (DATA_DIR / "hachimi").glob("*.wav")]
    total = 0

    for name in sorted(names):
        orig_path = DATA_DIR / "original" / f"{name}.wav"
        hach_path = DATA_DIR / "hachimi" / f"{name}.wav"
        if not orig_path.exists() or not hach_path.exists():
            continue

        print(f"\n[{name}]")
        n = slice_pair(name, orig_path, hach_path)
        print(f"  → {n} segments")
        total += n

    print(f"\nTotal: {total} paired segments in {PAIRED_DIR}/")


if __name__ == "__main__":
    main()
