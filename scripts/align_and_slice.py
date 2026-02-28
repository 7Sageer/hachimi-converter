#!/usr/bin/env python3
"""Align and slice paired audio into training segments using chromagram DTW."""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PAIRED_DIR = DATA_DIR / "paired"
PAIRED_DIR.mkdir(exist_ok=True)

SR = 22050
SEGMENT_SEC = 3  # segment length in seconds
HOP = 512


def align_pair(orig_path: Path, hach_path: Path):
    """Use chromagram DTW to find time alignment between original and hachimi."""
    print(f"  Loading audio...")
    y_orig, _ = librosa.load(orig_path, sr=SR)
    y_hach, _ = librosa.load(hach_path, sr=SR)

    print(f"  orig: {len(y_orig)/SR:.1f}s, hach: {len(y_hach)/SR:.1f}s")

    # Compute chroma features
    print(f"  Computing chroma features...")
    chroma_orig = librosa.feature.chroma_cqt(y=y_orig, sr=SR, hop_length=HOP)
    chroma_hach = librosa.feature.chroma_cqt(y=y_hach, sr=SR, hop_length=HOP)

    # DTW alignment
    print(f"  Running DTW alignment...")
    D, wp = librosa.sequence.dtw(chroma_orig, chroma_hach, metric='cosine')
    wp = wp[::-1]  # reverse to get forward path

    return y_orig, y_hach, wp


def slice_aligned(name: str, y_orig, y_hach, wp):
    """Slice aligned audio into fixed-length segments."""
    seg_samples = SEGMENT_SEC * SR
    seg_frames = SEGMENT_SEC * SR // HOP

    # Sample evenly spaced anchor points from warping path
    n_segments = min(len(y_hach) // seg_samples, len(y_orig) // seg_samples)
    if n_segments == 0:
        print(f"  [skip] too short")
        return 0

    count = 0
    # Walk through hachimi version in fixed steps, find corresponding orig position
    for i in range(n_segments):
        hach_frame_start = i * seg_frames
        hach_frame_end = hach_frame_start + seg_frames

        # Find corresponding original frames via warping path
        mask = (wp[:, 1] >= hach_frame_start) & (wp[:, 1] < hach_frame_end)
        if not mask.any():
            continue
        orig_frames = wp[mask, 0]
        orig_frame_start = orig_frames[0]

        # Convert frames to samples
        hach_start = hach_frame_start * HOP
        hach_end = hach_start + seg_samples
        orig_start = orig_frame_start * HOP
        orig_end = orig_start + seg_samples

        if hach_end > len(y_hach) or orig_end > len(y_orig):
            continue

        seg_hach = y_hach[hach_start:hach_end]
        seg_orig = y_orig[orig_start:orig_end]

        # Save
        sf.write(PAIRED_DIR / f"{name}_orig_{count:03d}.wav", seg_orig, SR)
        sf.write(PAIRED_DIR / f"{name}_hach_{count:03d}.wav", seg_hach, SR)
        count += 1

    return count


def main():
    names = [p.stem for p in (DATA_DIR / "hachimi").glob("*.wav")]
    total = 0

    for name in sorted(names):
        orig_path = DATA_DIR / "original" / f"{name}.wav"
        hach_path = DATA_DIR / "hachimi" / f"{name}.wav"

        if not orig_path.exists() or not hach_path.exists():
            print(f"[skip] {name}: missing files")
            continue

        print(f"\n[align] {name}")
        y_orig, y_hach, wp = align_pair(orig_path, hach_path)
        n = slice_aligned(name, y_orig, y_hach, wp)
        print(f"  → {n} segments")
        total += n

    print(f"\nTotal: {total} paired segments in {PAIRED_DIR}/")


if __name__ == "__main__":
    main()
