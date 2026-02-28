#!/usr/bin/env python3
"""Find where hachimi version aligns within the original song using chroma cross-correlation."""

import librosa
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
SR = 22050
HOP = 512


def find_alignment(orig_path, hach_path):
    """Find the best offset of hachimi within original using chroma cross-correlation."""
    y_orig, _ = librosa.load(orig_path, sr=SR)
    y_hach, _ = librosa.load(hach_path, sr=SR)

    print(f"  orig: {len(y_orig)/SR:.1f}s, hach: {len(y_hach)/SR:.1f}s")

    # Compute chroma
    chroma_orig = librosa.feature.chroma_cqt(y=y_orig, sr=SR, hop_length=HOP)
    chroma_hach = librosa.feature.chroma_cqt(y=y_hach, sr=SR, hop_length=HOP)

    # Normalize
    chroma_orig = chroma_orig / (np.linalg.norm(chroma_orig, axis=0, keepdims=True) + 1e-8)
    chroma_hach = chroma_hach / (np.linalg.norm(chroma_hach, axis=0, keepdims=True) + 1e-8)

    # Sliding window cross-correlation
    hach_len = chroma_hach.shape[1]
    orig_len = chroma_orig.shape[1]

    if hach_len >= orig_len:
        print(f"  Hachimi is longer than original, no offset search needed")
        return 0, min(orig_len, hach_len), 1.0

    best_score = -1
    best_offset = 0

    for offset in range(orig_len - hach_len + 1):
        window = chroma_orig[:, offset:offset + hach_len]
        score = np.sum(window * chroma_hach) / hach_len
        if score > best_score:
            best_score = score
            best_offset = offset

    offset_sec = best_offset * HOP / SR
    duration_sec = hach_len * HOP / SR
    print(f"  Best match: offset={offset_sec:.1f}s, duration={duration_sec:.1f}s, score={best_score:.3f}")

    return best_offset, hach_len, best_score


def main():
    names = [p.stem for p in (DATA_DIR / "hachimi").glob("*.wav")]

    for name in sorted(names):
        orig_path = DATA_DIR / "original" / f"{name}.wav"
        hach_path = DATA_DIR / "hachimi" / f"{name}.wav"

        if not orig_path.exists() or not hach_path.exists():
            continue

        print(f"\n[{name}]")
        offset, length, score = find_alignment(orig_path, hach_path)

        if score < 0.3:
            print(f"  ⚠️ LOW SCORE — probably wrong original or heavily rearranged")
        elif score < 0.5:
            print(f"  ⚠️ Mediocre alignment — may need manual check")
        else:
            print(f"  ✅ Good alignment")


if __name__ == "__main__":
    main()
